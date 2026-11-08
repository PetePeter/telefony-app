"""Speech for calls. TTS backends run as separate programs (Piper is GPL-3.0, so it is never imported); STT is
whisper.cpp's server (GPU via its Vulkan build) and/or faster-whisper on the CPU."""

import asyncio
import atexit
import concurrent.futures
import importlib.util
import io
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, Protocol

import numpy as np

from telefony_app.bt.audio import resample, shape_for_phone

log = logging.getLogger(__name__)
PIPER_RATE = 22050  # medium voices; the model's .onnx.json says for sure


class Tts(Protocol):
    async def synthesize(self, text: str, rate: int) -> np.ndarray: ...


class PiperTts:
    """The `piper` executable (pip install piper-tts, or a release binary), fed text on stdin."""

    def __init__(self, executable: str, model: str) -> None:
        self.executable, self.model = executable, model
        voice = Path(model + ".json")
        audio = json.loads(voice.read_text(encoding="utf-8")).get("audio", {}) if voice.exists() else {}
        self.rate = int(audio.get("sample_rate", PIPER_RATE))

    async def synthesize(self, text: str, rate: int) -> np.ndarray:
        process = await asyncio.create_subprocess_exec(
            self.executable, "--model", self.model, "--output-raw",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=os.environ | {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},  # piper reads stdin as UTF-8 whatever the console
        )
        raw, err = await process.communicate((one_line(text) + "\n").encode("utf-8"))
        if process.returncode:
            detail = err.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
            raise RuntimeError(f"piper exited with {process.returncode}: {detail[0]}")
        return shape_for_phone(resample(np.frombuffer(raw, dtype=np.int16), self.rate, rate), rate)


class ResidentPiperTts:
    """Piper in this process (pip install piper-tts): the voice loads once, in the background from startup,
    so each phrase costs ~0.1 s instead of a ~13 s process start and model load."""

    def __init__(self, model: str, load: Callable[[str], Any] | None = None) -> None:
        if load is None:
            from piper import PiperVoice  # pyright: ignore[reportMissingImports]  optional: pip install piper-tts

            load = PiperVoice.load
        self._voice = concurrent.futures.ThreadPoolExecutor(1, "piper-load").submit(load, model)

    async def synthesize(self, text: str, rate: int) -> np.ndarray:
        try:
            voice = await asyncio.wrap_future(self._voice)
        except Exception as failed:  # a broken model fails every phrase; FallbackTts takes over
            raise RuntimeError(f"piper voice did not load: {failed}") from None
        raw = await asyncio.to_thread(lambda: b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(one_line(text))))
        return shape_for_phone(resample(np.frombuffer(raw, dtype=np.int16), voice.config.sample_rate, rate), rate)


def one_line(text: str) -> str:
    """Piper speaks each stdin line separately and chokes on control characters: send one clean line."""
    return re.sub(r"\s+", " ", re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)).strip()


class FallbackTts:
    """The configured voice, and a backup (Windows' voice) when it fails, so the caller never hears silence."""

    def __init__(self, primary: Tts, backup: Tts) -> None:
        self.primary, self.backup = primary, backup

    async def synthesize(self, text: str, rate: int) -> np.ndarray:
        try:
            return await self.primary.synthesize(text, rate)
        except (OSError, RuntimeError) as failed:
            log.warning("%s failed (%s) on %d chars; using the backup voice", type(self.primary).__name__, failed, len(text))
            return await self.backup.synthesize(text, rate)


class SapiSynthesizer(Protocol):
    async def synthesize(self, text: str, path: Path, voice_name: str | None) -> bool: ...


class PowerShellSapiSynthesizer:
    """Render through Windows SAPI; return whether voice selection used the default fallback."""

    async def synthesize(self, text: str, path: Path, voice_name: str | None) -> bool:
        script = (
            "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "if ($env:TELEFONY_APP_SAPI_VOICE_CONFIGURED -eq '1') { "
            "$defaultVoiceName = $s.Voice.Name; "
            "try { $s.SelectVoice($env:TELEFONY_APP_SAPI_VOICE) } "
            "catch { $s.SelectVoice($defaultVoiceName); [Console]::Error.WriteLine('__TELEFONY_SAPI_VOICE_FALLBACK__') } }; "
            f"$s.SetOutputToWaveFile('{path}'); $s.Speak([Console]::In.ReadToEnd()); $s.Dispose()"
        )
        env = os.environ.copy()
        env.pop("TELEFONY_APP_SAPI_VOICE", None)
        env.pop("TELEFONY_APP_SAPI_VOICE_CONFIGURED", None)
        if voice_name is not None:
            env["TELEFONY_APP_SAPI_VOICE"] = voice_name
            env["TELEFONY_APP_SAPI_VOICE_CONFIGURED"] = "1"
        process = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-Command", script,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        _, stderr = await process.communicate(text.encode())
        return b"__TELEFONY_SAPI_VOICE_FALLBACK__" in (stderr or b"")


class SapiTts:
    """Windows' built-in voice via System.Speech, through PowerShell."""

    def __init__(self, voice_name: str | None = None, synthesizer: SapiSynthesizer | None = None) -> None:
        self.voice_name = voice_name
        self.synthesizer = synthesizer if synthesizer is not None else PowerShellSapiSynthesizer()
        self._warned_missing_voice = False

    async def synthesize(self, text: str, rate: int) -> np.ndarray:
        fd, name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        path = Path(name)
        try:
            if await self.synthesizer.synthesize(text, path, self.voice_name) and not self._warned_missing_voice:
                log.warning("SAPI voice %r was not found; using the default Windows voice", self.voice_name)
                self._warned_missing_voice = True
            with wave.open(str(path)) as wav:
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
                source_rate = wav.getframerate()
        finally:
            path.unlink(missing_ok=True)
        return shape_for_phone(resample(pcm, source_rate, rate), rate)


def make_tts(kind: str, piper_model: str | None, sapi_voice: str | None = None) -> Tts | None:
    """auto: Piper if installed and a model is configured, else Windows SAPI, else no speech."""
    # A venv's Scripts folder is on PATH only when activated; pip install piper-tts puts piper there.
    # The Windows release ships it in a piper/ folder beside telefony-app.exe.
    here = Path(sys.executable).parent
    piper = shutil.which("piper") or shutil.which("piper", path=f"{here}{os.pathsep}{here / 'piper'}")
    if kind in ("auto", "piper") and piper_model and (importlib.util.find_spec("piper") or piper):
        primary = ResidentPiperTts(piper_model) if importlib.util.find_spec("piper") else PiperTts(str(piper), piper_model)
        return FallbackTts(primary, SapiTts(sapi_voice)) if sys.platform == "win32" else primary
    if kind == "piper":
        raise RuntimeError("voice.tts = piper needs the piper executable (on PATH or in this venv) and voice.piper_model")
    if kind in ("auto", "sapi") and sys.platform == "win32":
        return SapiTts(sapi_voice)
    return None


class Stt(Protocol):
    def transcribe(self, pcm16k: np.ndarray) -> str: ...


class WhisperStt:
    def __init__(self, model: str) -> None:
        from faster_whisper import WhisperModel  # pyright: ignore[reportMissingImports]  optional dependency: pip install telefony-app[stt]

        self.model = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=min(8, os.cpu_count() or 4))

    def transcribe(self, pcm16k: np.ndarray) -> str:
        segments, _ = self.model.transcribe(pcm16k.astype(np.float32) / 32768, language="en")
        return spoken_text(segments)


# Whisper encodes 30 s (1500 frames) per call however short the audio; 500 frames covers 10 s, ~3x faster.
# Audio past it is dropped unheard, so no chunk may be longer: BluetoothPhone sends at most a window plus its tail.
_AUDIO_CTX = 500


class WhisperCppStt:
    """whisper.cpp's whisper-server, started once and kept running so the model stays loaded (on the GPU with
    the Vulkan build); each utterance is POSTed to /inference as a 16 kHz WAV. It warms up from startup, so the
    first call does not wait for the model to load."""

    def __init__(self, server: str, model: str, ready_s: float = 60) -> None:
        self._command, self._ready_s = [server, "-m", model, "--host", "127.0.0.1", "-l", "en", "-ac", str(_AUDIO_CTX)], ready_s
        self._process: subprocess.Popen | None = None
        self._port = 0
        self._lock = threading.Lock()
        self._closed = False
        atexit.register(self.close)
        self._ready = concurrent.futures.ThreadPoolExecutor(1, "whisper-cpp-start").submit(self._start)

    def _start(self) -> None:
        """Launch on a free port and wait for an answer; another program can take the port first, so retry."""
        for _ in range(3):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                self._port = probe.getsockname()[1]
            with self._lock:  # close() may run while this starts: never leave a server behind it
                if self._closed:
                    raise RuntimeError("whisper-server closed")
                self._process = subprocess.Popen(
                    [*self._command, "--port", str(self._port)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                _die_with_this_process(self._process)
            deadline = time.monotonic() + self._ready_s
            while self._process.poll() is None and time.monotonic() < deadline:
                try:
                    self._post(np.zeros(1600, dtype=np.int16))
                    return
                except (urllib.error.URLError, ConnectionError):
                    time.sleep(0.2)
            self._kill()
        raise RuntimeError("whisper-server did not start")

    def _post(self, pcm16k: np.ndarray) -> dict:
        body, boundary = _multipart(_wav16k(pcm16k))
        request = urllib.request.Request(
            f"http://127.0.0.1:{self._port}/inference", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(request, timeout=30) as reply:
            return json.load(reply)

    def transcribe(self, pcm16k: np.ndarray) -> str:
        try:
            self._ready.result()
            result = self._post(pcm16k)
        except (OSError, urllib.error.URLError) as failed:  # FallbackStt takes over
            raise RuntimeError(f"whisper-server is not answering: {failed}") from None
        segments = result.get("segments") or [{"text": result.get("text", "")}]  # plain json has no segments
        return spoken_text(SimpleNamespace(text=s.get("text", ""), no_speech_prob=s.get("no_speech_prob", 0.0)) for s in segments)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._kill()

    def _kill(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.kill()
            self._process.wait()


_job: int | None = None


def _die_with_this_process(child: subprocess.Popen) -> None:
    """Windows: put the child in a job that closes, killing it, when this process ends however it ends.
    atexit never runs on a hard kill, and a leftover whisper-server holds the GPU memory."""
    global _job
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class _Limits(ctypes.Structure):  # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
            ("IoInfo", ctypes.c_uint64 * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    if _job is None:
        job = kernel32.CreateJobObjectW(None, None)
        limits = _Limits(LimitFlags=0x2000)  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not job or not kernel32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            log.warning("whisper-server may outlive telefony: no job object (%d)", ctypes.get_last_error())
            return
        _job = job  # held open for the life of this process; the OS closes it on exit
    if not kernel32.AssignProcessToJobObject(wintypes.HANDLE(_job), wintypes.HANDLE(int(child._handle))):  # type: ignore[attr-defined]
        log.warning("whisper-server may outlive telefony: not in the job (%d)", ctypes.get_last_error())


def _wav16k(pcm16k: np.ndarray) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm16k.astype(np.int16).tobytes())
    return out.getvalue()


def _multipart(wav: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    fields = [(b'name="response_format"', b"verbose_json"), (b'name="file"; filename="speech.wav"', wav)]
    body = b"".join(
        b"--" + boundary.encode() + b"\r\nContent-Disposition: form-data; " + header + b"\r\n\r\n" + value + b"\r\n"
        for header, value in fields
    )
    return body + b"--" + boundary.encode() + b"--\r\n", boundary


class FallbackStt:
    """The GPU engine, and the CPU one when it fails, so the far end's words still arrive."""

    def __init__(self, primary: Stt, backup: Stt) -> None:
        self.primary, self.backup = primary, backup

    def transcribe(self, pcm16k: np.ndarray) -> str:
        try:
            return self.primary.transcribe(pcm16k)
        except (OSError, RuntimeError) as failed:
            log.warning("%s failed (%s); using the CPU engine", type(self.primary).__name__, failed)
            return self.backup.transcribe(pcm16k)


# What Whisper says over silence or line noise: video outros from its training data, not the caller.
_HALLUCINATION = re.compile(r"(thanks?( you)? (so much )?for watching|please subscribe|subtitles by .*)[.!]*", re.IGNORECASE)
_NO_SPEECH = 0.6


def spoken_text(segments) -> str:
    """The segments' text, without those Whisper itself rates as no speech or that are known silence fillers."""
    kept = (s.text.strip() for s in segments if s.no_speech_prob < _NO_SPEECH)
    return " ".join(text for text in kept if text and not _HALLUCINATION.fullmatch(text)).strip()


def make_stt(model: str | None, whisper_cpp_server: str | None = None, whisper_cpp_model: str | None = None) -> Stt | None:
    """whisper.cpp when configured, backed by faster-whisper on the CPU when that is installed and configured."""
    cpu = None
    if model:
        try:
            cpu = WhisperStt(model)
        except ImportError:
            pass
    if whisper_cpp_server and whisper_cpp_model:
        gpu = WhisperCppStt(whisper_cpp_server, whisper_cpp_model)
        return FallbackStt(gpu, cpu) if cpu else gpu
    return cpu
