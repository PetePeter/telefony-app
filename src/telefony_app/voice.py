"""Speech for calls. TTS backends run as separate programs (Piper is GPL-3.0, so it is never imported); STT is faster-whisper."""

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import wave
from pathlib import Path
from typing import Protocol

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


class SapiTts:
    """Windows' built-in voice via System.Speech, through PowerShell."""

    async def synthesize(self, text: str, rate: int) -> np.ndarray:
        fd, name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        path = Path(name)
        script = (
            "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SetOutputToWaveFile('{path}'); $s.Speak([Console]::In.ReadToEnd()); $s.Dispose()"
        )
        try:
            process = await asyncio.create_subprocess_exec(
                "powershell", "-NoProfile", "-Command", script,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await process.communicate(text.encode())
            with wave.open(str(path)) as wav:
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
                source_rate = wav.getframerate()
        finally:
            path.unlink(missing_ok=True)
        return shape_for_phone(resample(pcm, source_rate, rate), rate)


def make_tts(kind: str, piper_model: str | None) -> Tts | None:
    """auto: Piper if installed and a model is configured, else Windows SAPI, else no speech."""
    # A venv's Scripts folder is on PATH only when activated; pip install piper-tts puts piper there.
    # The Windows release ships it in a piper/ folder beside telefony-app.exe.
    here = Path(sys.executable).parent
    piper = shutil.which("piper") or shutil.which("piper", path=f"{here}{os.pathsep}{here / 'piper'}")
    if kind in ("auto", "piper") and piper and piper_model:
        return FallbackTts(PiperTts(piper, piper_model), SapiTts()) if sys.platform == "win32" else PiperTts(piper, piper_model)
    if kind == "piper":
        raise RuntimeError("voice.tts = piper needs the piper executable (on PATH or in this venv) and voice.piper_model")
    if kind in ("auto", "sapi") and sys.platform == "win32":
        return SapiTts()
    return None


class WhisperStt:
    def __init__(self, model: str) -> None:
        from faster_whisper import WhisperModel  # pyright: ignore[reportMissingImports]  optional dependency: pip install telefony-app[stt]

        self.model = WhisperModel(model, device="cpu", compute_type="int8")

    def transcribe(self, pcm16k: np.ndarray) -> str:
        segments, _ = self.model.transcribe(pcm16k.astype(np.float32) / 32768, language="en")
        return " ".join(s.text.strip() for s in segments).strip()


def make_stt(model: str | None) -> WhisperStt | None:
    if not model:
        return None
    try:
        return WhisperStt(model)
    except ImportError:
        return None
