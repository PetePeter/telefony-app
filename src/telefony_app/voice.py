"""Speech for calls. TTS backends run as separate programs (Piper is GPL-3.0, so it is never imported); STT is faster-whisper."""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import wave
from pathlib import Path
from typing import Protocol

import numpy as np

from telefony_app.bt.audio import resample, shape_for_phone

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
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        raw, _ = await process.communicate(text.encode())
        if process.returncode:
            raise RuntimeError(f"piper exited with {process.returncode}")
        return shape_for_phone(resample(np.frombuffer(raw, dtype=np.int16), self.rate, rate), rate)


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
        return PiperTts(piper, piper_model)
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
