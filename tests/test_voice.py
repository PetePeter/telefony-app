import logging
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

import pytest

from telefony_app.voice import FallbackTts, PiperTts, PowerShellSapiSynthesizer, SapiTts, make_tts


@pytest.mark.skipif(sys.platform != "win32", reason="SAPI is Windows-only")
async def test_sapi_speaks_at_the_link_rate():
    pcm = await SapiTts().synthesize("hello", 8000)
    assert 0.2 < len(pcm) / 8000 < 3  # about half a second of speech at 8 kHz
    assert abs(pcm).max() > 1000


def test_auto_without_piper_model_falls_back_past_piper():
    tts = make_tts("auto", piper_model=None)
    assert tts is None or isinstance(tts, SapiTts)


@pytest.mark.parametrize("kind", ["sapi", "auto"])
def test_selected_sapi_backend_receives_configured_voice(monkeypatch, kind):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr("telefony_app.voice.shutil.which", lambda *args, **kwargs: None)

    tts = make_tts(kind, piper_model=None, sapi_voice="Microsoft Zira Desktop")

    assert isinstance(tts, SapiTts)
    assert tts.voice_name == "Microsoft Zira Desktop"


@pytest.mark.parametrize("kind", ["auto", "piper"])
def test_piper_backup_sapi_receives_configured_voice(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr("telefony_app.voice.shutil.which", lambda *args, **kwargs: "piper.exe")

    tts = make_tts(kind, piper_model=str(tmp_path / "voice.onnx"), sapi_voice="Microsoft Zira Desktop")

    assert isinstance(tts, FallbackTts)
    assert isinstance(tts.backup, SapiTts)
    assert tts.backup.voice_name == "Microsoft Zira Desktop"


def test_none_means_no_speech():
    assert make_tts("none", "model.onnx") is None


def fake_piper(tmp_path, script: str) -> str:
    """A stand-in piper executable: a .cmd wrapper around a Python script."""
    (tmp_path / "piper.py").write_text(script, encoding="utf-8")
    exe = tmp_path / "piper.cmd"
    exe.write_text(f'@"{sys.executable}" "{tmp_path / "piper.py"}" %*\n', encoding="utf-8")
    return str(exe)


BROKEN = "import sys\nsys.stdin.read()\nsys.stderr.write('espeak: unknown phoneme')\nsys.exit(1)\n"
ECHO_TEXT = (
    "import sys\ntext = sys.stdin.buffer.read().decode('utf-8')\nopen(sys.argv[2] + '.seen', 'w', encoding='utf-8').write(text)\n"
    "sys.stdout.buffer.write(bytes(4410))\n"
)


@pytest.mark.skipif(sys.platform != "win32", reason=".cmd wrapper")
async def test_piper_failure_reports_its_stderr(tmp_path):
    tts = PiperTts(fake_piper(tmp_path, BROKEN), str(tmp_path / "voice.onnx"))
    with pytest.raises(RuntimeError, match="unknown phoneme"):
        await tts.synthesize("hello", 8000)


@pytest.mark.skipif(sys.platform != "win32", reason=".cmd wrapper")
async def test_piper_gets_one_line_of_plain_utf8(tmp_path):
    tts = PiperTts(fake_piper(tmp_path, ECHO_TEXT), str(tmp_path / "voice.onnx"))
    await tts.synthesize("It’s fine —\r\nreally.\x00\x07  Café", 8000)
    assert (tmp_path / "voice.onnx.seen").read_text(encoding="utf-8") == "It’s fine — really. Café\n"


class Recorder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def synthesize(self, text: str, rate: int):
        self.texts.append(text)
        return np.ones(rate // 10, dtype=np.int16)


@pytest.mark.skipif(sys.platform != "win32", reason=".cmd wrapper")
async def test_failed_piper_falls_back_to_the_backup_voice(tmp_path):
    backup = Recorder()
    tts = FallbackTts(PiperTts(fake_piper(tmp_path, BROKEN), str(tmp_path / "voice.onnx")), backup)
    pcm = await tts.synthesize("are you there", 8000)
    assert len(pcm) == 800 and backup.texts == ["are you there"]


class FakeSapiSynthesizer:
    def __init__(self, voice_fallback: bool = False) -> None:
        self.voice_fallback = voice_fallback
        self.calls: list[tuple[str, str | None]] = []
        self.selected_voice: str | None = None

    async def synthesize(self, text: str, path: Path, voice_name: str | None) -> bool:
        self.calls.append((text, voice_name))
        self.selected_voice = None if self.voice_fallback else voice_name
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(np.ones(1600, dtype=np.int16).tobytes())
        return self.voice_fallback


async def test_sapi_passes_configured_voice_to_injected_synthesizer():
    synthesizer = FakeSapiSynthesizer()
    tts = SapiTts("Microsoft Zira Desktop", synthesizer=synthesizer)

    pcm = await tts.synthesize("hello", 8000)

    assert synthesizer.calls == [("hello", "Microsoft Zira Desktop")]
    assert len(pcm) == 800


async def test_unknown_sapi_voice_uses_default_and_logs_warning(caplog):
    synthesizer = FakeSapiSynthesizer(voice_fallback=True)
    tts = SapiTts("Missing voice", synthesizer=synthesizer)

    with caplog.at_level(logging.WARNING, logger="telefony_app.voice"):
        pcm = await tts.synthesize("hello", 8000)
        await tts.synthesize("again", 8000)

    assert synthesizer.calls == [("hello", "Missing voice"), ("again", "Missing voice")]
    assert synthesizer.selected_voice is None
    assert caplog.text.lower().count("default windows voice") == 1
    assert len(pcm) == 800


async def test_sapi_voice_is_literal_environment_data(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    class Process:
        async def communicate(self, text):
            captured["input"] = text
            return b"", b"__TELEFONY_SAPI_VOICE_FALLBACK__"

    async def create_subprocess_exec(*args: Any, **kwargs: Any) -> Process:
        captured["args"] = args
        captured["env"] = kwargs["env"]
        return Process()

    monkeypatch.setattr("telefony_app.voice.asyncio.create_subprocess_exec", create_subprocess_exec)
    voice_name = "Zira's voice; Write-Output injected"
    synthesizer = PowerShellSapiSynthesizer()

    used_default = await synthesizer.synthesize("hello", tmp_path / "speech.wav", voice_name)

    script = captured["args"][3]
    assert captured["env"]["TELEFONY_APP_SAPI_VOICE"] == voice_name
    assert captured["env"]["TELEFONY_APP_SAPI_VOICE_CONFIGURED"] == "1"
    assert captured["input"] == b"hello"
    assert voice_name not in script
    assert "$env:TELEFONY_APP_SAPI_VOICE" in script
    assert used_default is True


async def test_unconfigured_sapi_ignores_inherited_voice_environment(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    class Process:
        async def communicate(self, text):
            return b"", b""

    async def create_subprocess_exec(*args: Any, **kwargs: Any) -> Process:
        captured["env"] = kwargs["env"]
        return Process()

    monkeypatch.setenv("TELEFONY_APP_SAPI_VOICE", "stale voice")
    monkeypatch.setenv("TELEFONY_APP_SAPI_VOICE_CONFIGURED", "1")
    monkeypatch.setattr("telefony_app.voice.asyncio.create_subprocess_exec", create_subprocess_exec)

    await PowerShellSapiSynthesizer().synthesize("hello", tmp_path / "speech.wav", None)

    assert "TELEFONY_APP_SAPI_VOICE" not in captured["env"]
    assert "TELEFONY_APP_SAPI_VOICE_CONFIGURED" not in captured["env"]


@pytest.mark.skipif(sys.platform != "win32", reason="SAPI is Windows-only")
async def test_unknown_sapi_voice_falls_back_through_powershell_and_warns(caplog):
    tts = SapiTts("__telefony_test_missing_voice_9f2d__", synthesizer=PowerShellSapiSynthesizer())

    with caplog.at_level(logging.WARNING, logger="telefony_app.voice"):
        pcm = await tts.synthesize("hello", 8000)

    assert len(pcm) > 0
    assert "default Windows voice" in caplog.text
