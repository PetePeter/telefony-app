import sys

import numpy as np

import pytest

from telefony_app.voice import FallbackTts, PiperTts, SapiTts, make_tts


@pytest.mark.skipif(sys.platform != "win32", reason="SAPI is Windows-only")
async def test_sapi_speaks_at_the_link_rate():
    pcm = await SapiTts().synthesize("hello", 8000)
    assert 0.2 < len(pcm) / 8000 < 3  # about half a second of speech at 8 kHz
    assert abs(pcm).max() > 1000


def test_auto_without_piper_model_falls_back_past_piper():
    tts = make_tts("auto", piper_model=None)
    assert tts is None or isinstance(tts, SapiTts)


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
