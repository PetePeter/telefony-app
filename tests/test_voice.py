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


async def test_resident_piper_loads_the_voice_once_and_speaks_at_the_link_rate():
    # Seen live: spawning piper per phrase reloaded a 63 MB voice each time, 13 s of silence before every line.
    from telefony_app.voice import ResidentPiperTts

    loads: list[str] = []

    class Chunk:
        audio_int16_bytes = np.ones(22050, dtype=np.int16).tobytes()  # one second at the voice's rate

    class Voice:
        class config:
            sample_rate = 22050

        def synthesize(self, text: str):
            return [Chunk()]

    def load(model: str) -> Voice:
        loads.append(model)
        return Voice()

    tts = ResidentPiperTts("voice.onnx", load)
    first, second = await tts.synthesize("Hello.", 8000), await tts.synthesize("Bye.", 8000)
    assert loads == ["voice.onnx"]
    assert abs(len(first) - 8000) < 400 and len(second) == len(first)


def test_whisper_silence_hallucinations_are_dropped_and_real_speech_kept():
    # Seen live: a quiet line came back as "Thanks for watching!" and reached the agent as the far end's words.
    from types import SimpleNamespace

    from telefony_app.voice import spoken_text

    def segment(text: str, no_speech_prob: float = 0.1) -> SimpleNamespace:
        return SimpleNamespace(text=text, no_speech_prob=no_speech_prob)

    assert spoken_text([segment(" Thanks for watching!")]) == ""
    assert spoken_text([segment(" Please subscribe.")]) == ""
    assert spoken_text([segment(" Who is this?"), segment(" Mm.", no_speech_prob=0.9)]) == "Who is this?"
    assert spoken_text([segment(" Thank you."), segment(" Bye.")]) == "Thank you. Bye."


FAKE_WHISPER_SERVER = r'''
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
log = args[args.index("--log") + 1]
with open(log + ".args", "w") as f:
    f.write(" ".join(args))


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        with open(log, "ab") as f:
            f.write(body)
        segments = [{"text": " Thanks for watching!", "no_speech_prob": 0.1},
                    {"text": " We open at nine.", "no_speech_prob": 0.1},
                    {"text": " Mm.", "no_speech_prob": 0.9}]
        reply = json.dumps({"text": "", "segments": segments}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *a):
        pass


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''


def fake_whisper_server(tmp_path) -> str:
    """A stand-in whisper-server executable: a .cmd wrapper around a tiny HTTP server answering /inference."""
    (tmp_path / "server.py").write_text(FAKE_WHISPER_SERVER, encoding="utf-8")
    exe = tmp_path / "whisper-server.cmd"
    exe.write_text(f'@"{sys.executable}" "{tmp_path / "server.py"}" %* --log "{tmp_path / "requests.bin"}"\n', encoding="utf-8")
    return str(exe)


@pytest.mark.skipif(sys.platform != "win32", reason="the fake server is a .cmd wrapper")
def test_whisper_cpp_sends_a_16k_wav_and_keeps_only_spoken_text(tmp_path):
    from telefony_app.voice import WhisperCppStt

    stt = WhisperCppStt(fake_whisper_server(tmp_path), "ggml-base.en.bin")
    try:
        text = stt.transcribe(np.zeros(16000, dtype=np.int16))
    finally:
        stt.close()

    assert text == "We open at nine."
    # Without a short audio context whisper encodes 30 s for every chunk: ~1.5 s each instead of ~0.5 s.
    assert "-ac 500" in (tmp_path / "requests.bin.args").read_text()
    sent = (tmp_path / "requests.bin").read_bytes()
    assert b"RIFF" in sent and b"WAVE" in sent
    wav = sent[sent.index(b"RIFF"):]
    assert int.from_bytes(wav[24:28], "little") == 16000  # sample rate in the WAV header


def test_whisper_cpp_that_cannot_start_raises_so_the_fallback_can_take_over(tmp_path):
    from telefony_app.voice import WhisperCppStt

    stt = WhisperCppStt(str(tmp_path / "missing-server.exe"), "ggml-base.en.bin", ready_s=1)
    with pytest.raises(RuntimeError):
        stt.transcribe(np.zeros(1600, dtype=np.int16))
    stt.close()


def test_failed_gpu_stt_falls_back_to_the_cpu_engine(caplog):
    from telefony_app.voice import FallbackStt

    class Broken:
        def transcribe(self, pcm16k):
            raise RuntimeError("whisper-server is not answering")

    class Cpu:
        def transcribe(self, pcm16k):
            return "hello"

    with caplog.at_level(logging.WARNING):
        assert FallbackStt(Broken(), Cpu()).transcribe(np.zeros(160, dtype=np.int16)) == "hello"
    assert "whisper-server is not answering" in caplog.text


@pytest.mark.skipif(sys.platform != "win32", reason="Job objects are Windows-only")
def test_whisper_server_dies_with_a_hard_killed_telefony(tmp_path):
    # Seen live: killing Timbre (and so telefony) by force left whisper-server running, holding GPU memory.
    import subprocess
    import time as clock

    exe = fake_whisper_server(tmp_path)
    host = tmp_path / "host.py"
    host.write_text(
        "import sys, time, numpy as np\n"
        "from telefony_app.voice import WhisperCppStt\n"
        f"stt = WhisperCppStt({exe!r}, 'ggml.bin')\n"
        "stt.transcribe(np.zeros(1600, dtype=np.int16))\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    telefony = subprocess.Popen([sys.executable, str(host)], stdout=subprocess.PIPE, text=True)
    try:
        assert telefony.stdout and telefony.stdout.readline().strip() == "ready"
        args = (tmp_path / "requests.bin.args").read_text()
        port = args.split("--port ")[1].split()[0]
        telefony.kill()  # no atexit, no close(): only the OS can clean up
        telefony.wait()
        import socket
        deadline = clock.monotonic() + 5
        while clock.monotonic() < deadline:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", int(port))):
                    return  # nothing listening: the server went with it
            clock.sleep(0.2)
        pytest.fail("whisper-server outlived telefony")
    finally:
        if telefony.poll() is None:
            telefony.kill()
