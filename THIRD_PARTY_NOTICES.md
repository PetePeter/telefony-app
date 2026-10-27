# Third-party notices

telefony-app is MIT-licensed. It uses, but does not include, the following:

| Component | License | How it is used |
|---|---|---|
| [Bumble](https://github.com/google/bumble) | Apache-2.0 | Python dependency: the user-space Bluetooth stack |
| [websockets](https://github.com/python-websockets/websockets) | BSD-3-Clause | Python dependency: PhAP transport |
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | MIT | Python dependency: MCP projection |
| [NumPy](https://numpy.org) / [SciPy](https://scipy.org) | BSD-3-Clause | Python dependency: audio resampling and filtering |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT | Optional dependency (`[stt]`): speech-to-text |
| [Piper](https://github.com/OHF-Voice/piper1-gpl) | GPL-3.0 | **Not a dependency.** Optionally run as a separate program found on `PATH`; never imported or distributed |
| Piper voice models | per model (see each model's card) | Not distributed; the user downloads them |
| Windows SAPI (System.Speech) | part of Windows | Optionally invoked through PowerShell |

The Bluetooth modules under `src/telefony_app/bt/` started as the author's own telefony-cube proof of concept.
