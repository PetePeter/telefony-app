"""The PhAP JSON Schema, vendored from github.com/PetePeter/phone-agent-protocol (schema/2026-10-27)."""

import json
from pathlib import Path

PROTOCOL_VERSION = "2026-10-27"
SCHEMA = json.loads((Path(__file__).parent / f"schema-{PROTOCOL_VERSION}.json").read_text(encoding="utf-8"))
