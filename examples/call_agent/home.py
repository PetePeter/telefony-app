"""The demo agent's home folder in the user profile: its config and the phonebook of people it may reach."""

import os
import sys
from pathlib import Path

PHONEBOOK_TEMPLATE = "name,number\n"


def default_home() -> Path:
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "telefony-call-agent"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "telefony-call-agent"


def init_home(home: Path) -> None:
    """Create the folder with template files; files that already exist are left alone."""
    home.mkdir(parents=True, exist_ok=True)
    templates = {
        "call_agent.toml": (Path(__file__).parent / "call_agent.example.toml").read_text(encoding="utf-8"),
        "phonebook.csv": PHONEBOOK_TEMPLATE,
    }
    for name, text in templates.items():
        if not (home / name).exists():
            (home / name).write_text(text, encoding="utf-8")
