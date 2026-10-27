"""Build the Windows release: dist/telefony-app-win64.zip (+ .sha256).

telefony-app/telefony-app.exe   telefony-app (MIT), PyInstaller onedir
telefony-app/piper/piper.exe    Piper (GPL-3.0), a separate program telefony-app runs as a subprocess

Run from the repo root with the dev venv: .venv/Scripts/python packaging/build_windows.py
"""

import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD, DIST = ROOT / "build", ROOT / "dist"
APP = DIST / "telefony-app"
ZIP = DIST / "telefony-app-win64.zip"


def pyinstaller(entry: str, name: str, *collect: str) -> Path:
    args = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir", "--console", "--name", name,
            "--distpath", str(BUILD / "dist"), "--workpath", str(BUILD / "work"), "--specpath", str(BUILD)]
    for package in collect:
        args += ["--collect-all", package]
    subprocess.run([*args, str(ROOT / "packaging" / entry)], check=True)
    return BUILD / "dist" / name


def main() -> None:
    shutil.rmtree(BUILD, ignore_errors=True)
    shutil.rmtree(DIST, ignore_errors=True)
    app = pyinstaller("telefony_app_entry.py", "telefony-app",
                      "telefony_app", "bumble", "usb1", "libusb_package", "faster_whisper", "ctranslate2")
    piper = pyinstaller("piper_entry.py", "piper", "piper", "onnxruntime")
    shutil.copytree(app, APP)
    shutil.copytree(piper, APP / "piper")
    for doc in ("LICENSE", "README.md", "THIRD_PARTY_NOTICES.md", "telefony.example.toml"):
        shutil.copy(ROOT / doc, APP / doc)
    (APP / "piper" / "SOURCE.txt").write_text(
        "Piper is GPL-3.0-or-later and runs here as a separate program.\n"
        "Source: https://github.com/OHF-Voice/piper1-gpl\n", encoding="utf-8")
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(APP.rglob("*")):
            archive.write(path, path.relative_to(DIST))
    digest = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    (DIST / f"{ZIP.name}.sha256").write_text(f"{digest}  {ZIP.name}\n", encoding="utf-8")
    print(ZIP, f"{ZIP.stat().st_size / 1e6:.0f} MB", digest)


if __name__ == "__main__":
    main()
