import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / 'src'
VENV_PACKAGES = ROOT_DIR / '.venv' / 'lib' / 'python3.13' / 'site-packages'
CREWAI_DIR = ROOT_DIR / '.crewai'

CREWAI_DIR.mkdir(parents=True, exist_ok=True)
os.environ['CREWAI_STORAGE_DIR'] = str(CREWAI_DIR)
os.environ['CREWAI_CREDENTIALS_DIR'] = str(CREWAI_DIR)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if VENV_PACKAGES.exists() and str(VENV_PACKAGES) not in sys.path:
    sys.path.insert(0, str(VENV_PACKAGES))

try:
    import google.protobuf.runtime_version
    google.protobuf.runtime_version.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
except Exception:
    pass
