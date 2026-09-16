import os
from pathlib import Path
import sys
root = Path(os.environ["VITALY_BOT_TOF_ROOT"]).expanduser().resolve()
os.chdir(root)
os.execv(str(root / ".venv/bin/python"), [str(root / ".venv/bin/python"), str(root / "scripts/tof_remote_workspace.py"), *sys.argv[1:]])
