import subprocess
import sys
from pathlib import Path


root = Path(__file__).resolve().parent
start_at = int(sys.argv[1]) if len(sys.argv) > 1 else 1
for notebook in sorted((root / "notebooks").glob("[0-9][0-9]_*.ipynb")):
    if int(notebook.name[:2]) < start_at:
        continue
    print(f"Running {notebook.name}", flush=True)
    subprocess.run([sys.executable, str(root / "execute_notebook.py"), str(notebook)], cwd=root, check=True)
    print(f"Completed {notebook.name}", flush=True)
