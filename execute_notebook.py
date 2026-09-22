import base64
import contextlib
import io
import os
import sys
import traceback
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import nbformat


root = Path(__file__).resolve().parent
os.chdir(root)
sys.path.insert(0, str(root))
path = Path(sys.argv[1]).resolve()
notebook = nbformat.read(path, as_version=4)
namespace = {"__name__": "__main__"}
figures = []


def capture_figures(*args, **kwargs):
    for number in plt.get_fignums():
        figure = plt.figure(number)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight", dpi=95)
        figures.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    plt.close("all")


plt.show = capture_figures
count = 0
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    count += 1
    figures.clear()
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(compile(cell.source, f"{path.name}:cell_{count}", "exec"), namespace)
    except Exception:
        cell.outputs = []
        if output.getvalue():
            cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=output.getvalue()))
        cell.outputs.append(nbformat.v4.new_output("error", ename="CellError", evalue=traceback.format_exc(), traceback=[traceback.format_exc()]))
        cell.execution_count = count
        nbformat.write(notebook, path)
        raise
    cell.execution_count = count
    cell.outputs = []
    if output.getvalue():
        cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=output.getvalue()))
    for png in figures:
        cell.outputs.append(nbformat.v4.new_output("display_data", data={"image/png": png}, metadata={}))
    print(f"  Cell {count} OK; outputs: {len(cell.outputs)}", flush=True)
nbformat.write(notebook, path)
