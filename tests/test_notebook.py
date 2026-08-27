import json
from pathlib import Path


def test_notebook_is_valid_json_and_code_compiles() -> None:
    path = Path("notebooks/CT_Restore_Colab_RunPod_End_to_End.ipynb")
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    assert any(cell["cell_type"] == "markdown" for cell in notebook["cells"])
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
