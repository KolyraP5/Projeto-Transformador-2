"""Execute o notebook atual e preserve a execução, o código e o ambiente."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=("quick", "full"), required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("Use letras, números, hífen ou sublinhado em --run-name.")

    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks/atual/AZT1D_Setas_e_Tendencia.ipynb"
    project = root / "results/azt1d_seta"
    run = project / "runs" / args.run_name
    if (run / "execucao.json").exists():
        raise SystemExit("Execução já registrada. Preserve-a e use outro --run-name.")
    document = nbformat.read(notebook, as_version=4)
    # O notebook continua autossuficiente para upload no Colab. A cópia em src
    # permite revisão normal do algoritmo, sem alterar sua versão congelada.
    embedded = next(
        ast.literal_eval(node.value)
        for cell in document.cells if cell.cell_type == "code"
        for node in ast.parse(cell.source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "CORE_SOURCE" for t in node.targets)
    )
    core = root / "src/seta_tendencia/azt1d_core.py"
    if core.read_bytes() != embedded.encode("utf-8"):
        raise SystemExit("Core de src e notebook divergentes. Sincronize antes de executar.")

    env = os.environ.copy()
    env.update(AZT1D_RUN_NAME=args.run_name, AZT1D_QUICK=str(args.mode == "quick"),
               AZT1D_TESTE_JA_CONSULTADO="true", AZT1D_LOCAL_PROJECT=str(project),
               MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    env.setdefault("AZT1D_LOCAL_DATA", str(root / "AZT1D 2025/CGM Records"))
    # Usa exatamente o Python que chamou este script, mesmo com outro kernel padrão.
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Célula {cell_index}: {cell.source.splitlines()[0][:100]}", flush=True)

    client = NotebookClient(document, km=manager, timeout=None,
                            resources={"metadata": {"path": str(root)}},
                            on_cell_start=progress)
    started = datetime.now(timezone.utc).isoformat()
    print(f"Execução {args.run_name}; modo {args.mode}; Python {sys.executable}", flush=True)
    error = None
    try:
        client.execute(env=env)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if run.exists():
            nbformat.write(document, run / "notebook_executado.ipynb")
            source_dir = run / "codigo"
            source_dir.mkdir(exist_ok=True)
            (source_dir / "azt1d_core.py").write_bytes(embedded.encode("utf-8"))
            (source_dir / "executar_experimento.py").write_bytes(Path(__file__).read_bytes())
            (source_dir / "notebook_fonte.ipynb").write_bytes(notebook.read_bytes())
            frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
            (run / "requirements-freeze.txt").write_text(frozen, encoding="utf-8")
            metadata = dict(run_name=args.run_name, mode=args.mode, started_utc=started,
                            finished_utc=datetime.now(timezone.utc).isoformat(),
                            status="failed" if error else "completed", error=error,
                            python=sys.executable, prior_test_access=True,
                            prior_test_reason="quick_v1 já acessou parte dos períodos de teste",
                            notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(),
                            core_sha256=hashlib.sha256(core.read_bytes()).hexdigest())
            (run / "execucao.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    print(f"Concluído: {run}", flush=True)


if __name__ == "__main__":
    main()
