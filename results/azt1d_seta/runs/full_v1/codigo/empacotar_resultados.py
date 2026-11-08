"""Empacote uma rodada concluída, incluindo diagnósticos e fontes, sem caches."""
import argparse
import json
from pathlib import Path
import re
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("Identificador de execução inválido.")
    root = Path(__file__).resolve().parents[1]
    run = root / "results/azt1d_seta/runs" / args.run_name
    if not (run / "execucao.json").exists() or json.loads(
            (run / "execucao.json").read_text(encoding="utf-8"))["status"] != "completed":
        parser.error("A execução precisa estar concluída e registrada.")
    archive = run.parent / f"{args.run_name}_resultados.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(run.rglob("*")):
            relative = path.relative_to(run)
            if (path.is_file() and not {"prepared", "__pycache__"}.intersection(relative.parts)
                    and path.suffix != ".pyc"):
                z.write(path, (Path(run.name) / relative).as_posix())
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or z.testzip() is not None:
            raise RuntimeError("Falha na integridade do ZIP.")
    print(f"ZIP íntegro: {archive} ({len(names)} arquivos)")


if __name__ == "__main__":
    main()
