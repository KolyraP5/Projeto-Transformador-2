"""Encerre somente os processos da bateria de pesquisa deste repositório."""
from pathlib import Path
from datetime import datetime, timezone
import json
import psutil

root = Path(__file__).resolve().parents[1]
for p in psutil.process_iter(["pid", "name", "cmdline"]):
    try:
        args = p.info["cmdline"] or []
        if (p.info["name"] and p.info["name"].lower() == "python.exe"
                and any(a.replace("\\", "/").endswith("scripts/executar_aprimoramento.py") for a in args)
                and Path(p.cwd()).resolve() == root):
            for child in p.children(recursive=True):
                child.terminate()
            p.terminate()
            print("Stopped research process", p.pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
exp = root / "results/azt1d_seta/experiments/aprimoramento_v2"
p = exp / "stages/confirmation.json"
d = json.loads(p.read_text(encoding="utf-8"))
d.update(status="interrupted_by_user", finished_utc=datetime.now(timezone.utc).isoformat(),
         reason="Prazo de entrega; nao executar mais blocos")
d["completed_model_artifacts"] = sorted(f.parent.name for f in (exp / "models").glob("*/completed.json"))
p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
print("Preserved", len(d["completed_model_artifacts"]), "complete model artifacts")
