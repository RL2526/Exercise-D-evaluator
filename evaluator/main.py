import sys
sys.path.insert(0, "/agents") # before all other imports, do not change

from .evaluation import evaluate
import importlib.util
import sys
from pathlib import Path
import multiprocessing as mp
import traceback
import json
import traceback

TIMEOUT = 300


def _worker(training_fn, agent_policy_fn, q):
    try:
        q.put(("ok", evaluate(training_fn, agent_policy_fn)))
    except Exception:
        q.put(("err", traceback.format_exc()))

def run_student(training_fn, agent_policy_fn):
    q = mp.Queue()
    p = mp.Process(target=_worker, args=(training_fn, agent_policy_fn, q))
    p.start()
    p.join(TIMEOUT)

    if p.is_alive():
        p.kill()
        p.join()
        return {"ok": False, "timeout": True}

    if q.empty():
        return {"ok": False, "error": "no result"}

    status, payload = q.get()
    return {"ok": status == "ok", "result": payload if status == "ok" else None, "error": None if status == "ok" else payload}


if __name__ == "__main__":
    folder_path = "/agents"
    folder = Path(folder_path)


    for file in folder.glob("*.py"):
        # sicheren Modulnamen erzeugen (kein -, keine Sonderzeichen)
        safe_name = "agent_" + file.stem.replace("-", "_")

        try:
            spec = importlib.util.spec_from_file_location(safe_name, file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[safe_name] = module
            spec.loader.exec_module(module)
            training_fn = module.training_algorithm
            agent_policy_fn = module.agent_policy
            result = run_student(training_fn, agent_policy_fn)
            with open(f"/out/{file.stem}.json", "w")as f:
                json.dump(result)

        except Exception as e:
            print(f"⚠️ Konnte {file.name} nicht laden:")
            traceback.print_exc()
