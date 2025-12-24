from .evaluation import evaluate
import importlib.util
import sys
from pathlib import Path
import multiprocessing as mp
import traceback
import json

TIMEOUT = 300

def load_all_student_functions(folder_path, func_name="solve"):
    folder = Path(folder_path)
    functions = {}

    for file in folder.glob("*.py"):
        module_name = f"student_{file.stem}_{abs(hash(file))}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            fn = getattr(module, func_name)
            functions[file.stem] = fn   # key = filename ohne .py

        except Exception as e:
            print(f"⚠️ Konnte {file.name} nicht laden: {e}")

    return functions

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
        module_name = f"student_{file.stem}_{abs(hash(file))}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            training_fn = getattr(module, "training_algorithm")
            agent_policy_fn = getattr(module, "agent_policy")
            result = run_student(training_fn, agent_policy_fn)
            with open(f"/out/{file.stem}.json", "w")as f:
                json.dump(result)

        except Exception as e:
            print(f"⚠️ Konnte {file.name} nicht laden: {e}")
