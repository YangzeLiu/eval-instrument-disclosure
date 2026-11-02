# regrade_v2 step 1: decrypt every HAL TAU-bench Airline run and dump, per record,
# the recorded trace, the embedded task, the agent's own LLM outputs and the user-simulator calls.
# Output: records.jsonl.gz (one JSON object per task record) in the working directory.
import sys, json, glob, os, gzip, re
from multiprocessing import Pool
WD = os.environ["HAL_PROBE_DIR"]  # same directory as in repro/README.md
sys.path.insert(0, WD + "/scripts")
from halio import load

CANON = [('claude-opus-4-1','claude-opus-4.1'),('claude-opus-4.1','claude-opus-4.1'),
 ('claude-opus-4','claude-opus-4'),('claude-3-7-sonnet','claude-3.7-sonnet'),
 ('claude-3.7-sonnet','claude-3.7-sonnet'),('claude-sonnet-4-5','claude-sonnet-4.5'),
 ('claude-sonnet-4','claude-sonnet-4'),('claude-haiku-4-5','claude-haiku-4.5'),
 ('deepseek-r1','deepseek-r1'),('deepseek-reasoner','deepseek-r1'),('r1','deepseek-r1'),
 ('deepseek-v3','deepseek-v3'),('deepseek-chat','deepseek-v3'),
 ('gpt-4.1','gpt-4.1'),('gpt-4o','gpt-4o'),('gpt-5','gpt-5'),
 ('gemini-2.0-flash','gemini-2.0-flash'),('gemini-2.5-pro','gemini-2.5-pro'),
 ('o4-mini','o4-mini'),('o3-mini','o3-mini'),('o3','o3')]
def canon(x):  # identical to fu11.py
    z = str(x).lower().split('/')[-1]; best = None
    for k, v in CANON:
        if z.startswith(k) and (best is None or len(k) > best[0]): best = (len(k), v)
    return best[1] if best else z

def first_content(c):
    msgs = (c.get("inputs") or {}).get("messages") or []
    if not msgs: return ""
    m0 = msgs[0]; s = m0.get("content") if isinstance(m0, dict) else str(m0)
    return s if isinstance(s, str) else json.dumps(s, default=str)

def is_user_sim(c):
    return "You are a user interacting with an agent" in first_content(c)

def last_input(c):
    msgs = (c.get("inputs") or {}).get("messages") or []
    if not msgs: return None
    m = msgs[-1]; s = m.get("content") if isinstance(m, dict) else m
    return s if isinstance(s, str) else json.dumps(s, default=str)

def one(z):
    try: d = load(z)
    except Exception as e: return [{"zipf": os.path.basename(z), "load_error": repr(e)}]
    cfg = d.get("config") or {}
    an = cfg.get("agent_name") or ""
    sc = "GEN" if "Generalist" in an else ("FEWSHOT" if "hot" in an.lower() else "TOOLCALL")
    mo = canon((cfg.get("agent_args") or {}).get("model_name") or an)
    bytask = {}
    n_none = {}
    seen = set()
    for c in d.get("raw_logging_results") or []:
        tid = str((c.get("attributes") or {}).get("weave_task_id"))
        o = c.get("output")
        if not isinstance(o, dict) or "choices" not in o:
            n_none[tid] = n_none.get(tid, 0) + 1; continue
        oid = o.get("id")
        if oid is not None:
            if oid in seen: continue          # same completion logged by both litellm and openai wrappers
            seen.add(oid)
        msg = (o["choices"][0] or {}).get("message") or {}
        ent = {"t": c.get("started_at"), "content": msg.get("content"),
               "tool_calls": [{"name": (tc.get("function") or {}).get("name"),
                               "arguments": (tc.get("function") or {}).get("arguments")}
                              for tc in (msg.get("tool_calls") or []) if isinstance(tc, dict)]}
        if is_user_sim(c):
            ent["kind"] = "user"; ent["last_in"] = last_input(c)
        else:
            ent["kind"] = "agent"; ent["n_in"] = len((c.get("inputs") or {}).get("messages") or [])
        bytask.setdefault(tid, []).append(ent)
    rows = []
    for tid, v in (d.get("raw_eval_results") or {}).items():
        if not isinstance(v, dict): continue
        t = v.get("task"); ta = v.get("taken_actions")
        if not isinstance(t, dict) or not isinstance(ta, list): continue
        calls = sorted(bytask.get(str(tid), []), key=lambda e: e["t"] or "")
        rows.append({"zipf": os.path.basename(z), "scaffold": sc, "model": mo, "agent_name": an,
                     "task": int(tid), "recorded": float(v.get("reward") or 0),
                     "taken_actions": [x for x in ta if isinstance(x, dict)],
                     "task_obj": t, "calls": calls, "n_failed_calls": n_none.get(str(tid), 0)})
    return rows

if __name__ == "__main__":
    zs = sorted(glob.glob(WD + "/raw/*taubench_airline*.zip"))
    with Pool(6) as p:
        res = p.map(one, zs)
    n = 0
    with gzip.open("records.jsonl.gz", "wt") as f:
        for rs in res:
            for r in rs:
                f.write(json.dumps(r, default=str) + "\n"); n += 1
    print("runs", len(zs), "records", n)
