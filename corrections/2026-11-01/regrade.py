# regrade_v2 step 3 (needs the pinned tau-bench): offline re-grading and per-record CSV.
# Usage: python regrade.py records.jsonl.gz all_prov.jsonl ../fu11_records.csv records_v2.csv
import sys, types, json, gzip, csv, copy, os
import multiprocessing as mp

class _F:
    def __getattr__(self, k): return _F()
    def __getitem__(self, k): return _F()
    def __iter__(self): return iter([_F()])
    def __call__(self, *a, **k): return {"role": "user", "content": ""}
    def __str__(self): return ""
    def __float__(self): return 0.0
m = types.ModuleType("litellm"); m.completion = lambda *a, **k: _F(); sys.modules["litellm"] = m
WD = os.environ["HAL_PROBE_DIR"]  # same directory as in repro/README.md
TB = WD + "/taubench/tau-bench-14bf0ef52e595922d597a38f32d3e8c0dce3a8f8"
sys.path.insert(0, TB)
from tau_bench.envs.airline import MockAirlineDomainEnv
from tau_bench.envs import user as _u
_u.LLMUserSimulationEnv.generate_next_message = lambda self, msgs: ""
_u.LLMUserSimulationEnv.get_total_cost = lambda self: 0.0
_u.LLMUserSimulationEnv.step = lambda self, c: ""
from tau_bench.types import Action, Task
from tau_bench.envs.airline.tasks_test import TASKS as PINNED

TERM = {"transfer_to_human_agents"}
WRITE = {"book_reservation", "cancel_reservation", "send_certificate", "update_reservation_baggages",
         "update_reservation_flights", "update_reservation_passengers"}

def nj(v): return json.dumps(v, sort_keys=True, default=str)

def mkenv(i, task_obj=None):
    e = MockAirlineDomainEnv(user_strategy="llm", user_model="gpt-4o", user_provider="openai",
                             task_split="test", task_index=i)
    e.reset(task_index=i)
    e.task = Task(**copy.deepcopy(task_obj)) if task_obj is not None else copy.deepcopy(e.task)
    return e

FIXES = [0]
def do(e, a, fix):
    nm = a.get("name"); kw = copy.deepcopy(a.get("kwargs") or {})
    if nm == "respond" or nm in TERM:
        e.actions.append(Action(name=nm, kwargs=kw)); return
    if fix and nm == "book_reservation" and isinstance(kw.get("payment_methods"), list):
        pm = kw["payment_methods"]
        for L in range(len(pm), 0, -1):
            k2 = dict(copy.deepcopy(kw), payment_methods=copy.deepcopy(pm[:L]))
            try: obs = e.tools_map[nm].invoke(data=e.data, **k2)
            except Exception as ex: obs = "Error: %s" % ex
            if not str(obs).startswith("Error: payment amount does not add up"):
                if L < len(pm): FIXES[0] += 1
                e.actions.append(Action(name=nm, kwargs=k2)); return
        e.actions.append(Action(name=nm, kwargs=kw)); return
    try: e.step(Action(name=nm, kwargs=kw))
    except Exception: pass

def score(i, actions, task_obj=None, fix=False):
    """FU11 procedure: replay the given actions on a fresh pinned env, then calculate_reward()."""
    e = mkenv(i, task_obj)
    for a in actions: do(e, a, fix)
    return e.calculate_reward().reward

def faithful(i, T, inj, triggers, task_obj=None, fix=False):
    """re-enact the recorded episode: agent actions are stepped, and at every done-trigger the grader is
    run exactly as tau-bench does (which resets the DB to the gold state, as in the original run);
    the injected positions are skipped because calculate_reward() re-creates them. Returns the reward
    of the last grading, i.e. what HAL should have recorded."""
    e = mkenv(i, task_obj); last = None; trig = set(triggers)
    for k, a in enumerate(T):
        if k in inj: continue
        do(e, a, fix)
        if k in trig: last = e.calculate_reward().reward
    return last

def work(args):
    r, p, f = args
    i = r["task"]; T = r["taken_actions"]; emb = r["task_obj"]
    inj = set(p["inj"]); own = [a for k, a in enumerate(T) if k not in inj]
    n = p["n"]; occ = p["occ"]; L = len(T)
    suffix_inj = any(q + n == L for q in occ if str(q) in p["occ_final"] and p["occ_final"][str(q)] == "injected")
    interior_inj = any(q + n < L for q in occ if p["occ_final"].get(str(q)) == "injected")
    graded = (len(inj) > 0) or (r["recorded"] > 0) or (n == 0 and len(p["triggers"]) > 0)
    basis = ("injected_block" if inj else ("reward_only" if r["recorded"] > 0 else
             ("trigger_no_gold" if (n == 0 and p["triggers"]) else "none")))
    pg = [a.name for a in PINNED[i].actions]; eg = [x["name"] for x in emb.get("actions") or []]
    emb_eq = nj([x.model_dump() for x in PINNED[i].actions]) == nj([{"name": x["name"], "kwargs": x.get("kwargs") or {}} for x in emb.get("actions") or []])
    names_own = [a.get("name") for a in own]
    out = dict(zipf=r["zipf"], scaffold=r["scaffold"], model=r["model"], task=i, n_gold_nt=n,
               gold_write=int(any(a.name in WRITE for a in PINNED[i].actions)),
               recorded=r["recorded"], n_actions=L,
               occ=";".join(map(str, occ)),
               occ_P=";".join("%s:%s" % (k, v) for k, v in p["occ_P"].items()),
               occ_S=";".join("%s:%s" % (k, v) for k, v in p["occ_S"].items()),
               occ_final=";".join("%s:%s" % (k, v) for k, v in p["occ_final"].items()),
               n_injected_actions=len(inj), injected_suffix=int(suffix_inj), injected_interior=int(interior_inj),
               triggers=";".join(map(str, p["triggers"])), n_gradings=len(p["triggers"]),
               graded_new=int(graded), graded_basis=basis,
               fu11_graded=f["leak"] if f else "", fu11_offline=f["offline"] if f else "",
               agent_tool_calls=sum(1 for x in names_own if x not in ("respond",)),
               agent_env_calls=sum(1 for x in names_own if x not in ("respond", "think")),
               agent_writes=sum(1 for x in names_own if x in WRITE),
               emb_gold_eq_pinned=int(emb_eq))
    try:
        out["off_new"] = score(i, own)
        out["off_new_emb"] = score(i, own, emb)
        FIXES[0] = 0
        out["off_new_fix"] = score(i, own, None, True)
        out["alias_fixes"] = FIXES[0]
        out["off_verbatim"] = score(i, T)
        out["faithful"] = faithful(i, T, inj, p["triggers"]) if p["triggers"] else ""
        out["faithful_fix"] = faithful(i, T, inj, p["triggers"], None, True) if p["triggers"] else ""
        out["faithful_emb"] = faithful(i, T, inj, p["triggers"], emb) if p["triggers"] else ""
        out["replay_error"] = ""
    except Exception as ex:
        out["replay_error"] = repr(ex)[:200]
    return out

if __name__ == "__main__":
    src, prov, fu11, dst = sys.argv[1:5]
    P = {(p["zipf"], p["task"]): p for p in map(json.loads, open(prov))}
    F = {(f["zipf"], int(f["task"])): f for f in csv.DictReader(open(fu11))}
    jobs = []
    with gzip.open(src, "rt") as fh:
        for line in fh:
            r = json.loads(line); r.pop("calls", None)
            jobs.append((r, P[(r["zipf"], r["task"])], F.get((r["zipf"], r["task"]))))
    with mp.Pool(8) as pool:
        rows = pool.map(work, jobs, chunksize=20)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("rows", len(rows), "replay errors", sum(1 for x in rows if x["replay_error"]),
          "missing fu11", sum(1 for j in jobs if j[2] is None))
