# regrade_v2 step 2: decide, per record, which entries of taken_actions the agent issued and which the
# grader injected, from two independent sources:
#   P (parse)  : the agent's own LLM outputs. Tool-Calling / Few-Shot: tau-bench message_to_action on
#                each completion (exact). Generalist: Python AST of each executed code blob, tool call
#                sites with literal kwargs (non-literal -> wildcard), aligned in time windows anchored on
#                ask_user/respond turns.
#   S (sim)    : environment mechanics. calculate_reward() runs only inside a done step (a respond whose
#                user-simulator reply contains ###STOP###, or transfer_to_human_agents) and replays the
#                non-terminal gold through step(), so an injected block can only start right after such a
#                trigger. The simulator replies are in the logs.
# Usage: python provenance.py records.jsonl.gz out_prefix
import sys, json, gzip, ast, re, collections, functools

TERM = {"transfer_to_human_agents"}
NONGOLD = {"respond", "think"}            # never in airline gold (checked: gold names are tools only)
STOP = "###STOP###"
SIG = {'book_reservation': ['user_id','origin','destination','flight_type','cabin','flights','passengers',
        'payment_methods','total_baggages','nonfree_baggages','insurance'], 'calculate': ['expression'],
       'cancel_reservation': ['reservation_id'], 'get_reservation_details': ['reservation_id'],
       'get_user_details': ['user_id'], 'list_all_airports': [], 'search_direct_flight': ['origin','destination','date'],
       'search_onestop_flight': ['origin','destination','date'], 'send_certificate': ['user_id','amount'],
       'think': ['thought'], 'transfer_to_human_agents': ['summary'],
       'update_reservation_baggages': ['reservation_id','total_baggages','nonfree_baggages','payment_id'],
       'update_reservation_flights': ['reservation_id','cabin','flights','payment_id'],
       'update_reservation_passengers': ['reservation_id','passengers'], 'ask_user': ['question']}
WILD = object()

def nv(v):
    if isinstance(v, bool): return v
    if isinstance(v, float) and v.is_integer(): return int(v)
    if isinstance(v, dict): return {str(k): nv(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [nv(x) for x in v]
    return v
def nj(v): return json.dumps(nv(v), sort_keys=True, default=str)
def akey(a): return (a.get("name"), nj(a.get("kwargs") or {}))

def teq(x, y):
    """issued value x vs recorded value y. Lists may have grown: tau-bench stores the payment_methods list of
    book_reservation by reference as the reservation's payment_history, so later cancels/updates append to
    the very list that is serialised in taken_actions. Accept recorded lists that extend the issued one."""
    x, y = nv(x), nv(y)
    if isinstance(x, dict) and isinstance(y, dict):
        return set(x) == set(y) and all(teq(x[k], y[k]) for k in x)
    if isinstance(x, list) and isinstance(y, list):
        return len(x) <= len(y) and all(teq(a, b) for a, b in zip(x, y))
    return nj(x) == nj(y)

# ---------------- P for Tool-Calling / Few-Shot -------------------------------------------------
def tc_actions(calls):
    out, bad = [], 0
    for c in calls:
        if c["kind"] != "agent": continue
        tcs = c.get("tool_calls") or []
        if tcs and tcs[0].get("name") is not None:
            try: kw = json.loads(tcs[0].get("arguments") or "{}")
            except Exception: kw = None; bad += 1
            out.append({"name": tcs[0]["name"], "kwargs": kw, "t": c["t"]})
        else:
            out.append({"name": "respond", "kwargs": {"content": c.get("content")}, "t": c["t"]})
    return out, bad

def lcs_embed(A, T):
    """earliest maximum alignment of agent action list A into recorded list T; returns matched T idx set"""
    def eq(i, j):
        a, t = A[i], T[j]
        if a["name"] != t.get("name"): return False
        if a["name"] == "respond" or a["kwargs"] is None: return True   # respond is never gold; R1 text is post-processed
        return teq(a["kwargs"], t.get("kwargs") or {})
    n, m = len(A), len(T)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n-1, -1, -1):
        for j in range(m-1, -1, -1):
            dp[i][j] = max(dp[i+1][j], dp[i][j+1], (1+dp[i+1][j+1]) if eq(i, j) else 0)
    i = j = 0; matched = {}
    while i < n and j < m:
        if eq(i, j) and dp[i][j] == 1+dp[i+1][j+1]: matched[j] = i; i += 1; j += 1
        elif dp[i][j+1] == dp[i][j]: j += 1
        else: i += 1
    return matched, n - len(matched)

# ---------------- P for the Generalist (code parse) ---------------------------------------------
FENCE = re.compile(r"```(?:py|python)?\s*\n(.*?)\n```", re.S)
def code_of(text):
    if not isinstance(text, str) or not text.strip(): return None
    if re.search(r"facts survey", text, re.I) and "Code:" not in text: return None   # planning step
    m = FENCE.findall(text)
    if m: return "\n\n".join(m)
    try: ast.parse(text); return text            # smolagents falls back to the raw text if it parses
    except Exception: return None

def lit(node):
    try: return nv(ast.literal_eval(node))
    except Exception: return WILD

class Sites(ast.NodeVisitor):
    def __init__(self): self.sites = []; self.loop = []; self.indef = 0; self.lid = 0
    def _loop(self, node):
        self.lid += 1; self.loop.append(self.lid); self.generic_visit(node); self.loop.pop()
    visit_For = visit_While = visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = visit_AsyncFor = _loop
    def visit_FunctionDef(self, node):
        self.indef += 1; self._loop(node); self.indef -= 1
    visit_Lambda = visit_AsyncFunctionDef = visit_FunctionDef
    def visit_Call(self, node):
        # visit arguments first: inner calls execute before the outer one
        for a in node.args: self.visit(a)
        for k in node.keywords: self.visit(k.value)
        f = node.func
        nm = f.id if isinstance(f, ast.Name) else None
        if nm in SIG:
            kw = {}; allw = False
            for i, a in enumerate(node.args):
                if isinstance(a, ast.Starred): allw = True; continue
                if i < len(SIG[nm]): kw[SIG[nm][i]] = lit(a)
            for k in node.keywords:
                if k.arg is None: allw = True
                else: kw[k.arg] = lit(k.value)
            self.sites.append({"name": "respond" if nm == "ask_user" else nm,
                               "kw": {("content" if (nm == "ask_user" and k == "question") else k): v for k, v in kw.items()},
                               "allw": allw, "loop": self.loop[0] if self.loop else None, "indef": self.indef > 0})
        else:
            self.visit(f)

def parse_sites(code, bidx):
    try: tree = ast.parse(code)
    except Exception: return None
    v = Sites(); v.visit(tree)
    for s in v.sites:
        s["blk"] = bidx
        if s["loop"] is not None: s["loop"] = (bidx, s["loop"])
    return v.sites

def site_match(s, a):
    if s["name"] != a.get("name"): return False
    kw = nv(a.get("kwargs") or {})
    for k, v in s["kw"].items():
        if v is WILD: continue
        if k not in kw or not teq(v, kw[k]): return False
    if not s["allw"]:
        for k in kw:                    # every recorded kwarg must be accounted for (literal or wildcard)
            if k not in s["kw"]: return False
    return True

def feasible(W, sites, pool):
    """can the action list W be produced, in order, by the concatenated call sites (loops repeatable,
    functions defined earlier callable any time)?"""
    W = tuple(W)
    lstart = {}
    for i, s in enumerate(sites):
        if s["loop"] is not None and s["loop"] not in lstart: lstart[s["loop"]] = i
    @functools.lru_cache(maxsize=None)
    def go(i, j):
        if i == len(W): return True
        a = W[i]
        if any(site_match(s, a) for s in pool) and go(i+1, j): return True
        for k in range(j, len(sites)):
            s = sites[k]
            if site_match(s, a):
                nj_ = lstart[s["loop"]] if s["loop"] is not None else k+1
                if go(i+1, nj_): return True
        return False
    return go(0, 0)

# ---------------- per record --------------------------------------------------------------------
def analyse(r):
    T = r["taken_actions"]; task = r["task_obj"]
    gold = [(a["name"], nj(a.get("kwargs") or {})) for a in task.get("actions") or [] if a["name"] not in TERM]
    n = len(gold); tk = [akey(a) for a in T]
    occ = [p for p in range(0, len(T)-n+1) if n >= 1 and tk[p:p+n] == gold]
    # --- S: map every respond to its user-simulator call
    users = [c for c in r["calls"] if c["kind"] == "user"]
    up = 0; resp = {}            # T index -> (time, stop) ; None if unmatched
    for i, a in enumerate(T):
        if a.get("name") != "respond": continue
        content = (a.get("kwargs") or {}).get("content")
        hit = None
        for k in range(up, len(users)):
            li = users[k].get("last_in")
            if li == content or (isinstance(li, str) and isinstance(content, str) and li.strip() == content.strip()):
                hit = k; break
        if hit is None: resp[i] = None
        else:
            resp[i] = (users[hit]["t"], STOP in (users[hit].get("content") or "")); up = hit+1
    triggers = [i for i, a in enumerate(T) if (a.get("name") == "respond" and resp.get(i) and resp[i][1]) or a.get("name") in TERM]
    unmatched_resp = sum(1 for v in resp.values() if v is None)
    s_inj = set(); s_bad = 0
    for i in triggers:
        if n == 0: continue
        if tk[i+1:i+1+n] == gold: s_inj.update(range(i+1, i+1+n))
        else: s_bad += 1             # trigger not followed by gold: should not happen under pinned semantics
    # --- P
    p_inj = set(); occ_lab = {}; p_note = ""
    agent_tool_calls = None
    if r["scaffold"] in ("TOOLCALL", "FEWSHOT"):
        A, bad = tc_actions(r["calls"])
        matched, a_unmatched = lcs_embed(A, T)
        p_inj = {j for j in range(len(T)) if j not in matched}
        agent_tool_calls = sum(1 for a in A if a["name"] not in ("respond",))
        p_note = "A=%d unmatchedA=%d parsefail=%d" % (len(A), a_unmatched, bad)
        for p in occ:
            blk = set(range(p, p+n))
            occ_lab[p] = "injected" if blk <= p_inj else ("agent" if not (blk & p_inj) else "mixed")
        p_extra = sorted(p_inj - set().union(*[set(range(p, p+n)) for p in occ if occ_lab[p] == "injected"])) if occ else sorted(p_inj)
    else:
        blocks = []
        for c in r["calls"]:
            if c["kind"] != "agent": continue
            code = code_of(c.get("content"))
            if code is None: continue
            s = parse_sites(code, len(blocks))
            if s is None: continue
            blocks.append((c["t"], s))
        bt = [b[0] for b in blocks]
        def blk_at(t):
            k = -1
            for i, x in enumerate(bt):
                if x < t: k = i
            return k
        agent_tool_calls = None
        ridx = [i for i, a in enumerate(T) if a.get("name") == "respond"]
        def window(p, q):          # T positions [p,q) -> (prev respond, next respond)
            prev = max([i for i in ridx if i < p], default=None)
            nxt = min([i for i in ridx if i >= q], default=None)
            return prev, nxt
        def blkrange(prev, nxt):
            lo = 0 if prev is None or resp.get(prev) is None else max(blk_at(resp[prev][0]), 0)
            hi = len(blocks)-1 if nxt is None or resp.get(nxt) is None else blk_at(resp[nxt][0])
            if prev is not None and resp.get(prev) is None: lo = 0
            return lo, max(hi, lo)
        def test(prev, nxt, drop):
            lo, hi = blkrange(prev, nxt)
            sites = [s for b in blocks[lo:hi+1] for s in b[1] if s["name"] != "respond"]
            pool = [s for b in blocks[:lo] for s in b[1] if s["indef"] and s["name"] != "respond"]
            a0 = (prev+1) if prev is not None else 0
            a1 = nxt if nxt is not None else len(T)
            W = [T[i] for i in range(a0, a1) if i not in drop and T[i].get("name") != "respond"]
            return feasible(W, sites, pool)
        for p in occ:
            prev, nxt = window(p, p+n)
            blk = set(range(p, p+n))
            # most permissive setting for explaining p as agent-issued: every other occurrence removed.
            # "injected" therefore means: p cannot be produced by the agent's code even then, while the
            # rest of the window can once p is removed.
            others = set(o for q in occ if q != p for o in range(q, q+n))
            f_all = test(prev, nxt, others)
            f_drop = test(prev, nxt, others | blk)
            occ_lab[p] = ("agent_or_ambiguous" if f_all else ("injected" if f_drop else "unparsed"))
            if occ_lab[p] == "injected": p_inj |= blk
        # tool calls issued by the agent (any non-respond action not inside an injected block)
        p_extra = []
        if not blocks: p_note = "no_code_blocks"
    inj_final = set(p_inj) if r["scaffold"] != "GEN" else set()
    lab_final = {}
    for p in occ:
        blk = set(range(p, p+n))
        lp = occ_lab[p]; ls = "injected" if blk <= s_inj else "agent"
        if r["scaffold"] != "GEN":
            lab_final[p] = "injected" if lp == "injected" else "agent"
        else:
            if lp == "injected": lab_final[p] = "injected"
            elif lp == "agent_or_ambiguous": lab_final[p] = ls      # P cannot exclude the agent; S decides
            else: lab_final[p] = ls if ls == "injected" else "unparsed"
        if lab_final[p] == "injected": inj_final |= blk
    return dict(n=n, occ=occ, occ_P={str(k): v for k, v in occ_lab.items()},
                occ_S={str(p): ("injected" if set(range(p, p+n)) <= s_inj else "agent") for p in occ},
                occ_final={str(k): v for k, v in lab_final.items()},
                inj=sorted(inj_final), p_inj=sorted(p_inj), s_inj=sorted(s_inj),
                triggers=triggers, n_resp=len(resp), unmatched_resp=unmatched_resp, s_bad=s_bad,
                p_note=p_note, p_extra=len(p_extra) if r["scaffold"] != "GEN" else None)

if __name__ == "__main__":
    src, outp = sys.argv[1], sys.argv[2]
    with gzip.open(src, "rt") as f, open(outp + "_prov.jsonl", "w") as g:
        for line in f:
            r = json.loads(line)
            res = analyse(r)
            res.update(zipf=r["zipf"], task=r["task"], scaffold=r["scaffold"], model=r["model"], recorded=r["recorded"],
                       n_actions=len(r["taken_actions"]))
            g.write(json.dumps(res) + "\n")
