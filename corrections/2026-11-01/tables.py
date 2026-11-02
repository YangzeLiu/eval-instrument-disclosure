# regrade_v2 step 4: all tables (old = FU11 suffix rule, new = provenance rule). Usage: python tables.py records_v2.csv
import csv, sys, collections
R = list(csv.DictReader(open(sys.argv[1])))
f = lambda x: float(x) if x not in ("", None) else None
for r in R:
    for k in ("n_gold_nt", "gold_write", "graded_new", "n_injected_actions", "injected_suffix", "injected_interior",
              "agent_tool_calls", "agent_env_calls", "agent_writes", "n_gradings", "n_actions", "task", "emb_gold_eq_pinned"):
        r[k] = int(r[k])
    r["recorded"] = float(r["recorded"]); r["pub"] = r["recorded"] > 0
    r["old_g"] = r["fu11_graded"] == "1"
    r["old_c"] = r["pub"] if r["old_g"] else (f(r["fu11_offline"]) or 0) > 0
    r["new_c"] = r["pub"] if r["graded_new"] else (f(r["off_new"]) or 0) > 0
    r["new_c_emb"] = r["pub"] if r["graded_new"] else (f(r["off_new_emb"]) or 0) > 0
    r["new_c_fix"] = r["pub"] if r["graded_new"] else (f(r["off_new_fix"]) or 0) > 0
SC = ("GEN", "TOOLCALL", "FEWSHOT")
A = [r for r in R if r["n_gold_nt"] >= 1]; B = [r for r in R if r["n_gold_nt"] == 0]
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
P = lambda xs: "%.3f" % mean(xs)
def H(t): print("\n## " + t)

H("Record categories (all %d records)" % len(R))
print("| scaffold | records | GroupA | injected block (any) | suffix only | interior only | both | graded by reward only (no block) | GroupB graded by trigger | ungraded | agent-issued gold-equal blocks (records) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for s in SC:
    x = [r for r in R if r["scaffold"] == s]; a = [r for r in x if r["n_gold_nt"] >= 1]
    inj = [r for r in a if r["n_injected_actions"] > 0]
    agentocc = [r for r in a if "agent" in r["occ_final"]]
    print("| %s | %d | %d | %d | %d | %d | %d | %d | %d | %d | %d |" % (s, len(x), len(a), len(inj),
          sum(r["injected_suffix"] and not r["injected_interior"] for r in a),
          sum(r["injected_interior"] and not r["injected_suffix"] for r in a),
          sum(r["injected_suffix"] and r["injected_interior"] for r in a),
          sum(r["graded_basis"] == "reward_only" for r in x),
          sum(r["graded_basis"] == "trigger_no_gold" for r in x),
          sum(not r["graded_new"] for r in x), len(agentocc)))
c = collections.Counter()
for r in A:
    for item in filter(None, r["occ_P"].split(";")):
        k, v = item.split(":"); s_ = dict(i.split(":") for i in r["occ_S"].split(";"))[k]
        c[(r["scaffold"], v, s_)] += 1
print("\nGold-equal occurrences, P label x S label:")
for k in sorted(c): print("  ", k, c[k])
print("unclassified occurrences:", sum(1 for r in A for i in r["occ_final"].split(";") if i and i.split(":")[1] not in ("injected", "agent")))

H("Graded fraction, Group A (old = FU11 suffix rule, new = provenance)")
print("| scaffold | n | old graded | new graded | old graded & new ungraded | old ungraded & new graded |")
print("|---|---|---|---|---|---|")
for s in SC:
    a = [r for r in A if r["scaffold"] == s]
    print("| %s | %d | %.1f%% | %.1f%% | %d | %d |" % (s, len(a), 100 * mean([r["old_g"] for r in a]), 100 * mean([r["graded_new"] for r in a]),
          sum(r["old_g"] and not r["graded_new"] for r in a), sum(r["graded_new"] and not r["old_g"] for r in a)))
print("Group B (transfer-only/empty gold), graded = a done-trigger occurred:",
      {s: "%.1f%% of %d" % (100 * mean([r["graded_new"] for r in B if r["scaffold"] == s]), len([r for r in B if r["scaffold"] == s])) for s in SC})

H("Pooled accuracy, Group A")
print("| scaffold | n | published | corrected old (FU11) | corrected new | new, embedded gold | new, alias-fixed replay |")
print("|---|---|---|---|---|---|---|")
for s in SC:
    a = [r for r in A if r["scaffold"] == s]
    print("| %s | %d | %s | %s | %s | %s | %s |" % (s, len(a), P([r["pub"] for r in a]), P([r["old_c"] for r in a]), P([r["new_c"] for r in a]),
          P([r["new_c_emb"] for r in a]), P([r["new_c_fix"] for r in a])))
u = [r for r in A if r["scaffold"] == "GEN" and not r["graded_new"]]
print("GEN ungraded (new) n=%d, offline pass rate %.3f" % (len(u), mean([(f(r["off_new"]) or 0) > 0 for r in u])))

H("Per model, Group A (published / corrected old / corrected new)")
by = collections.defaultdict(list)
for r in A: by[(r["scaffold"], r["model"])].append(r)
mods = sorted({m for _, m in by})
print("| model | GEN n | GEN pub | GEN old | GEN new | TC n | TC pub | TC old | TC new | FS pub | FS new |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
def acc(s, m, k):
    x = by.get((s, m)); return mean([r[k] for r in x]) if x else None
fmt = lambda v: "-" if v is None else "%.3f" % v
for m in mods:
    print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (m, len(by.get(("GEN", m), [])) or "-", fmt(acc("GEN", m, "pub")), fmt(acc("GEN", m, "old_c")), fmt(acc("GEN", m, "new_c")),
          len(by.get(("TOOLCALL", m), [])) or "-", fmt(acc("TOOLCALL", m, "pub")), fmt(acc("TOOLCALL", m, "old_c")), fmt(acc("TOOLCALL", m, "new_c")),
          fmt(acc("FEWSHOT", m, "pub")), fmt(acc("FEWSHOT", m, "new_c"))))

H("Generalist vs Tool-Calling per model (PRIMARY comparison; gap = GEN - TC)")
print("| model | published gap | old corrected gap | new corrected gap | new gap, embedded gold | sign pub -> new |")
print("|---|---|---|---|---|---|")
cnt = collections.Counter()
for m in mods:
    if ("GEN", m) not in by or ("TOOLCALL", m) not in by: continue
    g = {k: acc("GEN", m, k) - acc("TOOLCALL", m, k) for k in ("pub", "old_c", "new_c", "new_c_emb")}
    sg = lambda v: "+" if v > 1e-9 else ("-" if v < -1e-9 else "0")
    tr = sg(g["pub"]) + " -> " + sg(g["new_c"]); cnt["models"] += 1
    cnt["pub_below"] += g["pub"] < -1e-9; cnt["old_below"] += g["old_c"] < -1e-9; cnt["new_below"] += g["new_c"] < -1e-9
    cnt["new_tie"] += abs(g["new_c"]) <= 1e-9; cnt["new_above"] += g["new_c"] > 1e-9
    cnt["flip_pub_new"] += (g["pub"] < -1e-9) and g["new_c"] > 1e-9
    cnt["flip_pub_old"] += (g["pub"] < -1e-9) and g["old_c"] > 1e-9
    cnt["flip_pub_new_emb"] += (g["pub"] < -1e-9) and g["new_c_emb"] > 1e-9
    print("| %s | %+.3f | %+.3f | %+.3f | %+.3f | %s |" % (m, g["pub"], g["old_c"], g["new_c"], g["new_c_emb"], tr))
print(dict(cnt))
# pooled over the TC-shared models
tcm = {m for s, m in by if s == "TOOLCALL"}
for k in ("pub", "old_c", "new_c"):
    gg = [r[k] for r in A if r["scaffold"] == "GEN" and r["model"] in tcm]; tt = [r[k] for r in A if r["scaffold"] == "TOOLCALL"]
    print("pooled over TC-shared models %s: GEN %.3f (n=%d) TC %.3f (n=%d) gap %+.3f" % (k, mean(gg), len(gg), mean(tt), len(tt), mean(gg) - mean(tt)))
# 3-scaffold ordering (secondary)
first = last = 0; n3 = 0
for m in mods:
    v = {s: acc(s, m, "new_c") for s in SC if (s, m) in by}
    if len(v) < 2 or "GEN" not in v: continue
    n3 += 1; o = sorted(v, key=lambda s: -v[s]); first += o[0] == "GEN"; last += o[-1] == "GEN"
print("all-scaffold order (secondary, includes withdrawn Few-Shot), models with GEN and >=1 other: %d; GEN first %d, last %d" % (n3, first, last))

H("Write-gold subset (30 tasks)")
W = [r for r in A if r["gold_write"]]
print("| scaffold | n | graded old | graded new | published | corrected old | corrected new |")
print("|---|---|---|---|---|---|---|")
for s in SC:
    w = [r for r in W if r["scaffold"] == s]
    print("| %s | %d | %.1f%% | %.1f%% | %s | %s | %s |" % (s, len(w), 100 * mean([r["old_g"] for r in w]), 100 * mean([r["graded_new"] for r in w]),
          P([r["pub"] for r in w]), P([r["old_c"] for r in w]), P([r["new_c"] for r in w])))
for lab, gk, ck in (("old", "old_g", "old_c"), ("new", "graded_new", "new_c")):
    ug = [r for r in W if r["scaffold"] == "GEN" and not r[gk] and r[ck]]
    print("GEN %s: ungraded-but-passing on write tasks %d, of which the agent issued >=1 write itself: %d" % (lab, len(ug), sum(r["agent_writes"] > 0 for r in ug)))
ugA = [r for r in A if r["scaffold"] == "GEN" and not r["graded_new"] and r["new_c"]]
print("GEN new: ungraded-but-passing, Group A: %d (write-gold %d, read-only gold %d)" % (len(ugA), sum(r["gold_write"] for r in ugA), sum(1 - r["gold_write"] for r in ugA)))

H("Finding 3 numbers")
inj = [r for r in R if r["n_injected_actions"] > 0]
print("records with an injected gold block anywhere: %d / %d = %.1f%% ; of Group A %d / %d = %.1f%%" % (
    len(inj), len(R), 100 * len(inj) / len(R), len([r for r in inj if r["n_gold_nt"] >= 1]), len(A), 100 * len([r for r in inj if r["n_gold_nt"] >= 1]) / len(A)))
print("  suffix only %d, interior only %d, both %d" % (sum(r["injected_suffix"] and not r["injected_interior"] for r in inj),
      sum(r["injected_interior"] and not r["injected_suffix"] for r in inj), sum(r["injected_suffix"] and r["injected_interior"] for r in inj)))
print("  records with >1 injected copy: %d" % sum(r["n_injected_actions"] > r["n_gold_nt"] for r in inj))
print("  FU11 suffix rule flagged %d records" % sum(r["old_g"] for r in R))
print("injected actions per record (all records): %.2f of %.2f recorded = %.1f%%" % (mean([r["n_injected_actions"] for r in R]), mean([r["n_actions"] for r in R]),
      100 * sum(r["n_injected_actions"] for r in R) / sum(r["n_actions"] for r in R)))
for k in ("agent_tool_calls", "agent_env_calls"):
    z = [r for r in inj if r[k] == 0]
    print("records with %s == 0 that nevertheless show an injected gold block: %d %s" % (k, len(z), dict(collections.Counter(r["scaffold"] for r in z))))
A2 = [r for r in R if r["n_gold_nt"] >= 2]
ag = lambda k, xs: mean([((f(r[k]) or 0) > 0) == r["pub"] for r in xs if r[k] not in ("", None)])
print("replay agreement with recorded reward, gold >= 2 non-terminal actions (n=%d):" % len(A2))
print("  verbatim taken_actions %.3f | FU11 stripped (suffix rule) %.3f | new stripped %.3f | new stripped, embedded gold %.3f | new stripped, alias-fixed %.3f" % (
      ag("off_verbatim", A2), ag("fu11_offline", A2), ag("off_new", A2), ag("off_new_emb", A2), ag("off_new_fix", A2)))
for s in SC:
    a2 = [r for r in A2 if r["scaffold"] == s]
    print("    %s n=%d verbatim %.3f fu11 %.3f new %.3f" % (s, len(a2), ag("off_verbatim", a2), ag("fu11_offline", a2), ag("off_new", a2)))
G = [r for r in R if r["graded_new"] and r["faithful"] not in ("", None)]
print("faithful re-enactment (grader re-run at each recorded done-trigger) vs recorded reward, graded records n=%d: pinned %.3f | alias-fixed %.3f | embedded gold %.3f" % (
      len(G), ag("faithful", G), ag("faithful_fix", G), ag("faithful_emb", G)))
for s in SC:
    g = [r for r in G if r["scaffold"] == s]
    print("    %s n=%d faithful %.3f fixed %.3f emb %.3f" % (s, len(g), ag("faithful", g), ag("faithful_fix", g), ag("faithful_emb", g)))
print("graded-record offline replay of agent-only actions vs recorded (validation as in FU11):")
for s in SC:
    g = [r for r in A if r["scaffold"] == s and r["graded_new"]]
    print("    %s n=%d new stripped %.3f | old (FU11 graded set) %.3f" % (s, len(g), ag("off_new", g), ag("fu11_offline", [r for r in A if r["scaffold"] == s and r["old_g"]])))
print("embedded gold != pinned gold:", sum(1 - r["emb_gold_eq_pinned"] for r in R), "records;", dict(collections.Counter(r["scaffold"] for r in R if not r["emb_gold_eq_pinned"])))
print("recorded reward > 0 without any injected block or trigger:", sum(1 for r in R if r["pub"] and r["n_gradings"] == 0))
print("replay errors:", sum(1 for r in R if r["replay_error"]))
