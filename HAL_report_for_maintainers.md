**Subject:** Three reproducible measurement issues in HAL's published costs, TAU-bench Generalist scores, and trace archives

Dear Sayash and Arvind,

We re-derived parts of the HAL leaderboard from the published artifacts alone, treating the released numbers as measurements to be checked rather than as ground truth. No agent was run and no model was called: everything below comes from the encrypted trace archives on `agent-evals/hal_traces`, the harness source, and the live leaderboard. We found three issues. The first inflates the published cost column. The second causes most HAL Generalist episodes on TAU-bench Airline to go ungraded, which understates that agent by a large margin. The third means the public trace archives record grader-written actions as if the agent had taken them. All three reproduce on CPU from public data with the scripts attached.

We are writing by email because the harness repository is archived and no longer accepts issues. The upstream half of Item 3 is tau-bench behaviour, so we are filing it there today together with a separate τ²-bench report, and we are leaving a short pointer on the `agent-evals/hal_traces` dataset page so that people who download the traces know to strip the injected actions. This report and the scripts are mirrored at https://github.com/YangzeLiu/HAL. We would welcome corrections: if any of the three is wrong, or already known and handled somewhere we did not look, please tell us and we will amend the public copies and say so. We are also happy to co-write a correction note if that would be useful.

---

## Item 1. `total_usage` and `total_cost` double count nested Weave calls

`hal/utils/weave_utils.py::get_total_cost` (L843) fetches calls with `CallsFilter(trace_roots_only=False)` (L855), requesting `columns=["summary"]` (L856), and sums `summary.usage` over every call returned (L866-867, L892, L905-908). When a `litellm.completion` parent span has an `openai.chat.completions.create` child span, both carry the same usage block, so every token is counted twice. There is no de-duplication in the module. The same `trace_roots_only=False` appears in `fetch_weave_calls` (L805) and in `get_task_cost` (L1130), so per-task costs are affected the same way. The sum reaches the leaderboard through `hal/benchmarks/base_benchmark.py:178` -> `:244` -> `:254-255`.

Measured on the archives:

- In 133 of 134 runs with Weave logs, `total_usage` equals the sum over all calls; it equals the root-only sum in only 23 runs, which are the runs with no nesting.
- 70,403 of 78,551 nested parent-child pairs share a response `output.id`.
- The reported-to-root token ratio is exactly 2.000 (+/- 0.001) in 70 runs, 1.000 in 46 runs, and in between in 18 runs with mixed client paths.
- Across TAU-bench Airline, ScienceAgentBench and CORE-Bench Hard, 94 of 98 leaderboard rows have an archive with Weave logs, and 44 of them carry a doubled `total_cost` in the archive's own results field: $3,007.06 against $1,503.45 deduplicated. The cells rendered on the site for the same rows sum to $1,937.00. We do not know why the two totals differ and we make no claim about it; the duplicated token counts are upstream of both, and the archive values are what downstream users download.
- Over the 133 runs whose calls carry usage data, the reported total is $10,123 against $7,415 deduplicated, a 36.5% overstatement. One CORE-Bench run with an empty usage block on every call is excluded because neither sum can be formed for it.
- Spot check: the archive for TAU-bench Airline / Tool Calling / Claude Opus 4.1 High (August 2025) reports $149.98; deduplicated it is $74.99. That cell still reads $149.98 on the site today.
- The inflation is scaffold dependent (SciCode Zero Shot 1.00x, other SciCode scaffolds 2.00x), so cost-accuracy Pareto frontiers are distorted rather than uniformly shifted.

Two existing statements are adjacent but do not cover this. PR #114 fixed a cached-token double count *inside* a single usage dict; the question here is *which calls* are summed. Appendix A12 of the ICLR version says "the resume path deletes stale traces so costs are never double-counted", which concerns duplicate traces across resumed runs, not a parent span and its child inside one trace tree. Appendix A7.1 describes removing "any redundant calls related to weave logging" when preparing transcripts for Docent; the same de-duplication is not applied on the cost path.

## Item 2. The HAL Generalist is graded in only 39% of TAU-bench Airline episodes

In tau-bench, `calculate_reward` runs only inside the `if done:` branch of `Env.step` (`tau_bench/envs/base.py:114-115`), and `done` becomes true in exactly two places: a `respond` action is routed to the user simulator and the reply contains `###STOP###` (L96-99), or the agent calls a terminating tool (L108-109; on airline only `transfer_to_human_agents`). The Generalist (`agents/hal_generalist_agent/main.py`) runs the smolagents CodeAgent loop to completion, issues one closing `Action(name="respond", ...)` (L2695-2696), and exits. That closing message is unsolicited, the simulator usually does not answer it with `###STOP###`, and no further step follows, so grading never runs and the record is written with reward 0. The action type is not the issue: the Generalist's own `ask_user` tool (L2633-2651) steps `respond` actions mid-episode and reads `observation.done`. The final turn is.

Measured on 47 airline runs and 2,334 well-formed task records (16 of 2,350 entries hold a malformed `task` field and are dropped):

- Grading is observable, because the grader appends the gold actions to `isolated_env.actions` (Item 3). Among the 1,914 records whose gold has at least one non-terminal action, Tool-Calling was graded in 98.4% of episodes and Few-Shot in 98.7%. The Generalist was graded in 39.3%.
- Ungraded Generalist episodes end at a median of 9 actions, well below any step cap, and 100% of them end on `respond`.
- Re-grading offline (strip the gold suffix, replay the agent's own actions, call `calculate_reward`) reproduces the recorded reward for 97.4% / 96.9% / 87.6% of graded Few-Shot / Tool-Calling / Generalist records against `sierra-research/tau-bench@14bf0ef`, and 100.0% / 100.0% / 88.0% against each record's own embedded gold.
- Corrected pooled accuracy over all airline runs: Few-Shot 0.442 -> 0.442 (n=815), Tool-Calling 0.409 -> 0.414 (n=486), Generalist 0.300 -> 0.468 (n=613). Over the nine models run under all three scaffolds: 0.445 -> 0.445, 0.409 -> 0.414, 0.282 -> 0.458. On the 30 tasks whose gold requires a write, the Generalist moves 0.275 -> 0.384.
- Of 136 ungraded Generalist episodes that pass offline, 57 are write-gold tasks on which the agent itself performed the write.
- Among the 10 models with a Generalist run and at least one task-specific run (claude-opus-4 has no Tool-Calling run), the scaffold ordering changes for 6. Published, the Generalist is last for 9 of 10; corrected, it is first for 3, second for 4, last for 3.

Two notes on our method, so the numbers can be checked rather than trusted. First, HAL pins two tau-bench revisions: the benchmark setup script installs upstream `@14bf0ef` (`hal/benchmarks/taubench/taubench_setup.sh:5`), while the agents come from the fork `benediktstroebl/tau-bench@807e348` (`pyproject.toml:49/59/71`, `agents/hal_generalist_agent/requirements.txt:14`). The fork's airline gold differs from upstream on exactly three tasks (2, 10, 23), and the gold embedded in the run records matches the fork; that is why the re-grade is reported against both golds. Recording the tau-bench revision per run next to the leaderboard entry would remove the ambiguity. Second, we include Few-Shot because it is on the leaderboard and in the archives, not as a baseline: Appendix A5 withdraws it for test-set leakage, and its trace archives are still on Hugging Face.

Separately, and not a bug: the Generalist receives the policy document as a file (`wiki.md`, L2655-2657, instruction at L2690) while the task-specific scaffolds inject `wiki` into the system prompt. The Generalist runs with `max_steps=200` (L2683), which Appendix A8 states; the scaffolds keep the fork default `max_num_steps=30`, which is not stated, and neither is the asymmetry, where the scaffolds are compared.

## Item 3. Public trace archives record grader-injected gold actions as agent `taken_actions`

tau-bench's `Env.calculate_reward` replays the gold sequence through `self.step`, which appends each gold action to `self.actions` (`base.py:91`, `:134-136`). HAL serialises `isolated_env.actions` as `taken_actions` at six sites: `taubench_tool_calling/tool_calling.py:810`, `taubench_few_shot/few_shot.py:181`, `taubench_react/react.py:103`, `taubench_example_agent/main.py:52`, `hal_generalist_agent/main.py:2708-2710`, and `openai_codex_agent/main.py:1235`.

Measured on the same 47 runs and 2,334 records:

- The gold suffix is present in 1,523 records (65.3%), and in 79.6% of the 1,914 records where it is detectable.
- On average 3.72 of 19.95 recorded actions per affected record are grader-injected (18.6%).
- 58 records show the complete gold tool sequence although the agent made zero tool calls.
- On the 1,259 records whose gold has at least two non-terminal actions, replaying the verbatim `taken_actions` reproduces the recorded reward in 70.3% of cases; with the suffix stripped, 91.4%, or 93.4% against each record's embedded gold.
- 50 Generalist records contain the gold sequence strictly in the interior with agent actions after it: grading fired mid-episode, `calculate_reward` left `self.data` in the gold state, and the agent kept acting on that state.

The root cause is upstream in tau-bench and we have written it up there as well; the file is unchanged on `main`, and the open PR #89 covers only the `self.data` half, opt-in. We raise it here too because the archives are what people download. PR #164 audited ground-truth leakage and cleared TauBench, but that audit covered benchmark *inputs*; this is on the output side, which it did not reach.

---

## Reproduction

Attached: `repro_eval_instrument_2026-09-03.zip` (24 files, no credentials, no private paths). CPU only, no API calls, no model access.

- `repro/hal/download.sh`, `halio.py`, `parse.py` fetch and decrypt the archives with the passphrase from HAL's own published decryption script.
- `repro/hal/dupcost.py` compares `total_usage` with the all-calls sum and the root-only sum. It uses no pricing table, so the 2x in Item 1 is a token-count identity.
- `repro/hal/goldleak.py` and `replay_all.py` measure the gold suffix and re-replay recorded actions (Item 3).
- `repro/hal/fu11.py`, `fu11agg.py`, `fu11b.py` re-grade every airline record offline (Item 2).
- `repro/README.md` records the exact commits and versions used.

## Suggested fixes

1. In `get_total_cost`, either sum over root calls only (`trace_roots_only=True`) or de-duplicate by response id before summing, at all three call sites (`weave_utils.py` L805, L855, L1130), then recompute the published costs for the affected runs. A one-line note on the leaderboard about which entries were recomputed would help downstream users of the cost column. Separately, the site's cost tooltip says costs are computed without accounting for caching benefits, but `get_total_cost` does apply `CACHED_PRICE_OVERRIDES` and `cache_read_price` at L918-931.
2. Treat a Generalist episode that ends on `respond` as terminal for grading, either by calling `isolated_env.calculate_reward()` when `done` is still false or by routing the final response through the user simulator so `###STOP###` can fire, then recompute the affected rows. If you take the first route, snapshot `isolated_env.actions` before the call, or the contamination in Item 3 becomes universal rather than partial. Consider stating the policy-document placement and both step budgets next to the scaffold comparison.
3. Snapshot `isolated_env.actions` before the terminal step, or strip the trailing gold sequence before writing `taken_actions`. For the archives already published, a note on the dataset card would let downstream users strip it: the last `len(gold)` actions equal the task's gold actions with terminating actions excluded.

Thank you for publishing the archives and the harness in the first place. None of this would have been checkable otherwise, which is the point we would like to make alongside the findings.

Best regards,
Yangze Liu
B.S. in Computer Science, UIUC (2025); research assistant, Shandong University
