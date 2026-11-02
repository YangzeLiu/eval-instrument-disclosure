# Correction, 2026-11-01: Item 2 (HAL Generalist grading on TAU-bench Airline)

The September numbers for Item 2 of `HAL_report_for_maintainers.md` were overstated. `fu11.py` counted an episode as graded when the task's gold actions appeared as a suffix of `taken_actions`. HAL's notice of 2026-10-05 on the TAU-bench Airline leaderboard page points out that the injected gold sometimes sits in the interior of that list. In the Generalist this happens when the agent calls `transfer_to_human_agents`: the grader replays the gold, and the harness then appends a closing `respond`. FU11 treated 73 such graded records as ungraded and re-graded them offline, where the injected gold writes counted as the agent's own. Thanks to the HAL maintainers for the notice that made this visible.

The scripts here decide, for every action in `taken_actions`, whether the agent issued it or the grader injected it. The decision comes from the agent's own LLM outputs and the user-simulator calls, not from the action's position. Only records that were never graded are re-graded offline.

## Corrected numbers

Records whose gold has at least one non-terminal action (n = 613 Generalist, 486 Tool-Calling, 815 Few-Shot):

| | September (FU11) | corrected |
|---|---|---|
| Generalist graded fraction | 39.3% | 51.2% |
| Tool-Calling, Few-Shot graded fraction | 98.4%, 98.7% | unchanged |
| Generalist pooled accuracy (published 0.300) | 0.468 | 0.426 |
| Generalist vs Tool-Calling over their 9 shared models (published 0.282 vs 0.409) | 0.458 vs 0.414 | 0.414 vs 0.414 |
| Generalist on the 30 write-gold tasks (published 0.275) | 0.384 | 0.326 |
| Ungraded write-gold episodes that pass offline | 57 | 23, all with the write issued by the agent |
| Models where the Generalist trails Tool-Calling (published 8 of 9) | 3 of 9 | 7 of 9 |
| Records with grader-injected gold (Item 3, of 2,334) | 1,523 (65.3%, suffix only) | 1,596 (68.4%: 1,431 suffix only, 73 interior only, 92 both) |

What still holds: the Generalist is graded far less often than the task-specific scaffolds, and grading its ungraded episodes closes its published gap to Tool-Calling. What does not hold: the per-model reordering in the September report. Against Tool-Calling, one model of nine changes sign (gemini-2.0-flash), and each per-model cell holds about 41 records. The Item 3 figure of 3.72 of 19.95 injected actions per affected record was not reproduced by the new classification, which gives 2.72 of 16.82 over all records. Items 1 and 3 otherwise stand. HAL withdrew the Few-Shot agent from the leaderboard on 2025-12-08 for test-set leakage, so Tool-Calling is the comparison that matters.

## Running

Inputs are the same as in `repro/README.md`. Set `HAL_PROBE_DIR` as there, copy `repro/hal/halio.py` to `$HAL_PROBE_DIR/scripts/`, and unpack the pinned tau-bench to `$HAL_PROBE_DIR/taubench/tau-bench-14bf0ef52e595922d597a38f32d3e8c0dce3a8f8`.

```bash
python3 extract.py                                    # -> records.jsonl.gz
python3 provenance.py records.jsonl.gz all            # -> all_prov.jsonl
python3 regrade.py records.jsonl.gz all_prov.jsonl fu11_records.csv records_v2.csv
python3 tables.py records_v2.csv > tables_output.txt
```

`records_v2.csv` (one row per record, no trace content) and `tables_output.txt` are included. `fu11_records.csv` comes from `repro/hal/fu11.py`. Checks: no action is left unclassified, and re-running the grader at each recorded episode-ending step reproduces the recorded reward on all 1,951 graded records when the gold embedded in the record is used.
