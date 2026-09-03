# Measurement issues in HAL, tau-bench and tau2-bench (disclosure, 2026-09-03)

Reproducible measurement issues found by re-deriving public agent-leaderboard artifacts offline. No agent was run and no model was called; everything comes from the published trace archives, the harness source and the live leaderboards.

- `HAL_report_for_maintainers.md`: three items on the Holistic Agent Leaderboard (cost double count, ungraded Generalist episodes on TAU-bench Airline, grader-injected actions in the public traces). Sent by email to the HAL corresponding authors on 2026-09-03; the harness repository is archived, so there is no issue tracker. A short pointer was posted on the `agent-evals/hal_traces` dataset page.
- `tau-bench_calculate_reward_mutates_actions.md`: filed as an issue on `sierra-research/tau-bench`.
- `tau2-bench_regrade_silent_score_changes.md`: filed as an issue on `sierra-research/tau2-bench`.
- `repro_eval_instrument_2026-09-03.zip`: reproduction scripts (CPU only) with a README listing the exact commits and versions.

Corrections are welcome: open an issue on this repository. If any claim here turns out to be wrong or already handled, it will be amended and the change noted.

Yangze Liu
