# Measurement issues in HAL, tau-bench and tau2-bench (disclosure, 2026-09-03)

Reproducible measurement issues found by re-deriving public agent-leaderboard artifacts offline. No agent was run and no model was called; everything comes from the published trace archives, the harness source and the live leaderboards.

## Status (checked 2026-11-01)

- HAL: on 2026-10-07 the HAL maintainers replied that they had reproduced and confirmed items 1 to 3 on their own archive, and they added disclosures to the leaderboard pages (a cost caveat, a lower-bound note on the HAL Generalist rows for TAU-bench Airline, and a notice on using the downloaded traces). Thanks to them for the careful and quick follow-up. The reply is in [agent-evals/hal_traces discussion #9](https://huggingface.co/datasets/agent-evals/hal_traces/discussions/9).
- tau-bench: [sierra-research/tau-bench#92](https://github.com/sierra-research/tau-bench/issues/92) is open, with no maintainer reply yet.
- tau2-bench: [sierra-research/tau2-bench#502](https://github.com/sierra-research/tau2-bench/issues/502) is open, with no maintainer reply yet. A pull request from another contributor, [sierra-research/tau2-bench#508](https://github.com/sierra-research/tau2-bench/pull/508), proposes a fix for the solo-mode check and is open.

- `HAL_report_for_maintainers.md`: three items on the Holistic Agent Leaderboard (cost double count, ungraded Generalist episodes on TAU-bench Airline, grader-injected actions in the public traces). Sent by email to the HAL corresponding authors on 2026-09-03; the harness repository is archived, so there is no issue tracker. A short pointer was posted on the dataset page: [agent-evals/hal_traces discussion #9](https://huggingface.co/datasets/agent-evals/hal_traces/discussions/9).
- `tau-bench_calculate_reward_mutates_actions.md`: filed as [sierra-research/tau-bench#92](https://github.com/sierra-research/tau-bench/issues/92).
- `tau2-bench_regrade_silent_score_changes.md`: filed as [sierra-research/tau2-bench#502](https://github.com/sierra-research/tau2-bench/issues/502).
- `repro_eval_instrument_2026-09-03.zip`: reproduction scripts (CPU only) with a README listing the exact commits and versions.

Corrections are welcome: open an issue on this repository. If any claim here turns out to be wrong or already handled, it will be amended and the change noted.

Yangze Liu
