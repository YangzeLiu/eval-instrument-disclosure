Title: `Env.calculate_reward` replays gold through `self.step`, appending gold actions to `self.actions`

**Relation to PR #89.** PR #89 (open) already fixes half of this: behind `TAU_PRESERVE_DATA_ON_REWARD=1` it snapshots and restores `self.data` around the gold replay. It leaves `self.actions` untouched, and it is opt-in. This issue is about the `self.actions` half, and argues for both being on by default.

**Summary.** In `tau_bench/envs/base.py`, `step` appends every action to `self.actions` (`self.actions.append(action)` at L91). `calculate_reward` (L124) reloads the data (L133) and replays the gold actions by calling `self.step` (L134-136), skipping only actions in `terminate_tools` (L135). After grading, `self.actions` contains the agent's actions followed by the gold sequence minus its terminating actions, and `self.data` is left in the gold state. Any harness that records `env.actions` after the episode, or continues to act after a mid-episode grading call, is affected. This is still the behaviour on `main` (`59a200c`), where the file is byte-identical to the commonly pinned `14bf0ef`.

**Observed consequence downstream.** In the public HAL trace archives (`agent-evals/hal_traces`), 1,523 of 2,334 airline records (65.3%) carry the gold sequence as a suffix of `taken_actions`; 58 records show the full gold tool sequence although the agent made zero tool calls; and on the 1,259 records whose gold has at least two non-terminal actions, replaying the verbatim recorded actions reproduces the recorded reward in 70.3% of cases, against 91.4% once the suffix is stripped (93.4% when graded against each record's own embedded gold).

**Reproduction.** Against `main` today:

```python
from tau_bench.envs.airline import MockAirlineDomainEnv
from tau_bench.envs.base import Action

# user_strategy="human" avoids needing a model provider; the default UserStrategy.LLM
# raises ValueError("LLM user strategy requires a model provider") at construction.
env = MockAirlineDomainEnv(user_strategy="human", task_index=0)
env.step(Action(name="get_user_details", kwargs={"user_id": "mia_li_3668"}))
print([a.name for a in env.actions])   # ['get_user_details']
h0 = env.get_data_hash()
env.calculate_reward()
print([a.name for a in env.actions])   # ['get_user_details', 'book_reservation']  <- gold appended
print(h0 != env.get_data_hash())       # True  <- data left in the gold state
```

`repro/hal/goldleak.py` in `repro_eval_instrument_2026-09-03.zip` (attached, and mirrored at https://github.com/YangzeLiu/eval-instrument-disclosure) measures the extent on the public traces.

**Suggested fix.** Replay the gold actions through a private helper that does not touch `self.actions` (or snapshot and restore `self.actions` and `self.data` around the replay, which is what #89 does for `self.data`). This cannot change any score: the only reward path that reads `self.actions` is the output check at L144-161, which scans for `respond` actions, and no airline (0 of 50) or retail (0 of 115) gold task contains a `respond` action, so the injected suffix can never satisfy or break an output check.
