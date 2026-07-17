# The fully-async parameter-sync cycle

How `G`, `staleness_threshold`, `trigger_parameter_sync_step`, and `require_batches`
interact in verl's **fully-async** (off-policy) GRPO loop, and how the
`scripts/timing_summary.py` "Parameter-sync cycle" section surfaces it.

Numbers below come from `grpo_gsm8k.yaml` and are cross-checked against the real run
`slurm-2610398.out`.

## The four knobs and the derived quantities

From `grpo_gsm8k.yaml`:

- `n: 8` → **G = 8** (responses generated per prompt)
- `ppo_mini_batch_size: 32`, `require_batches: 2`
- `staleness_threshold: 0.1`
- `trigger_parameter_sync_step: 4`
- `total_rollout_steps: 51200`

The rollouter computes three derived numbers at startup
(`fully_async_rollouter.py`, `set_max_required_samples`):

```text
required_samples     = ppo_mini_batch_size * require_batches              = 32 * 2  = 64
max_required_samples = required_samples * (staleness_threshold + 1) * trigger_parameter_sync_step
                     = 64 * 1.1 * 4                                       ≈ 281
total_train_steps    = total_rollout_steps / (required_samples * trigger_parameter_sync_step)
                     = 51200 / (64 * 4)                                   = 200
```

`max_queue_size` is also set to `max_required_samples` (≈ 281).

## The two players

Fully-async means a **rollouter** and a **trainer** run as independent loops connected
by a **message queue**.

- **Rollouter**: continuously samples prompts, generates `G = 8` completions each, pushes
  finished samples into the queue. Each push bumps `staleness_samples += 1`.
- **Trainer**: pulls `required_samples = 64` out of the queue, does one local training step
  (an optimizer pass over `require_batches = 2` mini-batches), repeats.

## How `trigger_parameter_sync_step = 4` works (trainer side)

The trainer does **not** ship updated weights after every step. `_fit_update_local_step`
counts `local_trigger_step` from 1→4:

```python
if local_trigger_step < trigger_parameter_sync_step:   # < 4
    local_trigger_step += 1
else:
    current_param_version += 1     # bump weight version
    local_trigger_step = 1         # reset
```

`_fit_update_weights` only actually broadcasts weights when `local_trigger_step == 1`. So
the rhythm is:

> **4 local training steps on one weight version → then one `param_sync`** (the NCCL weight
> broadcast actor→rollout, `checkpoint_engine.backend: nccl`) → version N+1.

That's the point of the knob: `param_sync` is expensive (full weight broadcast), so you
amortize it over 4 updates instead of paying it every step. Per weight version the trainer
consumes `64 × 4 = 256` samples.

## How `staleness_threshold = 0.1` works (rollouter side)

While the trainer chews through those 4 steps, the rollouter keeps generating — **against
the old weights** (it only gets new weights at the next sync). How far ahead is it allowed
to get? That's the staleness leash. `_should_pause_generation`:

```python
if staleness_samples >= max_required_samples:   # >= 281
    return True   # pause generation
```

So the rollouter may produce up to **281** samples before it must stop and wait for the next
weight sync. Compare to the 256 the trainer consumes per version:

- **256** = exactly one sync-cycle's worth (on-policy-ish).
- The extra **+25** (the `0.1` slack) = samples it's allowed to generate with *stale* weights
  to keep the pipeline full so it doesn't idle while waiting for the sync.

`staleness_threshold = 0.1` literally means: *"tolerate up to 10% of a cycle's worth of
off-policy samples in flight."* When weights sync, `reset_staleness` recomputes
`staleness_samples = active_tasks + queue_size` — whatever's still in flight stays counted as
stale against the new version.

## How `G = 8` threads through

`G` doesn't appear in these formulas directly — it sits *inside* each sample. Each unit the
rollouter produces and the trainer batches carries its group of 8 completions, which is what
`compute_grpo_outcome_advantage` needs to form the per-prompt mean/std baseline. `G` affects
**rollout cost and queue throughput** (8× the generation work per prompt), not the
staleness/sync bookkeeping. Bigger `G` → rollouter fills the queue slower → more likely the
*trainer* waits, independent of the staleness knob.

## Putting it together — one cycle

1. Rollouter streams `G = 8`-completion samples into the queue, `staleness_samples` climbing
   toward 281.
2. Trainer pulls 64 at a time, does step 1,2,3,4 on weight version N (`local_trigger_step`
   1→4).
3. After step 4, `current_param_version` → N+1, `param_sync` broadcasts new weights.
4. `reset_staleness` fires: rollouter resumes (if it had paused at 281), staleness recounted
   against N+1.
5. Repeat for `total_train_steps = 200` steps.

## The tuning intuition

- **Smaller `staleness_threshold` (0.1)** → tight leash → data stays near on-policy → more
  *stable* GRPO, but the rollouter pauses sooner and **idles** waiting for syncs (throughput
  cost).
- **Larger `trigger_parameter_sync_step`** → fewer expensive `param_sync` broadcasts → higher
  throughput, but the rollouter spends more steps on stale weights → more off-policy.
- These two **fight each other**: raising `trigger_parameter_sync_step` widens the on-policy
  window (256 grows), so to keep the same *fraction* of stale data you'd lower
  `staleness_threshold` — and `max_required_samples` multiplies both, so the formula couples
  them exactly.

## What the run actually did (`slurm-2610398.out`)

`scripts/timing_summary.py slurm-2610398.out` reports:

```text
Parameter-sync cycle (fully_async_policy)
param syncs observed:   88   (param_version 0 -> 87)
param_sync per broadcast: avg   2.28s  min   2.11s  max   8.96s   (total 201.0s)
sync cadence (trigger_parameter_sync_step): 4   (4 local steps per weight version)
local_trigger_step observed max: 4   (matches config)
rollouter pauses:   87   (87 staleness-leash, 0 full-queue)
  staleness leash hit at staleness_samples >= 281 (max_required_samples)
```

Which confirms the walkthrough exactly:

- **88 weight versions** (0→87), each from **4 local steps** (cadence 4).
- **Staleness leash hit at `>= 281`** = `required_samples(64) × 1.1 × 4 ≈ 281`, the derived
  `max_required_samples`.
- **87 staleness-leash pauses, 0 full-queue** → the rollouter races ahead and waits on the
  leash every version. With `rollouter/idle_ratio` high and `trainer/idle_ratio` low, the run
  is **TRAINER-BOUND**: the lever is trainer throughput (or a larger
  `trigger_parameter_sync_step` to sync less often), not the staleness threshold — the leash
  isn't what's binding.

---

# Interpreting timings in async (why the numbers stop being simple)

In **sync** PPO, `timing_s/gen` is unambiguous: the controller calls `generate_sequences()`,
*blocks*, and the timer wraps that call. The phases sum to the step. Easy.

In **async**, generation runs continuously on separate rollouter workers, decoupled from the
trainer's update. So "the time to generate the samples" has no single start/end. There are
**three different "generation" numbers**, each answering a different question:

## 1. `timing_s/gen` (trainer side) — this is NOT generation time

`_fit_generate` (`fully_async_trainer.py:461`):

```python
with marked_timer("gen", timing_raw, color="red"):
    epoch, batch = await self._get_samples_from_queue()
```

So the `gen` row in the phase pie wraps `_get_samples_from_queue()` — the time the **trainer
blocked waiting for the queue** to have a batch, i.e. *trainer starvation*, NOT generation.

- Start: trainer asks the queue for `required_samples`.
- End: that many already-finished samples have been pulled.

Near-zero ⇒ rollouter keeps up; large ⇒ trainer is starved. This is why
`trainer/idle_ratio = timing_s/gen / timing_s/step` (`detach_utils.py:327`) — a queue-wait in
the numerator makes sense as an idle ratio; a generation time would not. The parser relabels
this row `gen (queue-wait)` in async mode with a footnote so it isn't misread.

`trainer/idle_ratio` is a fraction of **time**, not of steps: both terms are `time_sum`-aggregated,
so it is `Σ gen_seconds / Σ step_seconds ∈ [0,1]`. 1 ms of wait inside a 100 s step is ≈ 0.00001,
not 1.

**Averaging caveat (how the script reports ratios).** `timing_summary.py` takes the unweighted
arithmetic mean of each metric's per-row logged values. For ratio metrics that is a *mean of
per-row ratios*, which differs from verl's duration-weighted *ratio-of-sums* (`Σgen/Σstep`) unless
all rows have equal length. The interpretation is identical; only the last digit may not match a
single whole-run figure. (The `--cycles` `trn_idle` column avoids this — it reads verl's
already-computed per-version `idle_ratio` directly.)

## 2. `fully_async/processing_time/*` — actual per-*sample* generation latency

`addition_process` (`detach_utils.py:77`) reads `item["generate_sequences"]` — the wall-time of
*one* sample's generate call on the engine — reduced to avg/tp50/tp95/tp99/max
(`detach_utils.py:134`).

- Start: that one sample began decoding on a rollout worker.
- End: it hit EOS or `max_response_length`.

These boundaries live on the rollouter, **before** the trainer pulled the batch. It's per-sample,
not per-cycle: the samples in one batch were decoded concurrently, at different absolute times.

## 3. `fully_async/rollouter/{active_time, version_time}` — rollouter wall-clock per version

`reset_staleness` (`fully_async_rollouter.py:477`, with `step_start_time` reset at line 493):

```python
version_time = time.time() - step_start_time     # whole version window
active_time  = ... (version_time minus idle)
idle_ratio   = 1 - active_time / version_time
```

- Start: the previous weight-sync (`step_start_time` reset each version).
- End: the next weight-sync's `reset_staleness`.

This **is** "wall-clock between two weight updates, as seen by the rollouter."

**A high rollouter `idle_ratio` means throttled, not starved.** "Starved" = waiting for input you
can't get — that is the *trainer* when the queue is empty. The rollouter always has prompts; when
its `idle_ratio` is high it has been *paused on purpose* (`_should_pause_generation`: staleness leash
or full queue) because it out-produces the trainer. So a high rollouter idle means it is
**over-provisioned relative to the trainer ⇒ trainer-bound**, the opposite of starvation.

| Question | Metric | Clean start/end? |
|---|---|---|
| How long did the trainer wait for a batch? | `timing_s/gen` (queue-wait) | Yes — per cycle |
| How long to generate one sample? | `processing_time/*` (per sample) | Yes — but per sample, not per cycle |
| How busy was the rollouter this version? | `rollouter/active_time` ÷ `version_time` | Yes — but per *version window* |
| How long to generate *this cycle's batch*? | — | **No such interval exists** |

## "step" in the async phase pie is a weight-version CYCLE, not a training step

A subtle but important consequence of how verl aggregates: the `MetricsAggregator` uses the
`time_sum` rule for any `timing_s/` metric (`detach_utils.py:238,247`) and flushes **once per
weight version**. So in a fully-async log, `timing_s/step` (and `update_actor`, `gen`, …) is the
**sum over the `trigger_parameter_sync_step` training steps of that version**, logged once per
version — not a single training step. The data confirms it: on each trainer line
`training/global_step` jumps by 4 (= the cadence) while `current_param_version` increments by 1,
and there are exactly as many `timing_s/step` lines as versions.

Practical upshot: in `timing_summary.py`'s phase pie, **"sec/step" in async means seconds per
weight-version cycle** (the same window as `--cycles`/`version_time`), so it can be read alongside
the cycle table. In sync PPO one row really is one training step. The script prints a `NOTE (async)`
header to flag this.

---

# Defining a "rollout cycle" as the time between two weight updates

This is the natural definition (and verl already computes it as `rollouter/version_time`). The
boundary **event** is clearly detectable: `current_param_version` increments, a `param_sync` line
is logged (keyed by version), and the rollouter brackets the window `step_start_time →
reset_staleness`. The problem is not detection — it's that the **window is a poor unit of work**:

1. **The boundary slices through in-flight generations.** With `partial_rollout: True` a sample
   started under version N finishes under N+1 — exactly what `partial_ratio` /
   `param_version_diff` measure (`detach_utils.py:151`). A sample belongs to two cycles at once
   (`min_global_steps ≠ max_global_steps`), so you can detect the time boundary but **cannot
   cleanly assign a sample to a cycle** (one-to-many).
2. **"Generated in window N" ≠ "trained on in version N."** The staleness slack means the
   rollouter pre-generates ahead, so samples produced during window N feed versions N, N+1, …
   Production and consumption are pipelined and offset.
3. **The window bundles everything.** With `trigger_parameter_sync_step=4` one window spans 4
   trainer steps + the `param_sync` broadcast + any pause idle + checkpoint/validation. That's
   why verl decomposes `version_time` into `active_time` / `idle_ratio` instead of stopping at the
   raw window.
4. **The boundary isn't a single global instant.** It's a trainer-side decision propagated by an
   NCCL broadcast; each replica swaps weights slightly later and learns of it at its own
   `reset_staleness`. Across replicas with in-flight requests, the "moment" is a short skew window.
5. **Cycles aren't equal-length.** `param_sync` ranged 2.1–9.0s in the real run; pauses and
   checkpoints vary; v0 (dummy `load_format`) and the final cycle are truncated/atypical. Comparing
   cycle N to cycle M mixes apples and oranges unless you first subtract idle/sync/checkpoint.

**Good for:** utilization — `version_time` split by `active_time` / `idle_ratio` answers "between
two syncs, how busy was the rollouter?" **Not** generation time, **not** a clean sample set, **not**
the data the matching training version consumed.

## The `--cycles` flag

`scripts/timing_summary.py <log> --cycles` brackets the run by weight version and prints, per
cycle, `param_sync_s` / `version_time_s` / `active_s` / `roll_idle` / `trn_idle` — joined **by
weight version** (rollouter `step == version`, `param_sync` `current_param_version`, and the
trainer's per-version `fully_async/count/current_param_version`). `partial_ratio` is reported as a
distribution and the straddling samples are explicitly flagged rather than pretending the bucket is
clean. `--warmup` / `--last` apply to cycles (warmup drops the atypical v0 startup).

`roll_idle` and `trn_idle` are the two lanes' idle fractions (rollouter `1 - active/version`;
trainer `sum(gen)/sum(step)`). They read opposite ends — `roll_idle` HIGH points to TRAINER-bound,
`trn_idle` HIGH to ROLLOUT-bound — and are **not additive** (the lanes overlap in wall-clock; see
the mental model). On `slurm-2610398.out` the steady-state cycles look like:

```text
 ver  param_sync_s  version_time_s   active_s  roll_idle   trn_idle
  86          2.20           84.34      16.21      0.808      0.103
  87          2.16           85.10      16.21      0.809          -
...
roll_idle        min   0.808  mean   0.813  max   0.819   (HIGH => rollouter idle, waiting on syncs)
trn_idle         min   0.096  mean   0.100  max   0.105   (HIGH => trainer idle, waiting on the queue)
straddling (partial) samples: partial_ratio  min 0.000  mean 0.000  max 0.000  (87 batches)
```

i.e. ~85s wall-clock per cycle but only ~16s active → the rollouter is **idle ~81% of every
cycle** (`roll_idle≈0.81`) while the trainer is busy (`trn_idle≈0.10`) — the trainer-bound picture,
and note `0.81 + 0.10 ≠ 1` because the two lanes run concurrently. `partial_ratio=0` because here
the rollouter always finishes and pauses *before* the sync, so nothing straddled. The last version
shows `trn_idle = -` (88 rollouter versions but 87 trainer-aggregated lines — the final version
hadn't logged its trainer aggregation).
