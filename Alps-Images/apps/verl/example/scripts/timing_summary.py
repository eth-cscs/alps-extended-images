"""Summarize verl's per-step timing from a console training log -- "where does a step go".

verl logs wall-clock seconds for each top-level PPO phase as ``timing_s/<phase>`` (and
``timing_per_token_ms/<phase>``) via the ``console`` logger. This script (pure stdlib) parses that
text and prints an averaged breakdown for both *sync* PPO and the *fully-async* (off-policy)
trainer; for async runs it also reports rollouter utilization, the weight-sync cycle, and
per-weight-version cycles.

Rationale, the async timing model, and worked examples live in
``scripts/fully_async_param_sync_cycle.md``; this docstring defines the quantities only.

Usage:
    python3 -m verl.trainer.main_ppo trainer.logger=['console'] ... 2>&1 | tee run.log
    python3 scripts/timing_summary.py run.log
    cat run.log | python3 scripts/timing_summary.py -          # read from stdin
    python3 scripts/timing_summary.py run.log --warmup 2       # drop the first 2 rows
    python3 scripts/timing_summary.py run.log --last 20        # only the last 20 rows
    python3 scripts/timing_summary.py new.log --vs base.log    # diff two runs (baseline vs candidate)
    python3 scripts/timing_summary.py run.log --cycles         # (async) per-weight-version cycle table

Numbers are an unweighted arithmetic mean of each metric's logged values over the selected window
(``--warmup`` drops leading warmup rows, default 1; ``--last N`` keeps the final N). Ratio metrics
are therefore a mean of per-row ratios, not verl's duration-weighted ratio-of-sums (see the md file).

UNIT OF "step": sync -> one row is one training step (phases ~sum to the step). Fully-async -> verl
sums each metric over the trigger_parameter_sync_step steps of a weight version and logs once per
version, so a row is one WEIGHT-VERSION CYCLE (the --cycles window), not a single step. The script
flags this in the async header.

QUANTITIES
==========

Phase pie (default), per phase, averaged over rows:
  sec/step    seconds the phase took per row (training step in sync; weight-version cycle in async).
  % of step   share of timing_s/step (or of the phase sum if step is absent).
  ms/token    timing_per_token_ms/<phase>: time normalized by tokens.
  (unaccounted)  step minus the sum of phases: data load / dispatch / untimed overhead (sync).
  (overlap)      shown instead when phases exceed the step (async: phases run concurrently).
  TOTAL step     timing_s/step, the authoritative per-row wall-clock.
  gen (queue-wait)   async relabel of ``gen``: time the trainer blocked in _get_samples_from_queue(),
                     NOT generation. Low => rollouter keeps up; high => trainer starved. (Per-sample
                     generation latency is processing_time/*, below.)
  perf/throughput    tokens/s/GPU.

Async pipeline health (from fully_async/* metrics):
  trainer idle_ratio     sum(timing_s/gen)/sum(timing_s/step): fraction of TIME the trainer blocked
                         on the queue. LOW => trainer-bound.
  rollouter idle_ratio   1 - active_time/version_time: fraction of a version window the rollouter was
                         paused (staleness leash / full queue). HIGH => rollouter throttled /
                         over-provisioned => trainer-bound.
  sample gen latency avg/tp50/tp95/tp99/max   per-sample generation wall-time on the engine
                         (fully_async/processing_time/*); tail percentiles expose stragglers.
  staleness_samples      in-flight + queued samples generated under older weights (off-policy lag).
  dropped_stale_samples  rollout work discarded for being too stale (want 0).
  queue pending / mq     prompts waiting vs finished samples waiting to be trained on.
  partial_ratio          fraction of samples whose generation straddled a weight update.

Per-rank wall-time distribution (only with VERL_PROFILE_NODE_TIMING=1):
  min/mean/median/max/spread of a phase's wall-time across ranks. spread = max-min; large spread =>
  one rank/node is the straggler gating that phase.

Parameter-sync cycle (from trainer/rollouter diagnostic lines):
  param syncs / version range   count of weight broadcasts and the param_version span.
  param_sync per broadcast      avg/min/max seconds of the actor->rollout weight broadcast.
  sync cadence                  trigger_parameter_sync_step: training steps per weight version.
  rollouter pauses              count, split into staleness-leash vs full-queue.

Per-weight-version cycles (--cycles), one row per version:
  param_sync_s     the weight broadcast that closed the cycle.
  version_time_s   rollouter wall-clock for the version window (fully_async/rollouter/version_time).
  active_s         of that, time actually generating.
  roll_idle        rollouter idle fraction = 1 - active/version.
  trn_idle         trainer idle fraction = sum(gen)/sum(step) (fully_async/trainer/idle_ratio).
  roll_idle HIGH => trainer-bound, trn_idle HIGH => rollout-bound; the two are NOT additive (the
  lanes run concurrently). These are UTILIZATION, not a clean per-cycle batch -- see the md file.
"""

import argparse
import re
import sys
from collections import defaultdict

# Matches "timing_s/<name>", "timing_per_token_ms/<name>", or the bare "step" counter,
# followed by ":" and a numeric value (int/float/scientific, plus nan/inf).
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf"
_TOKEN_RE = re.compile(
    rf"((?:[A-Za-z0-9_-]+/)?timing_s/[A-Za-z0-9_./-]+"
    rf"|timing_per_token_ms/[A-Za-z0-9_./-]+"
    rf"|(?:[A-Za-z0-9_-]+/)?perf/[A-Za-z0-9_./-]+"
    rf"|fully_async/[A-Za-z0-9_./-]+"
    rf"|step)"
    rf"\s*:\s*({_NUM})"
)

# Curated fully-async (off-policy pipeline) metrics that are relevant to *profiling* -- i.e.
# whether the bottleneck is the trainer or the rollouter, and whether samples are wasted.
# Confirmed against verl/experimental/fully_async_policy/detach_utils.py and README.md.
# Deliberately excludes fully_async/static/* (config constants) and learning metrics
# (actor/critic/loss/...), which are not what we're profiling for.
_ASYNC_HEALTH = [
    ("fully_async/trainer/idle_ratio", "trainer idle_ratio", "time frac sum(gen)/sum(step); LOW => trainer-bound"),
    ("fully_async/total_wait_time", "trainer total_wait_time", "s the trainer blocked on the queue"),
    ("fully_async/rollouter/idle_ratio", "rollouter idle_ratio", "HIGH => rollouter throttled/over-provisioned"),
    ("fully_async/rollouter/active_time", "rollouter active_time", "s"),
    ("fully_async/rollouter/version_time", "rollouter version_time", "s"),
    ("fully_async/processing_time/avg", "sample gen latency avg", "s"),
    ("fully_async/processing_time/tp50", "sample gen latency tp50", "s"),
    ("fully_async/processing_time/tp95", "sample gen latency tp95", "s"),
    ("fully_async/processing_time/tp99", "sample gen latency tp99", "s"),
    ("fully_async/processing_time/max", "sample gen latency max", "s straggler sample"),
    ("fully_async/count/staleness_samples", "staleness_samples", "off-policy lag"),
    ("fully_async/count/dropped_stale_samples", "dropped_stale_samples", "wasted rollout work"),
    ("fully_async/monitor/queue/pending_queue_size", "queue pending", "backpressure"),
    ("fully_async/monitor/queue/mq_queue_size", "queue mq", "ready samples waiting"),
    ("fully_async/partial/partial_ratio", "partial_ratio", "fraction of partial-rollout samples"),
]

# Maps a (role, phase) worker-timing group to a friendly PPO phase name. The per-node
# distribution comes from engine_workers logging perf/worker_time_<phase>/<stat>, renamed
# with the role prefix (actor/critic/ref) by the trainer.
_WORKER_TIME_RE = re.compile(r"(?:([A-Za-z0-9_-]+)/)?perf/worker_time_(train|infer)/(min|max|mean|median)")
_PHASE_LABEL = {
    ("actor", "train"): "update_actor",
    ("actor", "infer"): "old_log_prob",
    ("critic", "train"): "update_critic",
    ("critic", "infer"): "values",
    ("ref", "infer"): "ref",
}

# Free-text fully-async cycle lines (printed by fully_async_trainer / fully_async_rollouter).
# These are NOT the aggregated "step:N - key:value" lines, so they need their own patterns.
#   trainer: "_fit_update_weights, timing_s/param_sync: <s> seconds ... current_param_version: <v>"
_PARAM_SYNC_RE = re.compile(
    r"_fit_update_weights,\s*timing_s/param_sync:\s*([\d.]+)\s*seconds.*?current_param_version:\s*(\d+)"
)
#   trainer: "global_steps: <g> local_trigger_step: <l> trigger_parameter_sync_step: <k>"
_LOCAL_STEP_RE = re.compile(
    r"global_steps:\s*(\d+)\s+local_trigger_step:\s*(\d+)\s+trigger_parameter_sync_step:\s*(\d+)"
)
#   rollouter: "[ShouldPause] ... staleness_samples <n> >= max_required_samples <m>"
_PAUSE_STALENESS_RE = re.compile(r"staleness_samples\s+(\d+)\s*>=\s*max_required_samples\s+(\d+)")
#   rollouter: "[ShouldPause] ... due to full queue: size=<n>, max=<m>"
_PAUSE_QUEUE_RE = re.compile(r"full queue:\s*size=(\d+),\s*max=(\d+)")
# Trainer aggregated line (one per weight version). It carries multiple step: tokens, so we key
# on the explicit version field rather than the step counter. idle_ratio here is verl's own
# per-version sum(timing_s/gen)/sum(timing_s/step) (ratio-of-sums), so we read it directly.
_TRAINER_VERSION_RE = re.compile(rf"fully_async/count/current_param_version\s*:\s*({_NUM})")
_TRAINER_IDLE_RE = re.compile(rf"fully_async/trainer/idle_ratio\s*:\s*({_NUM})")


def _to_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _normalize_key(key):
    """Collapse an accidental doubled ``timing_s/timing_s/`` prefix to a single one.

    Some trainers (e.g. the fully-async trainer) pass an already-prefixed name like
    ``timing_s/param_sync`` to the timer, which the logging layer prefixes again, yielding
    ``timing_s/timing_s/param_sync``. Normalize so it shows up as the intended ``param_sync``.
    """
    while key.startswith("timing_s/timing_s/"):
        key = key[len("timing_s/") :]
    return key


def parse_log(lines):
    """Parse console log lines into a list of per-step metric dicts.

    Two kinds of line are kept and merged by step number:
      * trainer step lines (contain ``timing_s/step``) -- the phase timings, and
      * rollouter step lines (contain ``fully_async/rollouter/`` but not ``timing_s/step``) --
        emitted separately by the fully-async trainer, carrying rollouter utilization.
    Merging by step lets us show trainer- and rollouter-side async health side by side.
    Other log lines (validation, free-text progress) are ignored. Ray actor stdout prefixes
    such as ``(FullyAsyncTrainer pid=123)`` are tolerated because we scan anywhere in the line.
    """
    by_step = {}
    fallback_step = 0
    for line in lines:
        is_trainer = "timing_s/step" in line
        is_rollouter = "fully_async/rollouter/" in line and "step:" in line
        if not (is_trainer or is_rollouter):
            continue
        pairs = _TOKEN_RE.findall(line)
        if not pairs:
            continue
        record = {_normalize_key(key): _to_float(val) for key, val in pairs}
        # `step` here is the bare logged step counter, not a timing metric.
        if "step" in record:
            step = int(record.pop("step"))
        else:
            step, fallback_step = fallback_step, fallback_step + 1
        by_step.setdefault(step, {}).update(record)
    return [{**by_step[s], "_step": s} for s in sorted(by_step)]


def summarize(records, warmup=1, last=None):
    """Average timing across the selected window and return a printable breakdown."""
    selected = records[warmup:]
    if last is not None:
        selected = selected[-last:]
    if not selected:
        return None

    sums = defaultdict(float)
    counts = defaultdict(int)
    for rec in selected:
        for key, val in rec.items():
            if key == "_step" or val != val:  # skip the step id and NaNs
                continue
            sums[key] += val
            counts[key] += 1

    avg = {k: sums[k] / counts[k] for k in sums}
    return {
        "avg": avg,
        "n_steps": len(selected),
        "first_step": selected[0]["_step"],
        "last_step": selected[-1]["_step"],
    }


def format_table(summary):
    avg = summary["avg"]
    step_total = avg.get("timing_s/step")

    # In fully-async runs (detected by the presence of fully_async/* metrics) the trainer's
    # "gen" phase does NOT measure generation -- it wraps _get_samples_from_queue(), i.e. the
    # time the trainer blocked waiting for the rollouter to have a batch ready. Relabel it so
    # the row isn't misread as generation work; the real per-sample generation latency is in
    # the "sample gen latency" rows of the async-health section (fully_async/processing_time/*).
    is_async = any(k.startswith("fully_async/") for k in avg)
    gen_is_queue_wait = is_async and "timing_s/gen" in avg

    # Top-level phases = timing_s/<name> with no further "/". Nested sub-metrics such as
    # timing_s/agent_loop/tool_calls/max are not step phases (and would distort "% of step"),
    # so they're counted separately and noted rather than shown in the pie.
    phases, nested = [], 0
    for k, v in avg.items():
        if not k.startswith("timing_s/") or k == "timing_s/step":
            continue
        name = k[len("timing_s/") :]
        if "/" in name:
            nested += 1
        else:
            phases.append((name, v))
    phases.sort(key=lambda kv: kv[1], reverse=True)
    children_sum = sum(v for _, v in phases)
    denom = step_total if step_total else children_sum

    out = []
    out.append("")
    unit = "weight-version cycle" if is_async else "training step"
    out.append(
        f"Timing breakdown over {summary['n_steps']} rows "
        f"(steps {summary['first_step']}..{summary['last_step']}, warmup excluded)"
    )
    if is_async:
        out.append(
            "\n".join(
                [ "NOTE (async): each row / 'step' column is per WEIGHT-VERSION CYCLE, not per training ",
                  "step -- verl time-sums each metric over the trigger_parameter_sync_step steps of a ",
                  "version and logs once per version (same window as --cycles).",
                ]
            )
        )
    out.append(f"(one '{unit}' per row)")
    out.append("=" * 64)
    header = f"{'phase':<22}{'sec/step':>12}{'% of step':>12}{'ms/token':>14}"
    out.append(header)
    out.append("-" * 64)
    for name, secs in phases:
        pct = f"{100 * secs / denom:6.1f}%" if denom else "   n/a"
        per_tok = avg.get(f"timing_per_token_ms/{name}")
        per_tok_s = f"{per_tok:12.3f}" if per_tok is not None else f"{'-':>12}"
        label = "gen (queue-wait) [1]" if (name == "gen" and gen_is_queue_wait) else name
        out.append(f"{label:<22}{secs:>12.3f}{pct:>12}{per_tok_s:>14}")
    out.append("-" * 64)

    if step_total:
        out.append(f"{'(sum of phases)':<22}{children_sum:>12.3f}{100 * children_sum / step_total:>11.1f}%")
        overhead = step_total - children_sum
        if overhead >= 0:
            out.append(
                f"{'(unaccounted)':<22}{overhead:>12.3f}{100 * overhead / step_total:>11.1f}%"
                "   <- data load / dispatch / untimed"
            )
        else:
            out.append(
                f"{'(overlap)':<22}{overhead:>12.3f}{'':>12}"
                "   <- phases overlap (async mode); they exceed step time"
            )
        out.append(f"{'TOTAL step':<22}{step_total:>12.3f}{100.0:>11.1f}%")
    else:
        out.append("timing_s/step not found; percentages are relative to the sum of phases.")

    if nested:
        out.append(f"({nested} nested timing sub-metric(s) like timing_s/<x>/max hidden from the phase pie)")

    if gen_is_queue_wait:
        out.append(
            "[1] async mode: 'gen' wraps _get_samples_from_queue() = trainer time blocked waiting for\n"
            "    the rollouter's queue, NOT generation work (which overlaps prior steps on separate\n"
            "    workers). Near-zero => rollouter keeps up; large => trainer is starved. Real per-sample\n"
            "    generation latency is in the 'sample gen latency' rows below."
        )

    throughput = summary["avg"].get("perf/throughput")
    if throughput:
        out.append("")
        out.append(f"perf/throughput: {throughput:.1f} tokens/s/GPU")

    out.extend(_async_health_lines(avg))
    out.extend(_node_distribution_lines(avg))
    out.append("")
    return "\n".join(out)


def _async_health_lines(avg):
    """Render the curated fully-async pipeline-health metrics, if present.

    These answer the async-specific profiling question -- trainer-bound vs rollouter-bound,
    and how much rollout work is wasted -- which the plain phase pie cannot, because in async
    mode generation is offloaded and overlaps the trainer step.
    """
    rows = [(label, avg[key], hint) for key, label, hint in _ASYNC_HEALTH if key in avg]
    if not rows:
        return []

    lines = ["", "Async pipeline health (fully_async_policy)"]
    lines.append("=" * 70)
    for label, value, hint in rows:
        lines.append(f"{label:<26}{value:>12.3f}   {hint}")
    lines.append("-" * 70)

    # Headline interpretation: compare the two idle ratios (README's tuning signal).
    t_idle = avg.get("fully_async/trainer/idle_ratio")
    r_idle = avg.get("fully_async/rollouter/idle_ratio")
    if t_idle is not None and r_idle is not None:
        if r_idle > 0.3 and t_idle < 0.15:
            verdict = "TRAINER-BOUND: rollouter mostly idle -> give the trainer more GPUs (or rollouter fewer)."
        elif t_idle > 0.3 and r_idle < 0.15:
            verdict = "ROLLOUT-BOUND: trainer mostly idle -> give the rollouter more GPUs (or trainer fewer)."
        else:
            verdict = "Reasonably balanced pipeline (both idle ratios moderate)."
        lines.append(f"=> {verdict}")
    return lines


def _node_distribution_lines(avg):
    """Render the per-rank (across-nodes) wall-time distribution, if present in the log.

    Requires the run to be launched with VERL_PROFILE_NODE_TIMING=1 so engine_workers emit
    perf/worker_time_<phase>/<stat>. Shows max-min spread to surface stragglers: a large
    spread means one rank/node gates the phase.
    """
    # group[(label)] = {stat: value}
    groups = {}
    for key, value in avg.items():
        m = _WORKER_TIME_RE.fullmatch(key)
        if not m:
            continue
        role, phase, stat = m.group(1) or "?", m.group(2), m.group(3)
        label = _PHASE_LABEL.get((role, phase), f"{role}_{phase}")
        groups.setdefault(label, {})[stat] = value

    if not groups:
        return []

    lines = ["", "Per-rank wall-time distribution across all nodes (VERL_PROFILE_NODE_TIMING=1)"]
    lines.append("=" * 70)
    lines.append(f"{'phase':<18}{'min':>10}{'mean':>10}{'median':>10}{'max':>10}{'spread':>12}")
    lines.append("-" * 70)
    # Sort by max (the gating straggler) descending.
    for label in sorted(groups, key=lambda k: groups[k].get("max", 0.0), reverse=True):
        s = groups[label]
        mn, mx = s.get("min"), s.get("max")
        spread = f"{mx - mn:+.3f}" if (mn is not None and mx is not None) else "-"
        cells = "".join(
            f"{s[stat]:>10.3f}" if stat in s else f"{'-':>10}" for stat in ("min", "mean", "median", "max")
        )
        lines.append(f"{label:<18}{cells}{spread:>12}")
    lines.append("-" * 70)
    lines.append("spread = max - min; a large spread means one rank/node is a straggler for that phase.")
    return lines


def format_diff(base, cand, base_label, cand_label):
    """Print a per-phase diff between two summaries (baseline vs candidate).

    Rows are sorted by the magnitude of the change so the biggest movers surface first.
    A negative delta means the candidate is faster. Phases present in only one run are
    still shown (the missing side counts as 0).
    """
    base_avg, cand_avg = base["avg"], cand["avg"]
    phases = sorted(
        {
            k[len("timing_s/") :]
            for avg in (base_avg, cand_avg)
            for k in avg
            if k.startswith("timing_s/") and k != "timing_s/step"
        },
        key=lambda name: abs(cand_avg.get(f"timing_s/{name}", 0.0) - base_avg.get(f"timing_s/{name}", 0.0)),
        reverse=True,
    )

    def delta_pct(base_v, cand_v):
        if base_v:
            return f"{100 * (cand_v - base_v) / base_v:+7.1f}%"
        return "    new" if cand_v else "      -"

    out = []
    out.append("")
    out.append(f"Timing diff:  baseline = {base_label}   candidate = {cand_label}")
    out.append(f"  baseline: {base['n_steps']} steps ({base['first_step']}..{base['last_step']})")
    out.append(f"  candidate: {cand['n_steps']} steps ({cand['first_step']}..{cand['last_step']})")
    out.append("  (negative delta = candidate is faster)")
    out.append("=" * 70)
    out.append(f"{'phase':<20}{'base s':>11}{'cand s':>11}{'delta s':>11}{'delta %':>12}")
    out.append("-" * 70)
    for name in phases:
        b = base_avg.get(f"timing_s/{name}", 0.0)
        c = cand_avg.get(f"timing_s/{name}", 0.0)
        out.append(f"{name:<20}{b:>11.3f}{c:>11.3f}{c - b:>+11.3f}{delta_pct(b, c):>12}")
    out.append("-" * 70)

    b_step = base_avg.get("timing_s/step")
    c_step = cand_avg.get("timing_s/step")
    if b_step and c_step:
        out.append(
            f"{'TOTAL step':<20}{b_step:>11.3f}{c_step:>11.3f}{c_step - b_step:>+11.3f}{delta_pct(b_step, c_step):>12}"
        )

    b_tp = base_avg.get("perf/throughput")
    c_tp = cand_avg.get("perf/throughput")
    if b_tp and c_tp:
        out.append("")
        speedup = c_tp / b_tp if b_tp else float("nan")
        out.append(
            f"perf/throughput: {b_tp:.1f} -> {c_tp:.1f} tokens/s/GPU  "
            f"({delta_pct(b_tp, c_tp).strip()}, {speedup:.2f}x)"
        )

    # Async pipeline-health diff: did a rebalancing actually change trainer/rollouter
    # utilization or wasted work? Only shown when at least one run has fully_async metrics.
    health_rows = [
        (label, base_avg.get(key), cand_avg.get(key))
        for key, label, _hint in _ASYNC_HEALTH
        if key in base_avg or key in cand_avg
    ]
    if health_rows:
        out.append("")
        out.append("Async pipeline health diff")
        out.append("=" * 70)
        out.append(f"{'metric':<26}{'base':>11}{'cand':>11}{'delta':>11}{'delta %':>11}")
        out.append("-" * 70)
        for label, b, c in health_rows:
            b = b if b is not None else 0.0
            c = c if c is not None else 0.0
            out.append(f"{label:<26}{b:>11.3f}{c:>11.3f}{c - b:>+11.3f}{delta_pct(b, c):>11}")
        out.append("-" * 70)
        out.append("(idle ratios: closer to balanced is better; dropped_stale_samples: lower is better)")

    out.append("")
    return "\n".join(out)


def parse_async_cycle(lines):
    """Extract the fully-async parameter-sync cycle from the trainer/rollouter free-text logs.

    Pulls the category-(3) diagnostics that the aggregated step parser skips:
      * ``_fit_update_weights, timing_s/param_sync: <s> ... current_param_version: <v>`` -- the
        per-version NCCL weight-broadcast cost (averaged away in the phase pie),
      * ``global_steps: <g> local_trigger_step: <l> trigger_parameter_sync_step: <k>`` -- the
        cadence: <k> local training steps per weight version,
      * ``[ShouldPause] ... staleness_samples <n> >= max_required_samples <m>`` / ``full queue`` --
        when the rollouter hit its staleness leash (or queue cap) and stalled waiting for a sync.
    Together they let you watch one weight version = K local steps -> one param_sync.
    """
    syncs = []  # (version, duration_s)
    local_steps = []  # (global_step, local_trigger_step, trigger_k)
    staleness_pauses = []  # (staleness_samples, max_required_samples)
    queue_pauses = []  # (queue_size, max_queue_size)
    for line in lines:
        if m := _PARAM_SYNC_RE.search(line):
            syncs.append((int(m.group(2)), float(m.group(1))))
        elif m := _LOCAL_STEP_RE.search(line):
            local_steps.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
        elif m := _PAUSE_STALENESS_RE.search(line):
            staleness_pauses.append((int(m.group(1)), int(m.group(2))))
        elif m := _PAUSE_QUEUE_RE.search(line):
            queue_pauses.append((int(m.group(1)), int(m.group(2))))
    return {
        "syncs": syncs,
        "local_steps": local_steps,
        "staleness_pauses": staleness_pauses,
        "queue_pauses": queue_pauses,
    }


def format_async_cycle(cycle):
    """Render the parameter-sync cycle section, or "" when no cycle lines were found."""
    syncs = cycle["syncs"]
    local_steps = cycle["local_steps"]
    staleness_pauses = cycle["staleness_pauses"]
    queue_pauses = cycle["queue_pauses"]
    if not (syncs or local_steps or staleness_pauses or queue_pauses):
        return ""

    lines = ["", "Parameter-sync cycle (fully_async_policy)", "=" * 70]

    if syncs:
        durs = [d for _, d in syncs]
        versions = [v for v, _ in syncs]
        avg = sum(durs) / len(durs)
        total = sum(durs)
        lines.append(f"param syncs observed: {len(syncs):>4}   (param_version {min(versions)} -> {max(versions)})")
        lines.append(
            f"param_sync per broadcast: avg {avg:6.2f}s  min {min(durs):6.2f}s  max {max(durs):6.2f}s"
            f"   (total {total:.1f}s)"
        )

    if local_steps:
        ks = {k for _, _, k in local_steps}
        observed_max = max(l for _, l, _ in local_steps)
        k_str = ",".join(str(k) for k in sorted(ks))
        lines.append(f"sync cadence (trigger_parameter_sync_step): {k_str}   ({k_str} local steps per weight version)")
        verdict = "matches config" if observed_max in ks else f"!! exceeds configured cadence {k_str}"
        lines.append(f"local_trigger_step observed max: {observed_max}   ({verdict})")

    n_pause = len(staleness_pauses) + len(queue_pauses)
    if n_pause:
        lines.append(
            f"rollouter pauses: {n_pause:>4}   "
            f"({len(staleness_pauses)} staleness-leash, {len(queue_pauses)} full-queue)"
        )
        if staleness_pauses:
            leash = staleness_pauses[-1][1]
            lines.append(f"  staleness leash hit at staleness_samples >= {leash} (max_required_samples)")
        if queue_pauses:
            cap = queue_pauses[-1][1]
            lines.append(f"  queue backpressure hit at queue size >= {cap} (max_queue_size)")

    lines.append("-" * 70)
    lines.append(
        "One weight version = trigger_parameter_sync_step local steps; param_sync is the weight\n"
        "broadcast paid once per version. Pauses = the rollouter hit its staleness/queue limit and\n"
        "waited for the next sync (shows up as rollouter idle time)."
    )
    return "\n".join(lines)


# Mental model for a cycle (why you compare the two halves of a row but never add them):
#
#   cycle wall-clock  --------------------------------------------->
#    rollouter:  [ active_s ][ ........ idle ........ ]   sums to version_time_s
#    trainer:    [ queue_wait ][ update_actor ][ ... ]    sums to trainer_step_s
#               (both timelines run CONCURRENTLY, on different GPUs)
#
# Sum DOWN each lane (within one actor) -> valid:
#   rollouter: active_s + idle      == version_time_s   (idle_ratio = 1 - active/version)
#   trainer:   queue_wait + update_actor + ... ~= trainer_step_s (remainder = untimed overhead)
# Sum ACROSS lanes (rollouter + trainer) -> meaningless: they overlap in wall-clock, so adding
# double-counts time. version_time_s ~= trainer_step_s because both independently measure the
# SAME cycle window from two vantage points -- not because one sums into the other.
# The two idle ratios are the complementary read-outs of the two lanes:
#   roll_idle (rollouter) HIGH => rollouter waiting on syncs  (points to TRAINER-bound)
#   trn_idle  (trainer)   HIGH => trainer waiting on the queue (points to ROLLOUT-bound)
def parse_trainer_cycles(lines):
    """Per-weight-version trainer idle_ratio, keyed by ``current_param_version``.

    The trainer logs aggregated metrics once per version on a line that carries several ``step:``
    tokens, so we join on the explicit version field, not the (ambiguous) step counter. The value
    is verl's own per-version ``sum(timing_s/gen)/sum(timing_s/step)`` -- the authoritative
    ratio-of-sums -- so we read it straight from the log rather than reconstructing it.
    """
    out = {}
    for line in lines:
        v, r = _TRAINER_VERSION_RE.search(line), _TRAINER_IDLE_RE.search(line)
        if v and r:
            out[int(float(v.group(1)))] = float(r.group(1))
    return out


def parse_cycles(records, syncs, trainer_idle=None):
    """Bucket the async run into weight-version cycles ("between two weight updates").

    Joins, BY WEIGHT VERSION, the rollouter's per-version window (version_time / active_time /
    idle_ratio, emitted once per version with step == the version number), the param_sync duration
    (from the ``_fit_update_weights`` line, keyed by ``current_param_version``), and the trainer's
    per-version idle_ratio (``parse_trainer_cycles``, keyed by ``current_param_version``). This is
    the *utilization* view the cycle-boundary definition is actually good for -- it is NOT a clean
    per-cycle set of samples: partial rollouts straddle the boundary (see partial_ratio), and
    samples generated in a cycle feed later training versions (staleness slack).
    """
    cycles = {}
    for rec in records:
        if "fully_async/rollouter/version_time" in rec:
            c = cycles.setdefault(rec["_step"], {})  # rollouter step == weight version
            c["version_time"] = rec.get("fully_async/rollouter/version_time")
            c["active_time"] = rec.get("fully_async/rollouter/active_time")
            c["roll_idle"] = rec.get("fully_async/rollouter/idle_ratio")
    for version, dur in syncs:
        cycles.setdefault(version, {})["param_sync"] = dur
    for version, idle in (trainer_idle or {}).items():
        cycles.setdefault(version, {})["trn_idle"] = idle
    return [{"version": v, **cycles[v]} for v in sorted(cycles)]


def _stat(items, key):
    """(min, mean, max) over the numeric values of `key`, or None if there are none."""
    vals = [c[key] for c in items if isinstance(c.get(key), (int, float))]
    if not vals:
        return None
    return min(vals), sum(vals) / len(vals), max(vals)


def format_cycles(cycles, partials, warmup=1, last=None):
    """Render the per-weight-version cycle table + utilization summary, with straddling flagged."""
    selected = cycles[warmup:]
    if last is not None:
        selected = selected[-last:]
    if not selected:
        return (
            "No fully-async weight-version cycles found (need fully_async/rollouter/version_time "
            "or param_sync lines). Is this a fully-async run captured with the console logger?"
        )

    def cell(x, w=12, p=2):
        return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'-':>{w}}"

    out = ["", "Per-cycle rollout breakdown (fully_async_policy)  [cycle = between two weight updates]"]
    out.append("=" * 78)
    out.append(
        f"{'ver':>4}{'param_sync_s':>14}{'version_time_s':>16}{'active_s':>11}"
        f"{'roll_idle':>11}{'trn_idle':>11}"
    )
    out.append("-" * 78)
    for c in selected:
        out.append(
            f"{c['version']:>4}{cell(c.get('param_sync'), 14)}{cell(c.get('version_time'), 16)}"
            f"{cell(c.get('active_time'), 11)}{cell(c.get('roll_idle'), 11, 3)}{cell(c.get('trn_idle'), 11, 3)}"
        )
    out.append("-" * 78)

    vt, ps = _stat(selected, "version_time"), _stat(selected, "param_sync")
    ri, ti = _stat(selected, "roll_idle"), _stat(selected, "trn_idle")
    out.append(
        f"cycles: {len(selected)}   (versions {selected[0]['version']}..{selected[-1]['version']}, warmup excluded)"
    )
    if vt:
        out.append(f"version_time_s   min {vt[0]:7.1f}  mean {vt[1]:7.1f}  max {vt[2]:7.1f}")
    if ri:
        out.append(
            f"roll_idle        min {ri[0]:7.3f}  mean {ri[1]:7.3f}  max {ri[2]:7.3f}"
            "   (HIGH => rollouter idle, waiting on syncs)"
        )
    if ti:
        out.append(
            f"trn_idle         min {ti[0]:7.3f}  mean {ti[1]:7.3f}  max {ti[2]:7.3f}"
            "   (HIGH => trainer idle, waiting on the queue)"
        )
    if ps:
        total = sum(c["param_sync"] for c in selected if isinstance(c.get("param_sync"), (int, float)))
        out.append(f"param_sync_s     min {ps[0]:7.2f}  mean {ps[1]:7.2f}  max {ps[2]:7.2f}   (total {total:.1f}s)")

    if partials:
        out.append("")
        out.append(
            f"straddling (partial) samples: partial_ratio  min {min(partials):.3f}  "
            f"mean {sum(partials) / len(partials):.3f}  max {max(partials):.3f}  ({len(partials)} batches)"
        )
        out.append("  ^ samples that STARTED in one cycle and FINISHED in the next -- the weight-update")
        out.append("    boundary cuts through them, so per-cycle sample SETS are not clean (see caveat).")

    out.append("-" * 78)
    out.append(
        "roll_idle and trn_idle are the two lanes' idle fractions and are NOT added: rollouter and\n"
        "trainer run concurrently. Caveat: 'cycle' = wall-clock between two weight syncs; it bundles\n"
        "trigger_parameter_sync_step trainer steps + the param_sync broadcast + idle, and samples\n"
        "generated in a cycle feed LATER training versions (staleness slack). Read these as\n"
        "UTILIZATION, not generation time or a clean per-cycle batch. See\n"
        "scripts/fully_async_param_sync_cycle.md."
    )
    return "\n".join(out)


def read_lines(path):
    """Read a console log (or stdin when path == '-') into a list of lines (re-readable)."""
    if path == "-":
        return sys.stdin.readlines()
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.readlines()


def summarize_or_exit(path, warmup, last):
    lines = read_lines(path)
    records = parse_log(lines)
    if not records:
        sys.exit(
            f"No 'timing_s/step' lines found in {path!r}. Make sure the run used trainer.logger "
            "including 'console' and that you captured stdout (e.g. `... 2>&1 | tee run.log`)."
        )
    summary = summarize(records, warmup=warmup, last=last)
    if summary is None:
        sys.exit(
            f"Found {len(records)} step(s) in {path!r} but none left after --warmup {warmup}"
            f"{f'/--last {last}' if last else ''}. Lower --warmup."
        )
    return summary, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logfile", help="path to the console log file, or '-' for stdin")
    ap.add_argument("--vs", metavar="BASELINE", help="baseline log to diff against; prints a delta table")
    ap.add_argument("--warmup", type=int, default=1, help="number of initial steps/cycles to drop (default: 1)")
    ap.add_argument("--last", type=int, default=None, help="only average the last N steps (or show last N cycles)")
    ap.add_argument(
        "--cycles",
        action="store_true",
        help="(fully-async) show a per-weight-version cycle breakdown (version_time/active/idle/"
        "param_sync) instead of the phase pie; --warmup/--last apply to cycles",
    )
    args = ap.parse_args()

    candidate, cand_lines = summarize_or_exit(args.logfile, args.warmup, args.last)

    if args.vs:
        baseline, _ = summarize_or_exit(args.vs, args.warmup, args.last)
        print(format_diff(baseline, candidate, base_label=args.vs, cand_label=args.logfile))
    elif args.cycles:
        records = parse_log(cand_lines)
        syncs = parse_async_cycle(cand_lines)["syncs"]
        trainer_idle = parse_trainer_cycles(cand_lines)
        partials = [r["fully_async/partial/partial_ratio"] for r in records if "fully_async/partial/partial_ratio" in r]
        cycles = parse_cycles(records, syncs, trainer_idle)
        print(format_cycles(cycles, partials, warmup=args.warmup, last=args.last))
    else:
        print(format_table(candidate))
        # The parameter-sync cycle is a within-run cadence diagnostic, so show it for the
        # single-run view (not the diff, which stays focused on phase deltas).
        cycle = format_async_cycle(parse_async_cycle(cand_lines))
        if cycle:
            print(cycle)


if __name__ == "__main__":
    main()
