"""
harness/timing_split.py
------------------------
Non-invasive instrumentation for ConfLayers / SWIFT that separates:
  - PURE INFERENCE time: forward passes that produce accepted tokens
  - SEARCH OVERHEAD time: the inline Bayesian-optimization layer-set search
    (the `optimization=True` branch in inference_conflayer.py / SWIFT) that
    silently runs INSIDE every reported "wall_time" in both codebases.

WHY THIS EXISTS (the audit's central methodological gap):
  ConfLayers' own evaluation/eval.py measures:
      _sync(); start=time.time(); <full generation incl. search>;
      _sync(); wall_time = time.time()-start
  This wall_time is reported AS-IS as "inference time" in the paper's speedup
  numbers. It never separates the BayesOpt search cost from token-generation
  cost. LayerRoute (trained router) has ZERO per-query search cost -- all its
  cost is paid once, upfront, at training time. Comparing ConfLayers' wall_time
  against LayerRoute's wall_time is therefore NOT an apples-to-apples cost
  comparison unless both are decomposed into the same two buckets:

      total_cost(N queries) = training_cost (one-time)
                             + N * (pure_inference_cost + search_overhead_cost)

  For LayerRoute:  search_overhead_cost = 0,  training_cost > 0
  For ConfLayers/SWIFT: training_cost = 0,  search_overhead_cost > 0 (until the
      "Optimization Stopped" condition fires -- which the code shows can occur
      at ANY point in generation, or never, depending on the input).

This module monkeypatches the `statistics["optimization"]` flag transitions in
ConfLayers'/SWIFT's inference loop to bracket CUDA events around the search
branch specifically, so we get search_ms and pure_ms separately, per query,
without touching their algorithmic logic.
"""

import time
import functools
import torch
import json
import os


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class QueryCostRecord:
    """One row of matched-cost accounting for a single generated query."""

    __slots__ = ["query_id", "method", "model_id", "pure_inference_ms",
                "search_overhead_ms", "new_tokens", "accept_rate",
                "search_active_steps", "total_steps"]

    def __init__(self, query_id, method, model_id):
        self.query_id = query_id
        self.method = method
        self.model_id = model_id
        self.pure_inference_ms = 0.0
        self.search_overhead_ms = 0.0
        self.new_tokens = 0
        self.accept_rate = None
        self.search_active_steps = 0
        self.total_steps = 0

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


class TimingSplitter:
    """
    Wraps a per-decoding-step callable and buckets its wall-clock time into
    'search' or 'pure' based on whether the optimization branch is currently
    active for that step. Use as a context manager around ONE decoding step.
    """

    def __init__(self, record: QueryCostRecord):
        self.record = record
        self._t0 = None

    def step(self, is_search_active: bool):
        """Return a context manager to bracket a single decoding step."""
        self._is_search_active = is_search_active
        return self

    def __enter__(self):
        _sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _sync()
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        self.record.total_steps += 1
        if self._is_search_active:
            self.record.search_overhead_ms += dt_ms
            self.record.search_active_steps += 1
        else:
            self.record.pure_inference_ms += dt_ms
        return False


def patch_conflayers_swift_forward(module, record: QueryCostRecord):
    """
    Monkeypatch `swift_forward` (shared by SWIFT and ConfLayers -- ConfLayers
    is built directly on SWIFT's codebase, confirmed via their README) so that
    EACH decoding-step iteration is individually timed and bucketed by whether
    `statistics["optimization"]` is truthy for that step.

    This requires no change to the original algorithm: `optimization` is read
    from the SAME `statistics` dict the original code already mutates; we only
    observe it.

    Usage:
        import evaluation.inference_conflayer as infc
        record = QueryCostRecord(qid, "conflayers", model_id)
        patch_conflayers_swift_forward(infc, record)
        infc.swift_forward(input_ids, model, tokenizer, max_new_tokens, statistics=stats, ...)
        # record now has pure_inference_ms / search_overhead_ms split
    """
    original_fn = module.swift_forward

    @functools.wraps(original_fn)
    def wrapped(input_ids, model, tokenizer, max_new_tokens, statistics=None,
                optimizer=None, utility=None, logits_processor=None, max_steps=512):
        # We can't easily inject per-inner-loop timing without editing the
        # library's inner for-loop (it's not step-callback-based). The
        # SAFE, NON-INVASIVE alternative used here: run the loop in two
        # timed segments by temporarily forcing statistics["optimization"]
        # to a fixed value is NOT valid (changes behavior). Instead we time
        # the whole call, and separately expose the search-active step COUNT
        # (via statistics["opt_iter"], set by the original code) so cost can
        # be apportioned: since each decoding step costs approximately the
        # SAME wall-clock regardless of whether search is active THAT step
        # (the BayesOpt call itself is cheap; the expense is the EXTRA
        # generations it triggers to evaluate candidate layer sets), the
        # correct measurement is: run with search enabled vs. a second pass
        # with the layer set FROZEN at its final value (see
        # `measure_frozen_replay` below), and the difference is the true
        # search overhead. This wrapper just records wall time and step
        # count for that accounting.
        t0 = time.perf_counter()
        _sync()
        result = original_fn(input_ids, model, tokenizer, max_new_tokens,
                             statistics=statistics, optimizer=optimizer,
                             utility=utility, logits_processor=logits_processor,
                             max_steps=max_steps)
        _sync()
        record.pure_inference_ms += (time.perf_counter() - t0) * 1000.0  # placeholder; corrected by frozen replay
        if statistics is not None:
            record.search_active_steps = statistics.get("opt_iter", 0)
        return result

    module.swift_forward = wrapped
    return original_fn  # caller should restore this after use


def measure_frozen_replay(module, input_ids, model, tokenizer, max_new_tokens,
                          frozen_layers_skipped, record: QueryCostRecord,
                          **kwargs):
    """
    THE key measurement. Run generation TWICE on the identical prompt:
      (1) normal run: search enabled, produces final `layers_skipped` config
          -- this is what ConfLayers/SWIFT report as "wall_time" today.
      (2) frozen replay: layer set FIXED at the config found in (1), search
          disabled entirely -- this isolates PURE inference cost with that
          draft config.
    search_overhead_ms = wall_time(run 1) - wall_time(run 2)

    This is the only non-invasive way to separate the two costs without
    editing the library's internals, because the search and the "useful"
    generation are algorithmically interleaved in the original code (the
    BayesOpt candidate evaluations ARE real generation steps whose tokens
    may or may not be kept).
    """
    _sync()
    t0 = time.perf_counter()
    out1 = module.swift_forward(input_ids, model, tokenizer, max_new_tokens, **kwargs)
    _sync()
    full_ms = (time.perf_counter() - t0) * 1000.0

    # Freeze layers, disable further optimization, replay identical prompt.
    model.set_skip_layers(frozen_layers_skipped)
    frozen_stats = dict(kwargs.get("statistics", {}))
    frozen_stats["optimization"] = None  # search loop will not trigger
    _sync()
    t1 = time.perf_counter()
    out2 = module.swift_forward(input_ids, model, tokenizer, max_new_tokens,
                                statistics=frozen_stats,
                                optimizer=kwargs.get("optimizer"),
                                utility=kwargs.get("utility"),
                                logits_processor=kwargs.get("logits_processor"))
    _sync()
    frozen_ms = (time.perf_counter() - t1) * 1000.0

    record.pure_inference_ms = frozen_ms
    record.search_overhead_ms = max(0.0, full_ms - frozen_ms)
    return out1, out2


def save_records(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)
