"""High-precision CSV exports for stimulus search."""

from collections.abc import Sequence

from binary_entropy.markov_csv import CsvCell, markov_csv_text
from binary_entropy.stimulus_search_results import SearchResult, StimulusCandidate

_CANDIDATE_COLUMNS = (
    "stimulus_id",
    "selected",
    "group_id",
    "pair_id",
    "condition",
    "sequence",
    "symbol_mapping",
    "length",
    "a_count",
    "b_count",
    "a_proportion",
    "b_proportion",
    "switches",
    "switch_rate",
    "alternation_tendency",
    "longest_a_run",
    "longest_b_run",
    "longest_run",
    "predicted_symbol",
    "probability_a",
    "probability_b",
    "predictive_entropy_bits",
    "effective_context_depth",
    "context_used",
    "support_count",
    "surprisal_a_bits",
    "surprisal_b_bits",
    "rank_score",
    "target_symbol",
    "target_probability",
    "target_surprisal_bits",
    "target_congruency",
)


def stimulus_candidate_csv(result: SearchResult) -> str:
    """Export one summary row per accepted stimulus in rank order."""
    selected_ids = {candidate.stimulus_id for candidate in result.selected}
    return markov_csv_text(
        _CANDIDATE_COLUMNS,
        tuple(
            _candidate_row(candidate, selected=candidate.stimulus_id in selected_ids)
            for candidate in result.accepted
        ),
    )


def stimulus_scientific_csv(result: SearchResult) -> str:
    """Export every retained VMM evidence row."""
    columns = (
        "stimulus_id",
        "selected",
        "group_id",
        "pair_id",
        "condition",
        "symbol_mapping",
        "sequence",
        "requested_depth",
        "matched_context",
        "support",
        "count_a",
        "count_b",
        "probability_a",
        "probability_b",
        "predictive_entropy_bits",
        "status",
        "selected_depth",
        "context_used",
        "support_count",
        "predicted_symbol",
        "surprisal_a_bits",
        "surprisal_b_bits",
        "target_symbol",
        "target_probability",
        "target_surprisal_bits",
        "target_congruency",
        "result_scope",
        "minimum_support",
        "smoothing_alpha",
        "seed",
        "timestamp_utc",
        "package_version",
    )
    selected_ids = {candidate.stimulus_id for candidate in result.selected}
    rows = [
        (
            str(candidate.stimulus_id),
            candidate.stimulus_id in selected_ids,
            candidate.group_id,
            candidate.pair_id,
            candidate.condition,
            "|".join(candidate.symbol_mapping),
            _sequence(candidate.sequence),
            evidence.depth,
            _sequence(evidence.matched_suffix),
            evidence.support,
            evidence.count_a,
            evidence.count_b,
            evidence.probability_a,
            evidence.probability_b,
            evidence.predictive_entropy_bits,
            evidence.status.value,
            candidate.effective_context_depth,
            None
            if candidate.context_used is None
            else _sequence(candidate.context_used),
            candidate.support_count,
            _predicted_symbol(candidate),
            candidate.surprisal_a_bits,
            candidate.surprisal_b_bits,
            _target_symbol(candidate),
            candidate.target_probability,
            candidate.target_surprisal_bits,
            None
            if candidate.target_congruency is None
            else candidate.target_congruency.value,
            "per_sequence",
            result.config.vmm_config.minimum_support,
            result.config.vmm_config.smoothing.alpha,
            result.config.seed,
            result.timestamp_utc,
            result.package_version,
        )
        for candidate in result.accepted
        for evidence in candidate.depth_rows
    ]
    return markov_csv_text(columns, rows)


def stimulus_experiment_csv(candidates: Sequence[StimulusCandidate]) -> str:
    """Export experiment-ready assigned stimuli."""
    return markov_csv_text(
        _CANDIDATE_COLUMNS,
        tuple(_candidate_row(candidate, selected=True) for candidate in candidates),
    )


def _candidate_row(
    candidate: StimulusCandidate, *, selected: bool
) -> tuple[CsvCell, ...]:
    metrics = candidate.metrics
    return (
        str(candidate.stimulus_id),
        selected,
        candidate.group_id,
        candidate.pair_id,
        candidate.condition,
        _sequence(candidate.sequence),
        "|".join(candidate.symbol_mapping),
        metrics.length,
        metrics.a_count,
        metrics.b_count,
        metrics.a_proportion,
        metrics.b_proportion,
        metrics.switches,
        metrics.switch_rate,
        metrics.alternation_tendency,
        metrics.longest_a_run,
        metrics.longest_b_run,
        metrics.longest_run,
        _predicted_symbol(candidate),
        candidate.probability_a,
        candidate.probability_b,
        candidate.predictive_entropy_bits,
        candidate.effective_context_depth,
        None if candidate.context_used is None else _sequence(candidate.context_used),
        candidate.support_count,
        candidate.surprisal_a_bits,
        candidate.surprisal_b_bits,
        candidate.rank_score,
        _target_symbol(candidate),
        candidate.target_probability,
        candidate.target_surprisal_bits,
        None
        if candidate.target_congruency is None
        else candidate.target_congruency.value,
    )


def _sequence(sequence: tuple[int, ...]) -> str:
    return "".join("A" if symbol == 0 else "B" for symbol in sequence)


def _predicted_symbol(candidate: StimulusCandidate) -> str | None:
    return (
        None
        if candidate.probability_a is None
        else "tie"
        if candidate.predicted_target_index is None
        else "A"
        if candidate.predicted_target_index == 0
        else "B"
    )


def _target_symbol(candidate: StimulusCandidate) -> str | None:
    return (
        None
        if candidate.target_index is None
        else "A"
        if candidate.target_index == 0
        else "B"
    )
