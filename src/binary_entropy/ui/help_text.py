"""Centralized scientific terminology registry for contextual UI help.

Every user-facing technical control, metric, table column, and section heading
sources its short hover explanation from ``UI_HELP`` here so terminology stays
consistent across the Analyzer and Stimulus Search and scientific corrections
happen in one place. Definitions describe how *this* program computes each
quantity; they never restate a generic textbook formula divorced from the code.

Domains: entropy, prediction, markov, sequence_structure, stimulus_generation,
matching, target, export. Keys are flat so callers use ``UI_HELP["key"]``.
"""

from typing import Final

UI_HELP: Final[dict[str, str]] = {
    # ── entropy ──────────────────────────────────────────────────────────
    "predictive_entropy": (
        "Uncertainty of the model's next-symbol prediction, as the binary "
        "Shannon entropy of the predicted A/B distribution. 0 bits means one "
        "outcome is essentially certain; 1 bit means maximum uncertainty "
        "(about 50/50). Range 0-1 bits."
    ),
    "observed_entropy": (
        "Shannon entropy of the symbols already present in the sequence, from "
        "the observed A/B frequencies. It describes symbol composition, not a "
        "prediction. Higher means a more even A/B mix. Range 0-1 bits."
    ),
    "surprisal": (
        "Information carried by one particular symbol, computed as "
        "-log2(P(symbol)) under the model's predicted distribution. Rarer "
        "outcomes have higher surprisal; a probability of 0 gives infinite "
        "surprisal. Range 0+ bits."
    ),
    "target_surprisal": (
        "Surprisal of the supplied observed target, -log2(P(target)), scored "
        "only after the final prediction. It never influences fitting or "
        "context selection. Range 0+ bits."
    ),
    # ── prediction ───────────────────────────────────────────────────────
    "prediction_strength": (
        "The probability assigned to the model's most likely next symbol, "
        "max(P(A), P(B)). Values near 0.5 indicate a weak, near-even "
        "prediction; values near 1 indicate a strong one. Range 0.5-1.0."
    ),
    "prediction": (
        "The model's most probable next symbol for this sequence. Shown as "
        "'Tie' when both symbols are equally likely and 'Unavailable' when no "
        "supported prediction exists."
    ),
    "probability_a": (
        "Model probability that the next symbol is A given the selected "
        "context. Range 0-1."
    ),
    "probability_b": (
        "Model probability that the next symbol is B given the selected "
        "context. Range 0-1."
    ),
    # ── markov / VMM ─────────────────────────────────────────────────────
    "context_depth": (
        "The candidate context length being considered: how many immediately "
        "preceding symbols a context could use. This is the depth under "
        "examination, not necessarily the depth actually used."
    ),
    "effective_depth": (
        "The number of immediately preceding symbols the VMM actually used for "
        "this prediction after support and backoff logic. A larger depth means "
        "a longer context was relied on; it does not by itself mean the "
        "prediction is stronger."
    ),
    "context_support": (
        "How many within-record occurrences of the selected context were "
        "observed in the training data. Low support means the probability "
        "estimate rests on little evidence."
    ),
    "minimum_support": (
        "The smallest context occurrence count a suffix must reach to be used. "
        "The VMM uses the deepest suffix meeting this threshold; shorter "
        "supported suffixes are used through backoff otherwise. Higher values "
        "demand more evidence before a longer context is trusted."
    ),
    "backoff": (
        "When the deepest suffix lacks the minimum support (or is unseen), the "
        "model falls back to the next shorter supported suffix. Backoff is "
        "always shown, never silent, and records are never concatenated."
    ),
    "kt_smoothing": (
        "Krichevsky-Trofimov smoothing adds a fixed 0.5 pseudo-count to each "
        "outcome, P(x|c) = (N(c,x) + 0.5) / (N(c) + 1), so unseen or rare "
        "continuations do not receive extreme probabilities. This is the "
        "default estimator."
    ),
    "mle_smoothing": (
        "Maximum likelihood estimation uses raw counts with no smoothing "
        "(alpha = 0). An unseen context has no occurrences and is reported as "
        "unavailable rather than given an implicit probability."
    ),
    "additive_smoothing": (
        "Custom additive smoothing adds a positive alpha pseudo-count to each "
        "outcome, P(x|c) = (N(c,x) + alpha) / (N(c) + 2*alpha). Larger alpha "
        "pulls estimates toward 0.5; alpha = 0.5 reproduces KT smoothing."
    ),
    "vmm_smoothing": (
        "How next-symbol probabilities are estimated from context counts: KT "
        "(default, adds 0.5), MLE (raw counts, no smoothing), or a custom "
        "positive additive alpha."
    ),
    "markov_workflow": (
        "Variable-order Markov weighs progressively longer preceding contexts "
        "and uses the deepest supported one. First-order Markov is a baseline "
        "that conditions only on the single immediately preceding symbol."
    ),
    "markov_estimation": (
        "How first-order transition probabilities are estimated: maximum "
        "likelihood (alpha = 0), Laplace/add-one (alpha = 1), or a custom "
        "nonnegative additive alpha."
    ),
    "markov_prefix_mode": (
        "Fixed uses one transition matrix fitted to the whole scope and only "
        "the final observed symbol selects its row. Re-estimate refits from "
        "each prefix; this updates estimates with evidence but adds no "
        "higher-order memory."
    ),
    "result_scope": (
        "Pooled sums within-record counts across all sequences and analyzes "
        "each against the shared fit. Per-sequence fits each record "
        "separately. Record boundaries are always preserved."
    ),
    # ── HMM ──────────────────────────────────────────────────────────────
    "hmm_configured": (
        "The HMM is a fixed two-hidden-state, two-observable model you "
        "configure; it is not trained from the entered sequences. Each record "
        "is filtered independently under the same submitted parameters."
    ),
    "hmm_posterior": (
        "Filtered probability of each hidden state after consuming the entered "
        "prefix. At depth 0 no observation has been consumed, so the posterior "
        "is unavailable."
    ),
    "hmm_next_hidden": (
        "Predicted hidden-state distribution for the next step, obtained by "
        "advancing the posterior through the transition matrix."
    ),
    # ── sequence structure ───────────────────────────────────────────────
    "a_count": "Number of A symbols in the sequence.",
    "a_proportion": (
        "Fraction of the sequence that is A, count(A) / length. Range 0-1."
    ),
    "switches": (
        "Number of positions where the symbol differs from the one before it, "
        "i.e. how many times the sequence changes between A and B."
    ),
    "switch_rate": (
        "Switches divided by the number of adjacent pairs (length - 1): how "
        "often the sequence alternates between A and B. 0 means no changes; "
        "values near 1 mean it alternates almost every step. Range 0-1."
    ),
    "longest_run": (
        "Length of the longest unbroken block of the same symbol anywhere in "
        "the sequence."
    ),
    "longest_a_run": "Length of the longest unbroken block of A symbols.",
    "longest_b_run": "Length of the longest unbroken block of B symbols.",
    "run_structure": (
        "Summary of consecutive same-symbol blocks (runs). In matching it is "
        "compared through the complement-aligned longest-run difference; see "
        "the run-structure tolerance for the exact rule."
    ),
    # ── stimulus search generation ───────────────────────────────────────
    "sequence_length": (
        "Length of every generated candidate sequence, in symbols. At most 32."
    ),
    "desired_count": (
        "How many selected stimuli the search aims to accept. The result is "
        "'complete' when this many pass, otherwise 'partial'. No greater than "
        "the candidate limit."
    ),
    "candidate_limit": (
        "Maximum number of sequences sampled without replacement from the "
        "binary universe before ranking. The whole sample is analyzed; the "
        "search does not stop early. At most 5000."
    ),
    "search_seed": (
        "Integer seed for the reproducible sampler (0 through 2**63 - 1). The "
        "same seed and configuration reproduce the same sample."
    ),
    "predicted_symbol_constraint": (
        "Keep only candidates whose modal next-symbol prediction matches this "
        "choice. 'Either' accepts A, B, and ties."
    ),
    "hard_constraints": (
        "Filters that reject any candidate outside their range. Hard "
        "constraints are never relaxed automatically, even if the desired "
        "count cannot be reached."
    ),
    "soft_preferences": (
        "Ranking preferences, not filters. Accepted candidates are ordered by "
        "weighted absolute distance from each preference target; larger "
        "weights pull that metric closer to its target."
    ),
    "preference_target": "The metric value this preference ranks candidates toward.",
    "preference_weight": (
        "Relative importance of this preference in the weighted-distance rank "
        "score. Larger weights make matching this target matter more."
    ),
    "rank_score": (
        "Weighted absolute-distance score used to order accepted candidates; "
        "lower ranks first. Ties break by stimulus ID."
    ),
    "acceptance_rate": (
        "Fraction of the evaluated sample that passed every hard constraint."
    ),
    "partial_reason": (
        "Why fewer than the desired count were selected: 'candidate_limit' "
        "(the bounded sample ran out) or 'universe_exhausted' (every possible "
        "sequence was tried). Partial is a valid bounded result, not an error."
    ),
    # ── matching tolerances (maximum allowed difference) ─────────────────
    "matching_tolerances_section": (
        "How similar two stimuli must be to form a matched pair. Each "
        "tolerance is a maximum allowed difference; smaller tolerances enforce "
        "stricter matching but can make valid pairs harder to find. Matching "
        "pairs opposite predictions after exchanging A/B identities and "
        "requires equal sequence lengths."
    ),
    "tolerance_entropy": (
        "Maximum allowed difference in predictive entropy between two matched "
        "stimuli. Smaller values require more closely matched next-symbol "
        "uncertainty."
    ),
    "tolerance_prediction_strength": (
        "Maximum allowed difference between the prediction strengths "
        "(max(P(A), P(B))) of two matched stimuli. Smaller values produce more "
        "closely matched predictive confidence."
    ),
    "tolerance_a_proportion": (
        "Maximum allowed difference between one stimulus's A proportion and the "
        "other's B proportion (compared complement-aligned, since matching "
        "exchanges A/B identities). Smaller values enforce more similar symbol "
        "composition between the pair."
    ),
    "tolerance_switch_rate": (
        "Maximum allowed difference in switch rate between two matched "
        "sequences: how similarly often they alternate between A and B."
    ),
    "tolerance_run_structure": (
        "Maximum allowed run-structure difference, computed complement-aligned "
        "as max(|firstA_run - secondB_run|, |firstB_run - secondA_run|) over "
        "the longest A/B runs. Smaller values require more similar run "
        "structure after exchanging A/B identities."
    ),
    "tolerance_effective_depth": (
        "Maximum allowed difference between the effective context depths used "
        "to predict the two sequences. 0 requires exactly matching effective "
        "depths."
    ),
    "total_distance": (
        "Sum of the individual metric differences for a matched pair; smaller "
        "means a closer match. Only pairs within every tolerance are formed."
    ),
    # ── target assignment / QC ───────────────────────────────────────────
    "assignment_seed": (
        "Seed for the separate, reproducible target-assignment step. Targets "
        "never influence generation, fitting, search, ranking, or matching."
    ),
    "target_assignment": (
        "Policy for labeling each stimulus with an intended next target. "
        "Expected/unexpected balance congruent vs. incongruent predictions and "
        "leave ties unassigned; always-A/always-B label every stimulus."
    ),
    "target_congruency": (
        "Whether the assigned target agrees with the model's prediction: "
        "expected (modal), unexpected (lower-probability), tie, or unassigned."
    ),
    "descriptive_qc_section": (
        "Descriptive summary of the final set: prediction, tie, unavailable, "
        "target, and metric counts plus imbalance flags. It is quality "
        "control, not held-out, causal, or inferential validation."
    ),
    "imbalance_flags": (
        "Named descriptive warnings when predicted symbols or assigned targets "
        "are unevenly distributed across the final set."
    ),
    # ── input / intake ───────────────────────────────────────────────────
    "observable_labels": (
        "The two symbol names shared by every method. Labels are trimmed, "
        "nonempty, and distinct; spaces are allowed inside a label."
    ),
    "input_mode": (
        "How sequences are supplied: a single pasted sequence, a pasted batch "
        "(one sequence per line), a TXT upload, or a CSV upload with mapped "
        "columns. Record boundaries are never concatenated."
    ),
    "actual_target": (
        "Optional observed next symbol used only to score surprisal against "
        "the already-computed prediction. It never trains, fits, or changes "
        "any result (in-sample, not held out)."
    ),
    # ── export ───────────────────────────────────────────────────────────
    "export_precision": (
        "Downloads preserve raw float64 values (at least 12 decimal places); "
        "the on-screen three-decimal rounding never reaches exports."
    ),
    "candidate_csv": (
        "Every accepted candidate in rank order with selected status, metrics, "
        "VMM result, and any pair or target fields."
    ),
    "scientific_csv": (
        "Every retained VMM context-depth evidence row for accepted candidates."
    ),
    "reproducibility_json": (
        "The immutable search configuration, counts, status, violations, and "
        "any supplied matching tolerances, assignment seed, and validation "
        "report needed to reproduce the search."
    ),
    "experiment_csv": (
        "The explicit final candidate set after selection, matching, or target "
        "assignment, ready for presentation."
    ),
}
