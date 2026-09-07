# Binary Context Entropy Workbench Design Contract

Status: Binding implementation contract

Product surface: A dark, compact, responsive Streamlit workbench for comparing binary sequence entropy methods across one sequence or a batch.

## 1. Authority, Scope, and Product Character

This contract governs visual, interaction, accessibility, and scientific presentation decisions. If requirements conflict, use this order:

1. Mathematical correctness.
2. Transparent assumptions.
3. Reproducibility.
4. Compact usability.
5. Visual design.

The workbench serves researchers who need to configure only the analysis that applies, submit one sequence or a batch, compare selected methods, inspect exact values, and export raw scientific results. It has the restrained, operational information structure of an established research tool. It does not copy any brand, component library, logo, or product copy.

The interface must not imply that a descriptive statistic is a next target prediction, that a fitted model is available when it cannot be estimated, or that an evaluation target influenced the calculation. Visual polish supports inspection. It never obscures assumptions, validation, units, precision, or unavailable states.

### 1.1 Included workbench capabilities

1. Markov Chain is selected by default, with Variable-order Markov as its initial workflow and First-order Markov available as a baseline.
2. Users may select Hidden Markov Model and Observed Shannon Entropy alongside Markov Chain. Multiple methods run and appear together.
3. Method specific controls appear only for selected methods.
4. One shared intake accepts pasted single sequences, pasted batches, TXT uploads, and CSV uploads.
5. Results distinguish pooled and per sequence calculations while presenting accepted records through persistent One, Multiple, and All sequence scopes; compare selected methods; and expose each included result method through a tracked horizontal tab.
6. Results include visual summaries, exact value tables, charts, warnings, reproducibility details, and raw exports.
7. An optional evaluation target is assessed only against an already computed final prediction. It is not an input to fitting, selection, pooling, or prediction.

### 1.2 Explicit exclusions

This contract does not add simulation settings, unlocked HMM probability rows, marketing content, decorative media, or custom interaction that duplicates a usable native Streamlit control. Markov controls are limited to the workflows, estimation choices, and evidence states defined in Section 2.1.1. Suffix backoff is an automatic visible outcome, not a control.

### 1.3 Voice and content

1. Use sentence case, direct scientific language, and visible labels.
2. Name a quantity before notation. For example, `Markov predictive entropy, H(X_next | context)`.
3. Put units in metric labels, table headers, chart axes, tooltips, and raw export headers where applicable.
4. State assumptions, sample limits, and validation failures plainly. Never use promotional language or claims of certainty beyond the method.
5. Use no emojis, slogans, anthropomorphic language, decorative symbols, or icon only scientific actions.
6. Render user supplied labels and sequence identifiers as text, never trusted markup.

## 2. Scientific Semantics and Precision

### 2.1 Method contract

| Method | Selection and controls | Result meaning |
| --- | --- | --- |
| Markov Chain | Selected by default. Its nested, non-top-level workflows are Variable-order Markov and First-order Markov. Variable-order Markov is the initial workflow; First-order Markov remains available as a baseline. Show only the controls and estimation status defined in Section 2.1.1. | A next-symbol distribution, entropy, and optional target evaluation conditional on the selected observed context rule and available estimate. |
| Hidden Markov Model | Optional. Reveal the two-state HMM labels, initial distribution, transition matrix, emission matrix, and preset controls only when selected. | A next observable prediction and entropy conditional on the configured HMM parameters and consumed prefix. |
| Observed Shannon Entropy | Optional. It needs no model editor. | Observed symbol composition entropy. It is descriptive, not a next target prediction. |

The selection control is a labeled native multiselect or equivalent checkable control with exactly three top-level checkable models: `Markov Chain`, `Hidden Markov Model`, and `Observed Shannon Entropy`. Markov Chain is present on initial load. Selecting Hidden Markov Model and Observed Shannon Entropy at the same time is conforming. Deselecting a method removes its method controls and marks only that method’s old result as unavailable or stale. It must not alter or discard another method’s valid result.

### 2.1.1 Markov workflows, fitting, and evidence

Markov Chain exposes exactly these nested workflows, not additional top-level methods:

1. **Variable-order Markov.** This is the initial predictive workflow. It evaluates progressively longer preceding contexts from order 0 through the available suffix depth, reports evidence and availability at each depth, and uses the deepest context that meets the configured minimum support. It does not assume that a longer context improves prediction, lowers entropy, raises surprisal, or produces a monotonic trend.
2. **First-order Markov.** This baseline uses only the immediately preceding symbol.

The training dataset is the only dataset that fits context counts, selects an available context, determines support or sparsity, estimates probabilities, or establishes a backoff result. Each submitted record is independent: contexts and transitions never cross a record boundary. An optional target supplied with a training record is assessed only after final prediction and every affected table, chart, notice, and export labels the result `In-sample evaluation, not held out`.

For each eligible context, KT smoothing is the default and fixes `alpha = 0.5`; MLE fixes `alpha = 0` and uses `P(next = x | context) = count(context, x) / occurrence_count(context)`; custom additive smoothing requires a positive alpha and uses `P(next = x | context) = (count(context, x) + alpha) / (occurrence_count(context) + 2 alpha)`. The estimator and alpha are visible in controls, results, reproducibility details, and exports. An unseen context with MLE has the exact unavailable text `MLE unavailable: unseen context has no occurrences in the training dataset.` It never receives an implicit 0.5 probability or hidden smoothing.

Each context evidence row exposes its occurrence count, next-symbol A count, next-symbol B count, support status, and sparse status. Support and sparse criteria are named with their configured threshold or rule, rather than inferred from styling. Variable-order Markov backs off from an unavailable or insufficiently supported suffix to the next shorter supported suffix. Each affected result shows the effective depth, context used, and evidence status at every examined depth. Backoff is never silent, and no arbitrary order cap is imposed beyond the available preceding symbols.

Markov predictive entropy is the binary entropy of the fitted next-symbol distribution for the displayed context. Target surprisal is `-log2(P(observed target | displayed context))` and is available only for a supplied target in a predictive result. A zero predicted target probability is shown as an explicit infinite-surprisal condition, never as an ordinary finite value. These conditional values are not generalized stationary-distribution or entropy-rate claims.

### 2.2 HMM probability editor

The HMM editor remains fixed at two hidden states and two observed symbols. For every binary probability row, expose one editable probability and its derived read only complement. Label the relationship, show the resulting row sum of `1.000`, and keep user configured row and column labels visible.

1. The derived field is not focusable as an editable input.
2. Editing the source value updates only its visible derived complement.
3. The interface must not provide an unlock row option, a second independently editable value, silent normalization, clamping, redistribution, or inferred replacement value.
4. Invalid source values retain the entered text, receive a specific error, and cannot produce a valid HMM result.

### 2.3 MLE availability and warnings

When a selected method depends on a maximum likelihood estimate, the result must state whether the estimate is available for the submitted data. Insufficient observations, missing required transitions or contexts, malformed records, or another documented estimation precondition failure produce an explicit `MLE unavailable` state.

An unavailable MLE state names the method, workflow, requested context depth where applicable, scope, reason, and recovery condition. It never substitutes a default estimate, a zero filled table, an implicit 0.5 distribution, an infinite value styled as ordinary output, hidden smoothing, silent backoff, or an unrelated method’s result. Other selected methods may still render valid results. Warnings remain visible near the affected result and in reproducibility details.

### 2.4 Input, batch, and target semantics

One Data intake primitive accepts all supported training-data paths without changing the scientific contract:

1. A single sequence entered as text.
2. A batch pasted as documented records.
3. A TXT upload containing the same documented text record syntax.
4. A CSV upload containing the documented sequence identifier and sequence fields, with an optional evaluation target field.

There is no separate evaluation-data intake. A target supplied with a training record is an in-sample target assessment and is labeled `In-sample evaluation, not held out`; it is never described as held out.

Visible help defines delimiters, record boundaries, CSV column names, accepted symbol labels, and whether positions begin at zero or one. The parser reports the sequence identifier, token position, and reason for every invalid record without silently dropping, repairing, or reordering data.

Batch results always distinguish:

1. Per sequence results, one deterministic row or retrievable selector projection for each accepted sequence identifier.
2. Pooled results, only when the method’s pooling rule is mathematically defined and explicitly named.
3. Excluded or invalid records, with reasons and no contribution to a pooled result.

An evaluation target is optional and separate from the observed sequence. It may appear per sequence when supplied by the documented input format. It is evaluated only after the selected method has produced that sequence’s final prediction. It must not train, fit, choose, score, alter, or validate a method. It is an in-sample assessment, not held-out evaluation. When a selected method has no predictive distribution, such as Observed Shannon Entropy, the target assessment is shown as `Not applicable`, with a short reason.

### 2.5 Numeric presentation and exports

The visual precision policy is fixed at exactly three decimal places for finite decimal scientific values. This applies to metrics, matrices, row sums, table cells, chart labels, chart tooltips, and warnings that display a numeric value. Use a leading zero for values between negative one and one. Use tabular numerals and right aligned numeric table cells where Streamlit permits.

Visual formatting does not alter the source value. Streamlit data columns should use a three decimal display format, such as `%.3f`, while retaining source values for sort, search, charts, and export. Never show normal formatted output for `NaN`, positive infinity, or negative infinity. State the condition and preserve it only where raw scientific export requires a documented representation.

Machine readable CSV, TXT where applicable, and JSON exports preserve raw values at a minimum of 12 decimal places, or the exact available calculation representation when more precision is retained. Export headers name units, method, sequence scope, and precision. Display formatting and export serialization must use separate paths so three decimal UI rounding cannot reach raw export values.

## 3. Design System Tokens

All custom visual values resolve to these tokens. Add a token here before implementation uses a new color, spacing value, radius, type treatment, border, or chart setting.

### 3.1 Dark high contrast color tokens

The workbench is dark only. Deep navy layers establish the serious scientific hierarchy of the reference. `--color-accent` is the single cyan interactive and primary data accent. Green is reserved for valid, ready, and successful status. Semantic warning, error, and success colors communicate status only and are always paired with text, a label, or an icon from a supported SVG set when an icon is necessary.

| Token | Value | Use |
| --- | --- | --- |
| `--color-canvas` | `#07111f` | Browser and workspace background |
| `--color-sidebar` | `#0a1628` | Persistent native sidebar background |
| `--color-surface` | `#0d1b2a` | Main panels, native control wells, chart plot |
| `--color-surface-raised` | `#122338` | Nested panels and table header layers |
| `--color-surface-strong` | `#183047` | Selected neutral state, read only derived value |
| `--color-text-primary` | `#f3f7fc` | Headings, labels, values, body text |
| `--color-text-secondary` | `#b8c7d9` | Helper text, metadata, chart labels |
| `--color-text-muted` | `#8294aa` | Nonessential provenance only |
| `--color-border-subtle` | `#263b52` | Panel, table, and quiet divider boundaries |
| `--color-border-strong` | `#48617b` | Input and emphasized boundaries |
| `--color-accent` | `#22d3ee` | Primary action, links, focus, selection, primary chart series |
| `--color-accent-hover` | `#67e8f9` | Interactive hover state |
| `--color-accent-active` | `#06b6d4` | Interactive pressed state |
| `--color-on-accent` | `#04202a` | Text on the filled accent action |
| `--color-focus` | `#a5f3fc` | Focus outline |
| `--color-error` | `#ff9b98` | Error text and rule |
| `--color-error-surface` | `#3b2029` | Error notice background |
| `--color-warning` | `#f6c85f` | Warning text and rule |
| `--color-warning-surface` | `#372f1b` | Warning notice background |
| `--color-success` | `#4ade80` | Valid, ready, and successful status text and rule |
| `--color-success-surface` | `#123524` | Success notice background |

Rendered contrast must meet WCAG 2.2 AA. Normal text requires at least 4.5:1 contrast. Essential graphical objects, focus indicators, borders that communicate state, and large text require at least 3:1. Muted text is never the only presentation of essential content.

### 3.2 Typography tokens

| Token | Value | Use |
| --- | --- | --- |
| `--font-ui` | `"IBM Plex Sans", "Source Sans 3", "Noto Sans", system-ui, sans-serif` | Interface copy and headings |
| `--font-data` | `"IBM Plex Sans", "Source Sans 3", system-ui, sans-serif` | Numeric controls, metrics, tables, and chart text |
| `--font-mono` | `"Cascadia Mono", "JetBrains Mono", "Liberation Mono", ui-monospace, monospace` | Sequence values, compact notation, raw preview |
| `--font-weight-regular` | `400` | Body and data cells |
| `--font-weight-semibold` | `600` | Labels, headings, headers, emphasized values |
| `--text-caption` | `0.75rem / 1rem` | Provenance and compact metadata |
| `--text-small` | `0.875rem / 1.25rem` | Helper text, table cells, axis labels |
| `--text-body` | `1rem / 1.5rem` | Controls and body text |
| `--text-heading-3` | `1.125rem / 1.5rem` | Panel and method headings |
| `--text-heading-2` | `1.375rem / 1.75rem` | Major sections |
| `--text-heading-1` | `1.875rem / 2.25rem` | Compact workspace title |

Apply `font-variant-numeric: tabular-nums lining-nums` to every numeric surface, including input values, tables, charts, tooltips, exported-value preview, and row sums. There is no display type tier.

### 3.3 Spacing, geometry, and layout tokens

| Token | Value | Use |
| --- | --- | --- |
| `--space-1` | `0.25rem` | Tight inline separation |
| `--space-2` | `0.5rem` | Label to control and compact cell separation |
| `--space-3` | `0.75rem` | Standard cell padding |
| `--space-4` | `1rem` | Panel interior and compact group gaps |
| `--space-5` | `1.5rem` | Primitive separation |
| `--space-6` | `2rem` | Major section separation |
| `--radius-control` | `0.25rem` | Native control and button boundary |
| `--radius-panel` | `0.5rem` | Main panel boundary |
| `--radius-inset` | `0.25rem` | Nested panel and notice boundary |
| `--border-width` | `1px` | Standard boundary |
| `--border-width-emphasis` | `2px` | Focus and status rule |
| `--focus-width` | `3px` | Focus outline |
| `--focus-offset` | `2px` | Focus outline separation |
| `--control-min-block` | `2.75rem` | Compact minimum interactive height of 44 CSS px |
| `--motion-none` | `0ms` | Custom transition and animation duration |
| `--layout-min-supported` | `23.4375rem` | 375 px minimum target width |
| `--breakpoint-standard` | `48rem` | 768 px QA width |
| `--breakpoint-desktop` | `80rem` | 1280 px desktop QA width |
| `--sidebar-width` | `20rem` | Approximate expanded native sidebar width at desktop |
| `--workspace-max` | `96rem` | Maximum main workspace width after the sidebar |
| `--gutter-narrow` | `1rem` | 375 px gutter |
| `--gutter-standard` | `1.5rem` | 768 px gutter |
| `--gutter-desktop` | `2rem` | 1280 px gutter |
| `--table-row-compact` | `2.25rem` | Minimum desktop and tablet data row height |
| `--table-row-touch` | `2.75rem` | Minimum 375 px data row height |
| `--limit-multiple-sequences` | `12` | Maximum records rendered as individual selections in Multiple scope |
| `--limit-comparison-rows` | `25` | Maximum visible rows in a comparison table before explicit truncation |
| `--limit-evidence-rows` | `50` | Maximum visible rows in an evidence table before explicit truncation |

Panels use dark tonal layering, a visible border, and rounded geometry. The page never uses gradients, glass, backdrop blur, texture, decorative shadows, imagery, or visual effects that resemble a marketing surface.

### 3.4 Chart tokens

| Token | Value | Use |
| --- | --- | --- |
| `--chart-series-primary` | `var(--color-accent)` | Selected method or primary comparison series |
| `--chart-series-secondary` | `#b8c7d9` | Secondary selected method series, paired with a distinct marker or line pattern |
| `--chart-series-tertiary` | `#8294aa` | Third selected method series, paired with a distinct marker or line pattern |
| `--chart-grid` | `#263b52` | Major grid lines |
| `--chart-axis` | `var(--color-text-secondary)` | Ticks, titles, and baseline |
| `--chart-plot` | `var(--color-surface)` | Plot region |
| `--chart-height-narrow` | `18rem` | 375 px chart height |
| `--chart-height-standard` | `20rem` | 768 px chart height |
| `--chart-height-desktop` | `22rem` | 1280 px chart height |
| `--chart-line-width` | `2px` | Series line |
| `--chart-marker-size` | `0.5rem` | Series marker diameter |

Use a maximum of the three defined comparison series at once. Distinguish methods with named legend entries, markers, and line patterns, never color alone. The active Overview contains at most one primary chart. It is the dominant visualization in the workspace, and its exact value table remains authoritative.

## 4. Page Architecture and Responsive Rules

### 4.1 Persistent sidebar and workspace source order

The application is one selector-driven research workbench, not a setup gate followed by a separate results document. Configuration remains available before and after calculation. Use native `st.sidebar` for the persistent desktop rail and its native drawer behavior on narrow screens. Use native Streamlit controls wherever they meet the semantic requirement; narrowly scoped CSS may map tokens and hierarchy but must not depend on generated class names, undocumented DOM depth, or unsupported sticky behavior.

The semantic and keyboard source order is:

1. Sidebar identity and concise purpose.
2. Sidebar Analysis methods section, active method controls, Training data intake, optional evaluation target, validation, and `Calculate selected methods`.
3. Sidebar calculation status and complete export sections.
4. Main workspace title, current calculation summary, and scientific notices.
5. Tracked horizontal tabs in the fixed order Method Comparison, Markov Chain, Hidden Markov Model, Observed Shannon Entropy.
6. Persistent sequence scope and sequence selector.
7. Overview, Compare, and Evidence subview control.
8. The active selector projection, selected-scope exports, reproducibility details, and supporting help.

Method Comparison is available whenever the current calculation has at least one valid result. Each method-specific tab remains in the fixed order and preserves its disabled, uncalculated, unavailable, or not applicable explanation as appropriate. Methods are not rendered as vertically stacked full sections. Switching a tracked tab changes only the visible projection of stored results.

### 4.2 Selector-driven result architecture

The tracked tab row is the primary horizontal navigation within results. The active tab is tracked through the documented Streamlit 1.61.1 tab `.open` state so the application renders Method Comparison or the active method’s deep content rather than expanding every tab body. Method Comparison is enabled whenever the immutable current calculation contains at least one valid method result. Tab labels remain stable and do not carry scientific values that become ambiguous when truncated.

The sequence selector persists directly beneath the tracked tabs and exposes exactly three scopes:

1. **One.** Select one accepted sequence identifier and render a single deep dive for that record.
2. **Multiple.** Select up to `--limit-multiple-sequences` accepted identifiers in deterministic source order and render a bounded comparison, never a repeated expanded result section for every record.
3. **All.** Include every accepted record in aggregate summaries and bounded tables. At 5,000 records this remains an aggregate dashboard, not a record-by-record loop.

The subview control persists beneath the sequence selector and exposes exactly `Overview`, `Compare`, and `Evidence`. Overview is initial and carries the current scope’s summary plus at most one primary chart. Compare provides bounded, exact cross-method or cross-sequence rows with comparability warnings. Evidence provides bounded method-specific evidence and assumptions. Changing tracked tab, sequence scope, selected sequence identifiers, or subview is presentation-only: it reads the immutable current calculation, does not recalculate, does not alter target assessment, and does not mark results stale.

Visible comparison and evidence tables stop at `--limit-comparison-rows` and `--limit-evidence-rows`. Each truncated table states `Showing N of M rows`, the deterministic ordering rule, and how to obtain the complete export. Truncation is presentation-only. Complete exports include every eligible row, including rows not visible in the active projection.

### 4.3 Desktop, tablet, and mobile composition

| Viewport | Composition | Required behavior |
| --- | --- | --- |
| 375 px | Native sidebar is available as a drawer; the workspace uses one column with `--gutter-narrow`. | The drawer does not cover focused content after dismissal. Tabs and selector controls scroll horizontally only through native documented behavior or wrap without clipping. Cards and metrics stack. Long IDs wrap or truncate with an accessible full value. Tables use accessible horizontal overflow only when semantic columns cannot reflow. |
| 768 px | Native sidebar uses its narrow-screen drawer behavior; the workspace uses `--gutter-standard`. | Compact related controls may share a row only when labels, values, and focus remain readable at 200 percent zoom. The active workspace stays one coherent flow; no configuration/results column split is introduced. |
| 1280 px | Expanded native sidebar is approximately `--sidebar-width`; the remaining workspace is bounded by `--workspace-max` with `--gutter-desktop`. | Sidebar sections remain compact and persistent while the workspace gives the dominant chart and exact table the available width. Dense horizontal metric and control groups may be used without changing semantic order. |

At intermediate widths, prefer reflow over compressed labels, clipped tabs, or narrow scientific tables. The layout never creates a 40/60 configuration/results main-column split.

### 4.4 Scroll ownership

The browser document owns vertical scrolling. The sidebar follows native Streamlit scrolling and drawer behavior; the workspace, cards, tables, charts, expanders, tab bodies, and result projections must not add nested vertical scroll regions. Data tables may use horizontal overflow only for semantic columns that cannot reflow without losing meaning. An overflow wrapper needs a visible instruction, a keyboard-reachable focus target, and a clear focus indicator. Long identifiers and unbroken strings must wrap with `overflow-wrap: anywhere` or use an accessible visible truncation treatment with the full text available, and must never widen the page.

## 5. Reusable Primitives and States

### 5.1 Shared state language

| State | Contract |
| --- | --- |
| Default | Persistent label, standard border, clear surface, and helper text where needed. |
| Selected | Accent boundary or native selected treatment plus visible text or check state. |
| Hover | Only interactive elements change. Use the accent hover or stronger neutral boundary. No lift, glow, or motion. |
| Focus | Use `--focus-width` solid `--color-focus` with `--focus-offset`. Never remove focus or replace it with color fill alone. |
| Disabled | Keep the value legible and state the unavailable reason nearby. Disabled styling never hides essential information. |
| Error | Show a specific message and programmatic association, with an error boundary or rule. Color is supplementary. |
| Warning | State the scientific risk, affected method or sequence scope, and required interpretation. |
| Loading | Show static descriptive text, such as `Calculating selected methods...`. No spinner, shimmer, pulse, or animated placeholder. |
| Empty | Explain the required input or action. Do not render an empty chart frame or blank table. |
| Uncalculated | Name the method and state that no current calculation exists; provide the calculate path without placeholder values. |
| Stale | Replace affected results and exports with `Recalculation required`. Never present prior values as current. |
| Unavailable | Name the unavailable method, scope, reason, and recovery condition. Do not render invented numeric output. |
| Truncated | State `Showing N of M rows`, preserve deterministic order, and identify the complete export. |
| Valid | Pair the current calculated status with visible text and `--color-success`; never rely on green alone. |

### 5.2 Native sidebar section

**Use for:** persistent configuration, current calculation status, and complete exports.

**Anatomy:** visible section heading, compact helper text only where necessary, native controls in task order, subtle divider, and local status or validation text. The Analysis methods, Method controls, Training data, Calculation status, and Complete exports sections remain discoverable in the same order. The sidebar is not a setup screen and does not disappear after calculation.

**States:** collapsed sidebar sections retain visible native labels. A section with an error identifies its affected method or record without turning the entire sidebar into an error state. On narrow screens the native drawer restores focus logically when opened and closed. Sidebar controls persist across presentation-only selector changes.

### 5.3 Analysis method selection and method control panel

**Anatomy:** visible Analysis methods label, concise explanation, native multiselect or checkable controls, selected method summary, and controls for selected methods only. Markov Chain is selected initially with Variable-order Markov active. Markov Chain shows its workflow selector beneath the method heading: Variable-order Markov and First-order Markov. Variable-order Markov shows explicit smoothing, minimum-support, and result-scope controls. Its automatic deepest-supported suffix-backoff policy and outcome are visible in results and evidence. First-order Markov retains its estimator, prefix mode, and result-scope controls. HMM uses the fixed complement editor from Section 2.2. Observed Shannon Entropy has no model controls.

**States:** each selected method has a visible check state. A hidden method control panel has no keyboard stop. A selected method with missing or invalid controls remains selected but cannot produce a valid result. Deselecting an analysis method removes only its controls and marks only that method’s current calculation unavailable or stale. Changes to analysis method inclusion or a computational control require explicit recalculation; they are distinct from switching the tracked result method tab. Markov unavailable states identify the workflow, requested context depth when applicable, evidence condition, smoothing alpha, and automatic backoff outcome when a shorter suffix is selected. Inputs are never silently normalized.

### 5.4 Tracked horizontal method tabs

**Use for:** switching among Method Comparison and method-specific projections from one immutable current calculation.

**Anatomy:** native horizontal tabs with stable text labels, visible selected treatment, native focus behavior, and one open tab body. The fixed order is Method Comparison, Markov Chain, Hidden Markov Model, Observed Shannon Entropy. All four labels retain that order.

**States:** opening a tab persists the active tab and performs no calculation. Method Comparison is available whenever the current calculation has at least one valid result; otherwise it explains that comparison requires a valid current result. Each method-specific tab remains present and displays its current result, `Not calculated`, disabled state, or specific unavailable reason. The Hidden Markov Model tab uses `HMM disabled for the current calculation` when HMM was not included. Disabled or uncalculated HMM states contain no empty chart, zero-filled matrix, or invented prediction.

### 5.5 Persistent sequence scope and multiselect

**Use for:** One, Multiple, and All presentation scopes after accepted records exist.

**Anatomy:** visible Sequence scope label, native choice among One, Multiple, and All, record count, and a persistent selector. One uses one sequence selector. Multiple uses a native multiselect capped at `--limit-multiple-sequences`. All states the total included count and does not render an identifier control that implies records are excluded. IDs remain in deterministic source order.

**States:** the selected scope and still-valid identifiers persist across tracked-tab and subview changes. When a recalculation removes an identifier, the selector removes only invalid selections, explains the change, and chooses the first accepted identifier only when One would otherwise have no valid selection. Multiple with no selected IDs is an explicit empty state. All with no accepted records explains that calculation needs valid training data. Scope changes are presentation-only and never alter pooled fitting or the method result scope configured before calculation.

### 5.6 Overview, Compare, and Evidence subview control

**Anatomy:** persistent visible label and native segmented or equivalent single-choice control in the fixed order Overview, Compare, Evidence. Overview is selected initially. Its label, selected state, and focus state remain visible without relying on position or color alone.

**States:** changing subview persists the choice for the active calculation and performs no recalculation. If a subview has no applicable content, show why and identify another available subview; do not silently redirect. Compare states that predictive and descriptive quantities are not interchangeable. Evidence never hides a critical warning solely behind a subview switch.

### 5.7 Single deep-dive card

**Use for:** One scope in the active method tab.

**Anatomy:** selected sequence identifier, method and workflow, scope, compact status line, a restrained metric row, one dominant chart when applicable, exact values, method assumptions, and target evaluation. The card is a low-radius bordered tonal region, not a floating decorative card. Sequence values use `--font-mono`; numerical values use `--font-data` and exactly three decimals.

**States:** `Not calculated`, `Recalculation required`, `MLE unavailable`, `Not applicable`, invalid record, and valid replace the metric value as appropriate. A deep dive does not create an empty chart frame. Long sequence identifiers wrap without changing workspace width.

### 5.8 Aggregate dashboard and bounded multiple view

**Use for:** All scope aggregate summaries and Multiple scope selected-record summaries.

**Anatomy:** scope definition, accepted and excluded counts, compact metric blocks, comparability notice, at most one primary Overview chart, and one bounded exact summary table. Pooled results appear only when the method’s pooling rule is mathematically defined and explicitly named. Multiple scope summarizes no more than `--limit-multiple-sequences` selected records. All scope summarizes every accepted record without expanding one card or section per record.

**States:** deterministic source order is stated. If a method is unavailable, retain its named comparison status with the reason rather than closing the gap. A stale computational input removes current numeric projections. Empty and partial-availability states state the affected method and record counts.

### 5.9 Bounded comparison table and bounded evidence table

**Use for:** per-sequence summaries, pooled summaries, cross-method comparison, Markov context-depth evidence, evaluation rows, prefix rows, and exact chart values.

**Anatomy:** visible heading, scope and method caption, semantic headers with units, native Streamlit table or dataframe where practical, exactly three decimal display formatting, deterministic order statement, row-count disclosure, and the matching complete or selected export action. Comparison tables render at most `--limit-comparison-rows`; evidence tables render at most `--limit-evidence-rows`. Neither creates a record-by-record expanded loop.

A Markov context-depth table includes dataset role, record identifier, workflow, requested depth, actual depth, context, context occurrence count, next A count, next B count, support status, sparse status, estimation rule, smoothing alpha, automatic suffix-backoff outcome and reason when applicable, next A probability, next B probability, predictive entropy in bits, evaluation status, observed target when supplied, and target surprisal in bits when defined. The default row order is deterministic input order, then requested depth from shortest to longest, or prefix order. Sorting and search may be offered through native dataframe behavior, but they are presentation only. They never change calculation order, chart order, target evaluation, export order, or stored results.

**States:** loading, empty, stale, error, unavailable, and not applicable states replace the table with a textual state or scientific notice. A truncated table states `Showing N of M rows`; truncation is never described as data exclusion. Horizontal overflow is permitted. Vertical internal scrolling is not. The authoritative underlying values remain available through raw export at the required precision.

### 5.10 Dominant chart region

**Use for:** the active Overview’s selected method trend across context depth, sequence, or documented batch scope.

**Anatomy:** heading, one sentence explaining scope and quantity, one responsive chart, named legend, visible axes with units, exact hover values formatted to three decimals, and an in-document bounded exact table fallback. Markov Variable-order context analysis includes a context-depth entropy chart with requested depth on the horizontal axis and predictive entropy in bits on the vertical axis. The binary-entropy vertical axis is fixed from 0 to 1 bits. When probability is the selected documented quantity instead, the chart may plot next A and next B probability by requested depth with a visible 0 to 1 probability axis. The active Overview never shows both charts. Neither view implies that successive depths are monotonic, comparable when contexts differ, or evidence that longer contexts are better.

**States and rules:** use markers and named line patterns so selected methods do not rely on color alone. Plotly animation and animated redraw are disabled. No empty axes render for loading, empty, stale, unavailable, error, or not applicable states. A depth with unavailable MLE or unavailable requested context appears in the exact table and is not invented as a plotted value. Binary entropy axes use a fixed 0 to 1 bits range only when the plotted quantity is binary entropy. Do not force that range on another documented value type.

### 5.11 Labeled scientific field, data intake, and scientific notice

**Labeled field anatomy:** persistent label, optional unit or notation, native control, helper text, and specific validation message. Use for identifiers, labels, numeric values, sequence text, evaluation targets, and imports. Numeric values use `--font-data`. Sequence and raw input values use `--font-mono`. Retain entered values on validation failure. Move focus to the first invalid field after submission when Streamlit provides a stable accessible mechanism. Read-only derived HMM complements use `--color-surface-strong`, include the source relationship in the label, and never masquerade as editable controls.

**Data intake anatomy:** a sidebar section headed Training data, input-mode guidance, text area or native uploader, documented accepted formats, parsed count, accepted and rejected record summary, and local errors. It serves single input, batch input, TXT upload, CSV upload, parsed-record review, and file validation. An optional target belongs to its training record and is visibly labeled as in-sample assessment. Before submission, documented examples are not treated as data. A valid batch shows accepted sequence identifiers in deterministic source order. Invalid records remain visible with their identifier or line and error. Upload success confirms parsing only, not calculation. Changing text, file, or parsed records makes all dependent results stale.

**Scientific notice anatomy:** explicit text label such as `Warning`, `MLE unavailable`, `Error`, or `Recalculation required`; concise title; body; and optional action link. Use a tonal surface, `--radius-inset`, and an emphasized leading rule. Notices persist until resolved or deliberately dismissed. A notice is not focusable unless it contains an interactive element, and no decorative icon is required.

### 5.12 Complete exports, selected exports, and reproducibility

**Complete exports:** the persistent sidebar groups every currently valid, supported method-specific artifact for the complete calculated dataset. A complete export is never limited by active tab, sequence selector, subview, or visible row bounds. It preserves the existing artifact names, schemas, precision, deterministic ordering, and scientific scope. Observed Shannon Entropy remains without a download export unless its scientific export contract is separately specified.

**Selected exports:** the workspace offers secondary native download actions only for existing supported artifacts that can be scoped to the active method and selected One, Multiple, or All records without changing their schema. Labels name the artifact, format, method, and selected sequence scope. Selected exports include every eligible row for that selected scope even when the visible table is truncated. They do not invent a new aggregate, omit hidden evidence, or change precision.

Experimental Markov downloads are separately named `Context model export`, `Context evidence export`, and `Evaluation export`. Each carries an experimental-status notice and contains the required stimulus fields: dataset role, training dataset identifier, record identifier, source order, sequence stimulus, consumed-prefix stimulus, sequence length, consumed-prefix depth, displayed context, requested depth, actual depth, workflow, estimation rule, smoothing alpha, automatic suffix-backoff outcome and reason, context occurrence count, next A count, next B count, support status, sparse status, next A probability, next B probability, predictive entropy in bits, observed target when supplied, target probability, target surprisal in bits when defined, and evaluation status. The Context model export additionally records the `deepest_supported_suffix` configured depth selection and all fitted context distributions. The Context evidence export records every examined training context. The Evaluation export records in-sample status and never describes a training-data target as held out.

The reproducibility panel exposes selected methods, method assumptions, configured HMM values when used, Markov workflow, requested and actual depths, estimation rule, smoothing alpha, support and sparse rules, automatic suffix-backoff outcomes and reasons, availability, training dataset role, parsed record counts, accepted and rejected identifiers, sequence lengths, pooled rule, in-sample target assessment status, units, visible precision, raw export precision, stable ordering, and application or calculation version when recorded. Never claim a seed, version, dependency fact, held-out evaluation, stationary result, or entropy-rate interpretation that is not recorded.

**States:** exports remain disabled with a visible reason until valid current content exists. An uncalculated or disabled HMM has no enabled HMM result export; a valid HMM preset export follows its existing independent readiness contract. A stale computational change disables every dependent export. Export failure retains the action and shows a specific notice. File names are descriptive and deterministic, not timestamp dependent. The calculation action remains the sole filled primary action.

### 5.13 Help expander

**Use for:** input syntax, method definitions, pooling rules, target evaluation semantics, and precision policy.

**States and rules:** critical validation, MLE availability, and stale state information never live only in an expander or inactive subview. Native triggers have visible names, keyboard support, and focus treatment where Streamlit permits. Expanded content adds no nested vertical scroll and no decorative open or close motion.

## 6. Lifecycle, Validation, and Motion

### 6.1 Explicit calculation lifecycle

1. Initial state: Markov Chain with Variable-order Markov is selected. Intake and method controls show a documented starter state or clear empty instruction. Results explain what must be submitted.
2. Editing state: changing analysis method inclusion, Markov workflow, smoothing alpha, minimum support, training input text, upload, parsed records, computational result-scope inputs, or evaluation target invalidates only dependent outputs and exports.
3. Validation state: the explicit calculate action validates every active input. It preserves values, identifies the first invalid field, and summarizes additional failures without flooding the page.
4. Calculating state: show persistent static text naming the selected methods. Do not calculate on each keystroke.
5. Success state: store the valid result snapshot as immutable session state, render its selector projection, include applicable warnings, and enable only matching exports.
6. Partial availability state: render valid selected method results while clearly presenting unavailable methods and their reasons.
7. Failure state: preserve input and show reproducible diagnostics. Do not leave partial output styled as valid.

### 6.2 Presentation state and persistence

Tracked-tab `.open` state, sequence scope, selected sequence identifiers, and Overview/Compare/Evidence subview are presentation state. They persist across ordinary Streamlit reruns while still valid for the immutable current calculation. Changing any of them re-projects stored results only: it must not invoke analysis, change fitted values, alter target assessment, invalidate results, or disable exports. The active selector projection is derived from the immutable result snapshot rather than copied into a second mutable scientific result.

Computational inputs and presentation selectors use separate session-state keys and separate change paths. A new successful calculation replaces the immutable snapshot atomically, then reconciles presentation selectors against its included methods and accepted IDs. A failed calculation leaves the last result clearly stale or unavailable rather than presenting it as current.

### 6.3 Validation rules

1. Validate only controls required by selected methods and their active workflow.
2. Probability inputs accept only the documented finite range. The HMM complement remains derived, not independently validated as a user value.
3. Labels and identifiers must meet the documented distinctness and syntax rules.
4. Parse text, TXT, and CSV through one documented record model. Do not silently coerce labels, discard tokens, reorder records, or normalize probabilities.
5. Variable-order Markov derives available depths from each independent record and never crosses record boundaries.
6. A target supplied with a training record is excluded from every fit, count, support decision, smoothing denominator, context selection, and backoff decision. Its assessment is visibly labeled `In-sample evaluation, not held out`.
7. State clearly when pooled analysis is impossible, unavailable, or excluded by the active method’s rule.
8. Keep method assumptions and warnings visible near the result they qualify.

### 6.4 Extensible projection architecture

The presentation projection accepts explicit method, sequence-scope, selected-ID, and subview state. Future filtering may extend that typed selector state and the same deterministic projection boundary without changing stored calculations, scientific schemas, or export precision. This release adds no stimulus selector, filter control, hidden filter behavior, or filtered scientific claim. Reserved architecture is not visible UI and does not alter current results.

### 6.5 Action hierarchy

1. The sole filled primary action is `Calculate selected methods`.
2. Secondary actions include file upload, model preset import where HMM is selected, and downloads.
3. Tertiary actions include help and in document navigation.
4. No floating action, destructive styling for non destructive work, hidden context menu, or icon only scientific action is allowed.

### 6.6 Motion policy

There is no decorative motion. `--motion-none` applies to all custom transitions and animations. Do not add entrance effects, hover lift, parallax, pulsing status, animated gradients, skeleton shimmer, chart tweening, or decorative micro interactions. Native motion that cannot be removed must be nonessential. Respect `prefers-reduced-motion` without removing content or feedback.

## 7. Accessibility and Cognitive Usability

The release target is WCAG 2.2 AA at 375, 768, and 1280 px, at 200 percent zoom, and under text spacing overrides. Accessibility failure is not accepted debt.

### 7.1 Inclusive user contexts

1. **Keyboard-only analyst.** Must configure, calculate, open and close the sidebar drawer, switch method tabs, choose One/Multiple/All, select identifiers, change subviews, reach bounded table overflow, and download available artifacts without a pointer or keyboard trap.
2. **Low-vision researcher at 200 percent zoom.** Must read labels, units, values, status, focus, chart alternatives, and complete identifiers without overlapping content, clipped actions, or two-dimensional page scrolling.
3. **Narrow-screen field researcher at 375 px.** Must use the native sidebar drawer and the active workspace without losing context, encountering covered focus, or traversing vertically stacked results for inactive methods.
4. **Large-batch researcher with 5,000 sequences.** Must reach aggregate status, bounded comparison and evidence, specific selected records, and complete exports without 5,000 expanded cards, an unbounded control list, or nested vertical scrolling.

### 7.2 Interaction, reflow, and alternatives

1. Persistent labels, units, required state, descriptions, and validation messages are programmatically associated with controls where Streamlit supports it.
2. Focus order follows document order: sidebar identity, analysis method selection, active controls, intake, target, calculation action, status, complete downloads, workspace title, method tabs, sequence scope and selector, subview, active result interactions, selected downloads, and help.
3. Focus remains visible against every token surface. No result panel or CSS behavior may obscure the focused item.
4. Every control, native sidebar drawer, method tab, scope choice, multiselect, and subview is keyboard operable without a custom keyboard trap. Do not override native numeric input navigation or tab key behavior.
5. Tables have semantic headers and captions or equivalent descriptions. The default deterministic order is stated before sortable or searchable presentation.
6. Charts have an accessible name, scope summary, named series, visible units, and an exact table fallback. No information requires hover, color perception, or a pointer.
7. At 320 CSS px equivalent, text reflows without two dimensional page scrolling. Only documented data regions may scroll horizontally.
8. Do not reduce opacity on essential content. Use text, structure, and visible state alongside semantic color.
9. Keep task instructions procedural and short. Keep method definitions separate from data entry while placing validation rules near the field that needs them.
10. Do not auto dismiss scientific warnings, reset values after errors, impose time limits, or make stale results easy to mistake for current results.
11. At 200 percent zoom, sidebar and workspace controls reflow rather than overlap; the open drawer, focused element, active method, sequence scope, and subview remain perceivable.
12. Long identifiers, labels, sequences, file names, and unbroken strings wrap or use accessible truncation with the complete text available. They never clip silently, overlap actions, or widen the page.
13. Horizontal tab or table overflow has a visible cue and keyboard-reachable operation where Streamlit supports it. No component introduces nested vertical scrolling.

## 8. Implementation Governance, QA, and Release

### 8.1 Streamlit constraints

1. Target Streamlit 1.61.1 and prefer documented theme settings, native `st.sidebar`, native controls, uploaders, tabs, and dataframes before custom markup.
2. A native dataframe may support sorting and search. Its source values, default order, and raw export must remain deterministic.
3. CSS is scoped to stable application containers and token mapping. Do not rely on generated class names, undocumented DOM depth, or unsupported sticky APIs.
4. Use tracked horizontal tabs and their documented `.open` state to identify the active method projection. Do not emulate tabs with custom components or infer state from unsupported DOM inspection.
5. Use the native sidebar drawer on narrow screens. Do not represent `st.bottom`, a main-column panel, or custom sticky CSS as the sidebar.
6. Plotly tokens and no-animation configuration do not by themselves make a chart accessible. The exact table fallback remains required.
7. Any verified framework limitation belongs in Accepted debt before release if it cannot be fixed without violating a higher priority.

### 8.2 Primitive showcase and QA conditions

Before release, a development-only primitive showcase must demonstrate default, selected, focus, disabled, error, warning, loading, empty, stale, unavailable, uncalculated, truncated, and valid states for every applicable primitive. It must include native sidebar sections, tracked method tabs, One/Multiple/All sequence selection, the subview control, single deep-dive cards, aggregate dashboard, bounded comparison, bounded evidence, complete exports, selected exports, and HMM disabled or uncalculated empty states.

QA must capture and inspect screenshots at 375, 768, and 1280 px. The 1280 px capture must show the persistent native sidebar at approximately `--sidebar-width`, compact horizontal method tabs and selectors, the workspace title at `--text-heading-1`, dense low-radius organization, and one dominant Overview chart. The 768 and 375 px captures must show the native sidebar drawer, coherent single workspace flow, no clipped method or sequence selection, usable upload and export controls, long-ID handling, and only permitted horizontal table overflow. All three widths must also pass at 200 percent zoom.

QA must also prove:

1. Markov Chain appears selected initially with Variable-order Markov active; Hidden Markov Model and Observed Shannon Entropy can be selected together; only selected analysis method controls appear; tracked tabs remain in the fixed order Method Comparison, Markov Chain, Hidden Markov Model, Observed Shannon Entropy rather than vertically stacked full sections; Method Comparison is available whenever the current calculation has at least one valid result; and method-specific tabs preserve their disabled or uncalculated states.
2. The HMM editor permits one editable value and shows a read only derived complement, with no unlock row control.
3. Single text, batch text, TXT upload, and CSV upload all show consistent parsing, accepted records, rejected records, and stale state after change.
4. Pooled and per-sequence computational scope, One/Multiple/All presentation scope, method comparison, Overview/Compare/Evidence, tables, charts, warnings, MLE unavailable state, and evaluation-only target semantics are distinguishable.
5. Dataframe display, metric blocks, chart labels, and tooltips show exactly three decimals while the corresponding raw export preserves at least 12 decimals or exact retained representation.
6. Keyboard only operation reaches and visibly focuses method selection, upload controls, dataframes or overflow wrappers, downloads, and expanders.
7. Workspace document scrolling remains the only vertical scroll owner outside native sidebar behavior. No tab body, result card, dataframe, comparison, or evidence region adds nested vertical scrolling.
8. Variable-order Markov examines progressively longer suffix contexts, uses the deepest sufficiently supported context, and identifies the effective predictive context depth. First-order Markov uses only the immediately preceding symbol as a baseline.
9. KT, MLE, and custom additive smoothing choices and alpha values are visible. An unseen MLE context shows the exact unavailable text, no implicit 0.5 distribution or hidden smoothing. Automatic suffix backoff uses the deepest supported suffix, and requested depth, actual depth, and reason appear in the table and export.
10. Training records never cross boundaries. A target cannot alter fit, counts, support, smoothing, context selection, or backoff. Training-data target assessment is labeled `In-sample evaluation, not held out`.
11. Context-depth tables expose occurrences, A and B counts, support and sparse status, probabilities, entropy, and defined target surprisal. The entropy chart uses a fixed 0 to 1 bits axis; any probability chart is optional; neither implies monotonicity.
12. Experimental Context model, Context evidence, and Evaluation exports contain their required stimulus fields and retain at least 12 decimals or exact retained representation.
13. No gradient, glass effect, decorative shadow, imagery, emoji, marketing section, unsupported API claim, or decorative motion appears.
14. Switching tracked tab, sequence scope, selected identifiers, or subview preserves the control state, uses the immutable result snapshot, performs no recalculation, does not change target assessment, and does not mark current results stale.
15. One renders one deep dive; Multiple renders no more than 12 selected records as a bounded comparison; All remains aggregate at 5,000 records. No record-by-record expanded loop appears.
16. The active Overview contains at most one primary chart. Compare shows at most 25 visible rows and Evidence at most 50, each with `Showing N of M rows` when truncated and a route to a complete export.
17. Sidebar complete exports ignore presentation selectors and remain complete. Workspace selected exports include every eligible row for the selected method and records without changing existing schema, ordering, target semantics, or raw precision.
18. Keyboard-only use, native narrow sidebar drawer behavior, focus restoration, 200 percent zoom, 320 CSS px equivalent reflow, long IDs, unbroken strings, and accessible horizontal overflow pass without clipped content or a keyboard trap.

### 8.3 Release checklist

1. The priority order in Section 1 resolved every disputed visual choice.
2. The rendered theme is dark navy, high contrast, compact, low-radius, and uses one restrained cyan interactive accent with green reserved for valid or successful status.
3. Every custom visual value resolves to a token.
4. Every selected method has an explicit assumption, result meaning, validation path, warning path, and reproducibility record.
5. Markov Chain is the default selection with Variable-order Markov as the initial workflow and First-order Markov available as a baseline. Hidden Markov Model and Observed Shannon Entropy can run simultaneously with it.
6. No unselected method controls occupy the sidebar, and no obsolete setup gate or 40/60 configuration/results main-column layout remains.
7. Text, batch, TXT, and CSV inputs produce clear accepted, rejected, pooled, and per sequence states.
8. Predictive and descriptive methods are never presented as interchangeable. Evaluation targets are clearly evaluation only.
9. MLE unavailable results are explicit and never replaced by invented values. An unseen MLE context uses the stated unavailable text and never receives an implicit 0.5 probability, hidden smoothing, or silent backoff.
10. Markov Chain exposes exactly Variable-order Markov and First-order Markov as nested workflows. Variable-order examines progressively longer suffix contexts and uses the deepest sufficiently supported context; First-order uses only the immediately preceding symbol. No simulation or arbitrary order cap exists.
11. Training data alone fits Markov contexts. No context crosses records, and an optional target does not contribute to fit, counts, support, smoothing, selection, or backoff. In-sample target assessment is labeled not held out.
12. Context evidence tables report occurrence count, A and B counts, support and sparse status, requested and actual depth, automatic suffix-backoff outcome and reason when used, probabilities, entropy, and target surprisal when defined. Context-depth entropy charts use a fixed 0 to 1 bits range, probability charts are optional, and no monotonicity is implied.
13. Experimental Context model, Context evidence, and Evaluation exports preserve the required stimulus fields, float64 source values, and at least 12 decimal places or exact retained representation.
14. HMM complements are derived read-only values. No unlock row control exists.
15. The visual layer uses exactly three decimal places for finite decimal values. Raw exports retain at least 12 decimals or exact retained representation.
16. Exact tables remain available for all charted values, and raw exports remain separate from display formatting.
17. The workspace document owns vertical scroll outside native sidebar behavior. Horizontal table overflow is the only allowed internal result overflow and is accessible.
18. Screenshots and keyboard checks at 375, 768, and 1280 px pass, including 200 percent zoom and reduced motion.
19. WCAG 2.2 AA contrast, focus, reflow, table, chart, and status requirements pass in the rendered application.
20. The native sidebar and its narrow-screen drawer are used without custom sticky layout or unsupported Streamlit DOM claims.
21. Tracked tabs remain in the fixed order Method Comparison, Markov Chain, Hidden Markov Model, Observed Shannon Entropy; Method Comparison is available with at least one valid current result; method-specific tabs preserve disabled or uncalculated states; and tab, sequence, and Overview/Compare/Evidence changes re-project immutable stored results without recalculation.
22. One, Multiple, and All scopes remain bounded: one deep dive, at most 12 selected records, and aggregate all-record rendering. Comparison and evidence disclose 25-row and 50-row visible limits while exports remain complete.
23. The active Overview contains at most one dominant chart, and no inactive method body or record-by-record expanded loop is rendered as a vertical results document.
24. Complete sidebar exports and selected workspace exports preserve existing schemas, deterministic ordering, target semantics, and raw precision.
25. Keyboard-only, low-vision/200-percent-zoom, 375 px narrow-screen, and 5,000-sequence user contexts pass their stated tasks.

### 8.4 Accepted debt

No accepted debt is recorded initially.

Any accepted debt must name the affected token or primitive, user impact, accessibility impact, scientific impact, reason it cannot be resolved now, owner, exit criterion, and review date. Accessibility failures, mathematically incorrect output, hidden assumptions, stale scientific output, contradictory display and export precision, silent input normalization, unsupported API claims, unbounded record rendering, incomplete complete exports, and nested vertical scrolling are never acceptable debt.
