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

1. Markov Chain is selected by default, with First-order Markov as its initial workflow.
2. Users may select Hidden Markov Model and Observed Shannon Entropy alongside Markov Chain. Multiple methods run and appear together.
3. Method specific controls appear only for selected methods.
4. One shared intake accepts pasted single sequences, pasted batches, TXT uploads, and CSV uploads.
5. Results distinguish pooled and per sequence views, compare selected methods, and retain method specific sections.
6. Results include visual summaries, exact value tables, charts, warnings, reproducibility details, and raw exports.
7. An optional evaluation dataset or evaluation target is assessed only against an already computed final prediction. It is not an input to fitting, selection, pooling, or prediction.

### 1.2 Explicit exclusions

This contract does not add simulation settings, unlocked HMM probability rows, marketing content, decorative media, or custom interaction that duplicates a usable native Streamlit control. Markov controls are limited to the workflows, estimation choices, evidence states, and explicit backoff defined in Section 2.1.1.

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
| Markov Chain | Selected by default. Its nested, non-top-level workflows are First-order Markov, Higher-order Markov, and Variable-order context analysis. First-order Markov is the initial overall default. Variable-order context analysis is the recommended default when an advanced workflow is chosen. Show only the controls and estimation status defined in Section 2.1.1. | A next-symbol distribution, entropy, and optional target evaluation conditional on the selected observed context rule and available estimate. |
| Hidden Markov Model | Optional. Reveal the two-state HMM labels, initial distribution, transition matrix, emission matrix, and preset controls only when selected. | A next observable prediction and entropy conditional on the configured HMM parameters and consumed prefix. |
| Observed Shannon Entropy | Optional. It needs no model editor. | Observed symbol composition entropy. It is descriptive, not a next target prediction. |

The selection control is a labeled native multiselect or equivalent checkable control with exactly three top-level checkable models: `Markov Chain`, `Hidden Markov Model`, and `Observed Shannon Entropy`. Markov Chain is present on initial load. Selecting Hidden Markov Model and Observed Shannon Entropy at the same time is conforming. Deselecting a method removes its method controls and marks only that method’s old result as unavailable or stale. It must not alter or discard another method’s valid result.

### 2.1.1 Markov workflows, fitting, and evidence

Markov Chain exposes exactly these nested workflows, not additional top-level methods:

1. **First-order Markov.** The context is the immediately preceding symbol. This is the initial overall default.
2. **Higher-order Markov.** The context is the previous k symbols for a user-selected fixed positive integer k. The control labels k as context depth and never calls it whole-sequence unless the user has selected the full available sequence as the context.
3. **Variable-order context analysis.** This is the recommended default advanced workflow. It evaluates progressively longer preceding contexts up to the available prefix depth and reports the evidence and availability at each depth. It does not assume that a longer context improves prediction, lowers entropy, raises surprisal, or produces a monotonic trend.

The training dataset is the only dataset that fits context counts, selects an available context, determines support or sparsity, estimates probabilities, or establishes a backoff result. Each submitted record is independent: contexts and transitions never cross a record boundary. An optional separate evaluation dataset is held out from fitting and contains no contribution to counts, smoothing denominators, context selection, pooling, or backoff. When training records are also evaluated, every affected table, chart, notice, and export labels the result `In-sample evaluation, not held out`.

For each eligible context, MLE uses `P(next = x | context) = count(context, x) / occurrence_count(context)`. Additive smoothing uses the selected nonnegative alpha: `P(next = x | context) = (count(context, x) + alpha) / (occurrence_count(context) + 2 alpha)`. Alpha, including alpha zero for MLE, is visible in controls, results, reproducibility details, and exports. An unseen context with MLE has the exact unavailable text `MLE unavailable: unseen context has no occurrences in the training dataset.` It never receives an implicit 0.5 probability or hidden smoothing.

Each context evidence row exposes its occurrence count, next-symbol A count, next-symbol B count, support status, and sparse status. Support and sparse criteria are named with their configured threshold or rule, rather than inferred from styling. Optional explicit suffix backoff may be selected for Higher-order Markov or Variable-order context analysis. When used, each affected result shows requested depth, actual depth, and the reason the requested context was unavailable or insufficiently supported. Without that selected option, unavailable requested contexts remain unavailable. Backoff is never silent, and no arbitrary order cap is imposed beyond the available preceding symbols and any user-selected fixed k.

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

An optional Evaluation data intake uses the same record model and validation rules, but is visibly separate from the Training data intake. It may contain sequence identifiers, sequences, and documented targets. Evaluation records are assessed only against a model already fit from Training data. A target supplied with a training record is an in-sample target assessment and is labeled `In-sample evaluation, not held out`; it is never described as held out.

Visible help defines delimiters, record boundaries, CSV column names, accepted symbol labels, and whether positions begin at zero or one. The parser reports the sequence identifier, token position, and reason for every invalid record without silently dropping, repairing, or reordering data.

Batch results always distinguish:

1. Per sequence results, one deterministic row or section for each accepted sequence identifier.
2. Pooled results, only when the method’s pooling rule is mathematically defined and explicitly named.
3. Excluded or invalid records, with reasons and no contribution to a pooled result.

An evaluation target is optional and separate from the observed sequence. It may appear per sequence when supplied by the documented input format. It is evaluated only after the selected method has produced that sequence’s final prediction. It must not train, fit, choose, score, alter, or validate a method. Separate evaluation data never contributes to fit. When a selected method has no predictive distribution, such as Observed Shannon Entropy, the target assessment is shown as `Not applicable`, with a short reason.

### 2.5 Numeric presentation and exports

The visual precision policy is fixed at exactly three decimal places for finite decimal scientific values. This applies to metrics, matrices, row sums, table cells, chart labels, chart tooltips, and warnings that display a numeric value. Use a leading zero for values between negative one and one. Use tabular numerals and right aligned numeric table cells where Streamlit permits.

Visual formatting does not alter the source value. Streamlit data columns should use a three decimal display format, such as `%.3f`, while retaining source values for sort, search, charts, and export. Never show normal formatted output for `NaN`, positive infinity, or negative infinity. State the condition and preserve it only where raw scientific export requires a documented representation.

Machine readable CSV, TXT where applicable, and JSON exports preserve raw values at a minimum of 12 decimal places, or the exact available calculation representation when more precision is retained. Export headers name units, method, sequence scope, and precision. Display formatting and export serialization must use separate paths so three decimal UI rounding cannot reach raw export values.

## 3. Design System Tokens

All custom visual values resolve to these tokens. Add a token here before implementation uses a new color, spacing value, radius, type treatment, border, or chart setting.

### 3.1 Dark high contrast color tokens

The workbench is dark only. Neutral charcoal layers establish hierarchy. `--color-accent` is the single interactive and primary data accent. Semantic warning, error, and success colors communicate status only and are always paired with text, a label, or an icon from a supported SVG set when an icon is necessary.

| Token | Value | Use |
| --- | --- | --- |
| `--color-canvas` | `#101316` | Browser document background |
| `--color-surface` | `#171c21` | Main panels, native control wells, chart plot |
| `--color-surface-raised` | `#20272e` | Nested panels and table header layers |
| `--color-surface-strong` | `#2a333d` | Selected neutral state, read only derived value |
| `--color-text-primary` | `#f4f7fa` | Headings, labels, values, body text |
| `--color-text-secondary` | `#c1cbd4` | Helper text, metadata, chart labels |
| `--color-text-muted` | `#97a5b3` | Nonessential provenance only |
| `--color-border-subtle` | `#3a4652` | Panel, table, and quiet divider boundaries |
| `--color-border-strong` | `#6e7d8c` | Input and emphasized boundaries |
| `--color-accent` | `#4cc9f0` | Primary action, links, focus, selection, primary chart series |
| `--color-accent-hover` | `#7bdcf6` | Interactive hover state |
| `--color-accent-active` | `#23afd7` | Interactive pressed state |
| `--color-on-accent` | `#081216` | Text on the filled accent action |
| `--color-focus` | `#a5ecff` | Focus outline |
| `--color-error` | `#ff8f8a` | Error text and rule |
| `--color-error-surface` | `#432226` | Error notice background |
| `--color-warning` | `#ffd166` | Warning text and rule |
| `--color-warning-surface` | `#40361a` | Warning notice background |
| `--color-success` | `#7ee2ae` | Success text and rule |
| `--color-success-surface` | `#173a2b` | Success notice background |

Rendered contrast must meet WCAG 2.2 AA. Normal text requires at least 4.5:1 contrast. Essential graphical objects, focus indicators, borders that communicate state, and large text require at least 3:1. Muted text is never the only presentation of essential content.

### 3.2 Typography tokens

| Token | Value | Use |
| --- | --- | --- |
| `--font-ui` | `"Inter", "Source Sans 3", "Noto Sans", system-ui, sans-serif` | Interface copy and headings |
| `--font-data` | `"IBM Plex Sans", "Inter", "Source Sans 3", system-ui, sans-serif` | Numeric controls, metrics, tables, and chart text |
| `--font-mono` | `"Cascadia Mono", "JetBrains Mono", "Liberation Mono", ui-monospace, monospace` | Sequence values, compact notation, raw preview |
| `--font-weight-regular` | `400` | Body and data cells |
| `--font-weight-semibold` | `600` | Labels, headings, headers, emphasized values |
| `--text-caption` | `0.75rem / 1rem` | Provenance and compact metadata |
| `--text-small` | `0.875rem / 1.25rem` | Helper text, table cells, axis labels |
| `--text-body` | `1rem / 1.5rem` | Controls and body text |
| `--text-heading-3` | `1.125rem / 1.5rem` | Panel and method headings |
| `--text-heading-2` | `1.375rem / 1.75rem` | Major sections |
| `--text-heading-1` | `1.75rem / 2.25rem` | Compact page title |

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
| `--radius-control` | `0.375rem` | Native control and button boundary |
| `--radius-panel` | `0.75rem` | Main panel boundary |
| `--radius-inset` | `0.5rem` | Nested panel and notice boundary |
| `--border-width` | `1px` | Standard boundary |
| `--border-width-emphasis` | `2px` | Focus and status rule |
| `--focus-width` | `3px` | Focus outline |
| `--focus-offset` | `2px` | Focus outline separation |
| `--control-min-block` | `2.75rem` | Minimum practical interactive height |
| `--motion-none` | `0ms` | Custom transition and animation duration |
| `--layout-min-supported` | `23.4375rem` | 375 px minimum target width |
| `--breakpoint-standard` | `48rem` | 768 px QA width |
| `--breakpoint-desktop` | `80rem` | 1280 px desktop QA width |
| `--content-max` | `90rem` | Maximum workbench width |
| `--gutter-narrow` | `1rem` | 375 px gutter |
| `--gutter-standard` | `1.5rem` | 768 px gutter |
| `--gutter-desktop` | `2rem` | 1280 px gutter |
| `--config-ratio` | `4` | Desktop configuration column ratio |
| `--results-ratio` | `6` | Desktop results column ratio |
| `--table-row-compact` | `2.25rem` | Minimum desktop and tablet data row height |
| `--table-row-touch` | `2.75rem` | Minimum 375 px data row height |

Panels use dark tonal layering, a visible border, and rounded geometry. The page never uses gradients, glass, backdrop blur, texture, decorative shadows, imagery, or visual effects that resemble a marketing surface.

### 3.4 Chart tokens

| Token | Value | Use |
| --- | --- | --- |
| `--chart-series-primary` | `var(--color-accent)` | Selected method or primary comparison series |
| `--chart-series-secondary` | `#b7c4d1` | Secondary selected method series, paired with a distinct marker or line pattern |
| `--chart-series-tertiary` | `#8795a3` | Third selected method series, paired with a distinct marker or line pattern |
| `--chart-grid` | `#3a4652` | Major grid lines |
| `--chart-axis` | `var(--color-text-secondary)` | Ticks, titles, and baseline |
| `--chart-plot` | `var(--color-surface)` | Plot region |
| `--chart-height-narrow` | `18rem` | 375 px chart height |
| `--chart-height-standard` | `20rem` | 768 px chart height |
| `--chart-height-desktop` | `22rem` | 1280 px chart height |
| `--chart-line-width` | `2px` | Series line |
| `--chart-marker-size` | `0.5rem` | Series marker diameter |

Use a maximum of the three defined comparison series at once. Distinguish methods with named legend entries, markers, and line patterns, never color alone. The chart’s exact value table remains authoritative.

## 4. Page Architecture and Responsive Rules

### 4.1 Source order and page anatomy

The browser document is a single research document in this source order:

1. Header with title, purpose, active method summary, and optional version provenance.
2. Left configuration column containing Method selection, method specific controls, Data intake, evaluation target, validation, and the explicit calculate action.
3. Results column containing status, scope switch, comparison summary, method sections, exact tables, charts, exports, and reproducibility details.
4. Supporting help expanders for input syntax, definitions, method assumptions, and export precision.

After the comparison summary, render separate Markov, HMM, and Shannon sections for the selected methods in that order. An unselected method has no result section. A selected method that cannot produce results retains its named section with an unavailable or not applicable state and reason.

The desktop grid must not change keyboard or screen reader order. Native Streamlit controls are preferred wherever they meet the semantic requirement. Custom CSS may set tokens and hierarchy but must be narrowly scoped and avoid claims of control over undocumented Streamlit DOM.

### 4.2 Desktop and narrow composition

| Viewport | Composition | Required behavior |
| --- | --- | --- |
| 375 px | One document column with `--gutter-narrow`. | All controls and actions are full width. Method choices wrap without clipping. Panels stack. Metrics stack. Tables use horizontal overflow only when columns cannot remain readable. |
| 768 px | One document column with `--gutter-standard`. | The configuration and results sections remain stacked in source order. Compact rows may group closely related fields only when each control stays readable and keyboard reachable. |
| 1280 px | Centered area up to `--content-max`, with a roughly 40/60 configuration and results grid using `st.columns((4, 6))` or an equivalent documented ratio. | The left column contains configuration. The right column contains results, charts, exact tables, exports, and help related to results. Each panel preserves its token spacing and does not exceed the content maximum. |

Do not force a two column configuration at 768 px merely to preserve desktop composition. At intermediate widths, stack when a label, numeric control, method choice, or table heading would otherwise become unreadable.

### 4.3 Scroll ownership and cautious sticky behavior

The browser document owns vertical scrolling. Inputs, panels, tables, charts, expanders, and result sections must not create nested vertical scroll regions. Data tables may use horizontal overflow only for semantic columns that cannot reflow without losing meaning. An overflow wrapper needs a visible instruction, a keyboard reachable focus target, and a clear focus indicator.

Streamlit does not provide a supported sticky side column API. `st.bottom` is a bottom region, not a sticky results panel. A release may use narrowly scoped CSS to keep a desktop results summary visible only if browser QA proves all of the following: keyboard focus is never obscured, 200 percent zoom remains usable, long results remain reachable, narrow layouts stack normally, and no nested scroll owner is introduced. Without that proof, ordinary non sticky document flow is conforming. This contract does not claim that sticky behavior is implemented or verified.

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
| Stale | Replace affected results and exports with `Recalculation required`. Never present prior values as current. |
| Unavailable | Name the unavailable method, scope, reason, and recovery condition. Do not render invented numeric output. |

### 5.2 Method selection

**Anatomy:** visible group label, concise explanation, native multiselect or checkable controls, selected method summary, and method specific controls below it.

**States:** Markov Chain is selected initially with First-order Markov active. Each selected method has a visible check state. A method with invalid controls remains selected but cannot produce a valid result. Deselecting a method removes only its controls and result availability. Changing selection marks comparison, result sections, charts, exact tables, and exports stale until recalculation.

### 5.3 Labeled scientific field

**Use for:** identifiers, labels, numeric values, sequence text, evaluation targets, and import controls.

**Anatomy:** persistent label, optional unit or notation, native control, helper text, and specific validation message. Numeric values use `--font-data`. Sequence and raw input values use `--font-mono`.

**States:** retain entered values on validation failure. Move focus to the first invalid field after submission when Streamlit provides a stable accessible mechanism. Read only derived HMM complements use `--color-surface-strong`, include the source relationship in the label, and never masquerade as editable controls.

### 5.4 Data intake panel

**Use for:** single input, batch input, TXT upload, CSV upload, parsed record review, and file validation.

**Anatomy:** panel heading identifies either Training data or optional Evaluation data, input mode guidance, text area or native uploader, documented accepted formats, parsed count, accepted and rejected record summary, and local errors. The same validation language applies across all supported paths. Evaluation data is visually and programmatically distinct from Training data.

**States:** before submission, show documented examples without treating them as data. A valid batch shows accepted sequence identifiers in deterministic source order. Invalid records remain visible with their identifier or line and error. Upload success confirms parsing only, not calculation. Changing text, file, or parsed records makes all dependent results stale.

### 5.5 Method control panel

**Use for:** controls belonging to one selected method.

**Anatomy:** panel heading, one sentence about the method’s assumption, only the inputs that method needs, validation summary, and method specific warning area. Markov Chain shows its workflow selector beneath the method heading: First-order Markov, Higher-order Markov, and Variable-order context analysis. First-order Markov is selected on initial load; Variable-order context analysis is identified as the recommended advanced workflow. Higher-order Markov shows an explicit positive context-depth k control. Variable-order context analysis shows its examined depth range and optional explicit suffix-backoff control. HMM uses the fixed complement editor from Section 2.2. Observed Shannon Entropy has no model controls.

**States:** a hidden method panel has no keyboard stop. A selected method with missing or invalid required data has an error or unavailable result state, not a hidden failure. Markov unavailable states identify the workflow, requested context depth when applicable, evidence condition, smoothing alpha, and whether explicit backoff was selected. Inputs are never silently normalized.

### 5.6 Scientific notice

**Use for:** assumptions, parsing issues, scientific warnings, MLE availability, validation, stale results, errors, and export readiness.

**Anatomy:** explicit text label such as `Warning`, `MLE unavailable`, `Error`, or `Recalculation required`; concise title; body; and optional action link. Use a tonal surface, rounded inset geometry, and an emphasized leading rule. Do not require a decorative icon.

Notices persist until resolved or deliberately dismissed. A notice is not focusable unless it contains an interactive element.

### 5.7 Result scope and comparison panel

**Use for:** the switch between pooled and per sequence analysis, selected method comparison, and cross method warnings.

**Anatomy:** a visible scope control, scope definition, compact comparison metric grid, selected method status, units, and a warning when results are not comparable. Pooled scope appears only where the selected method’s pool rule is defined. Per sequence is available whenever at least one accepted sequence exists.

**States:** the default visible order follows submitted source order. A stale selection removes numeric values. If a method is unavailable, retain its named comparison slot with an unavailable reason rather than closing the comparison gap. The panel must state that predictive and descriptive quantities are not interchangeable.

### 5.8 Metric block

**Use for:** a small set of primary values in a method or comparison summary.

**Anatomy:** method name, metric label, three decimal value, unit, scope, and one short qualification. Metric blocks sit in a rounded tonal panel, not a floating decorative card.

**States:** `Not calculated`, `Recalculation required`, `MLE unavailable`, and `Not applicable` replace a numeric value when appropriate. A metric block is not interactive and receives no hover styling.

### 5.9 Exact result dataframe

**Use for:** per-sequence summaries, pooled summaries, method comparison, Markov context-depth evidence, evaluation rows, prefix rows, and exact chart values.

**Anatomy:** visible heading, scope and method caption, semantic headers with units, native Streamlit dataframe where practical, a three decimal display format, and raw export actions nearby. A Markov context-depth table includes dataset role, record identifier, workflow, requested depth, actual depth, context, context occurrence count, next A count, next B count, support status, sparse status, estimation rule, smoothing alpha, explicit backoff reason when applicable, next A probability, next B probability, predictive entropy in bits, evaluation status, observed target when supplied, and target surprisal in bits when defined. The default row order is deterministic input order, then requested depth from shortest to longest, or prefix order. Sorting and search may be offered through native dataframe behavior, but they are presentation only. They never change calculation order, chart order, target evaluation, or export order.

**States:** loading, empty, stale, error, unavailable, and not applicable states replace the dataframe with a textual state or scientific notice. Horizontal overflow is permitted. Vertical internal scrolling is not. The authoritative underlying values remain available through raw export at the required precision.

### 5.10 Chart region

**Use for:** method comparison and selected method trends across context depth, sequence, or documented batch scope.

**Anatomy:** heading, one sentence explaining scope and quantity, responsive chart, named legend, visible axes with units, exact hover values formatted to three decimals, and an in-document exact table fallback. Markov Variable-order context analysis includes a context-depth entropy chart with requested depth on the horizontal axis and predictive entropy in bits on the vertical axis. The binary-entropy vertical axis is fixed from 0 to 1 bits. An optional probability chart may plot next A and next B probability by requested depth with a visible 0 to 1 probability axis. Neither chart implies that successive depths are monotonic, comparable when contexts differ, or evidence that longer contexts are better.

**States and rules:** use markers and named line patterns so selected methods do not rely on color alone. Plotly animation and animated redraw are disabled. No empty axes render for loading, empty, stale, unavailable, error, or not applicable states. A depth with unavailable MLE or unavailable requested context appears in the exact table and is not invented as a plotted value. Binary entropy axes use a fixed 0 to 1 bits range only when the plotted quantity is binary entropy. Do not force that range on another documented value type.

### 5.11 Download action and reproducibility panel

**Use for:** raw result exports, model preset exports where applicable, experimental Markov exports, and the scientific record needed to reproduce an analysis.

**Anatomy:** a secondary native download action naming its exact artifact, format, method scope, readiness state, and a nearby reproducibility panel. The calculation action is the sole filled primary action.

Experimental Markov downloads are separately named `Context model export`, `Context evidence export`, and `Evaluation export`. Each carries an experimental-status notice and contains the required stimulus fields: dataset role, training dataset identifier, evaluation dataset identifier when present, record identifier, source order, sequence stimulus, consumed-prefix stimulus, sequence length, consumed-prefix depth, displayed context, requested depth, actual depth, workflow, estimation rule, smoothing alpha, suffix-backoff selection and reason, context occurrence count, next A count, next B count, support status, sparse status, next A probability, next B probability, predictive entropy in bits, observed target when supplied, target probability, target surprisal in bits when defined, and evaluation status. The Context model export additionally records configured depth selection and all fitted context distributions. The Context evidence export records every examined training context. The Evaluation export records held-out or in-sample status and never relabels training-data evaluation as held out.

The reproducibility panel exposes selected methods, method assumptions, configured HMM values when used, Markov workflow, requested and actual depths, estimation rule, smoothing alpha, support and sparse rules, suffix-backoff selection and reasons, availability, training and evaluation dataset roles, parsed record counts, accepted and rejected identifiers, sequence lengths, pooled rule, target evaluation status, units, visible precision, raw export precision, stable ordering, and application or calculation version when recorded. Never claim a seed, version, dependency fact, held-out evaluation, stationary result, or entropy-rate interpretation that is not recorded.

**States:** exports remain disabled with a reason until valid current content exists. A stale change disables every dependent export. Export failure retains the action and shows a specific notice. File names are descriptive and deterministic, not timestamp dependent.

### 5.12 Help expander

**Use for:** input syntax, method definitions, pooling rules, target evaluation semantics, and precision policy.

**States and rules:** critical validation, MLE availability, and stale state information never live only in an expander. Native triggers have visible names, keyboard support, and focus treatment where Streamlit permits. Expanded content adds no nested vertical scroll and no decorative open or close motion.

## 6. Lifecycle, Validation, and Motion

### 6.1 Explicit calculation lifecycle

1. Initial state: Markov Chain with First-order Markov is selected. Intake and method controls show a documented starter state or clear empty instruction. Results explain what must be submitted.
2. Editing state: changing methods, Markov workflow, context depth, smoothing alpha, suffix-backoff selection, training or evaluation input text, upload, parsed records, scope inputs, or evaluation target invalidates dependent outputs and exports.
3. Validation state: the explicit calculate action validates every active input. It preserves values, identifies the first invalid field, and summarizes additional failures without flooding the page.
4. Calculating state: show persistent static text naming the selected methods. Do not calculate on each keystroke.
5. Success state: render valid current results in document flow, include applicable warnings, and enable only matching exports.
6. Partial availability state: render valid selected method results while clearly presenting unavailable methods and their reasons.
7. Failure state: preserve input and show reproducible diagnostics. Do not leave partial output styled as valid.

### 6.2 Validation rules

1. Validate only controls required by selected methods and their active workflow.
2. Probability inputs accept only the documented finite range. The HMM complement remains derived, not independently validated as a user value.
3. Labels and identifiers must meet the documented distinctness and syntax rules.
4. Parse text, TXT, and CSV through one documented record model. Do not silently coerce labels, discard tokens, reorder records, or normalize probabilities.
5. Markov Chain validates a positive fixed k only for Higher-order Markov; Variable-order context analysis derives available depths from each independent record and never crosses record boundaries.
6. Separate evaluation data uses the documented record model but is excluded from every fit, count, support decision, smoothing denominator, context selection, and backoff decision. Training-data evaluation is visibly labeled not held out.
7. State clearly when pooled analysis is impossible, unavailable, or excluded by the active method’s rule.
8. Keep method assumptions and warnings visible near the result they qualify.

### 6.3 Action hierarchy

1. The sole filled primary action is `Calculate selected methods`.
2. Secondary actions include file upload, model preset import where HMM is selected, and downloads.
3. Tertiary actions include help and in document navigation.
4. No floating action, destructive styling for non destructive work, hidden context menu, or icon only scientific action is allowed.

### 6.4 Motion policy

There is no decorative motion. `--motion-none` applies to all custom transitions and animations. Do not add entrance effects, hover lift, parallax, pulsing status, animated gradients, skeleton shimmer, chart tweening, or decorative micro interactions. Native motion that cannot be removed must be nonessential. Respect `prefers-reduced-motion` without removing content or feedback.

## 7. Accessibility and Cognitive Usability

The release target is WCAG 2.2 AA at 375, 768, and 1280 px, at 200 percent zoom, and under text spacing overrides. Accessibility failure is not accepted debt.

1. Persistent labels, units, required state, descriptions, and validation messages are programmatically associated with controls where Streamlit supports it.
2. Focus order follows document order: method selection, active controls, intake, target, calculation action, result scope, result interactions, downloads, and help.
3. Focus remains visible against every token surface. No result panel or CSS behavior may obscure the focused item.
4. Every control is keyboard operable without a custom keyboard trap. Do not override native numeric input navigation.
5. Tables have semantic headers and captions or equivalent descriptions. The default deterministic order is stated before sortable or searchable presentation.
6. Charts have an accessible name, scope summary, named series, visible units, and an exact table fallback. No information requires hover, color perception, or a pointer.
7. At 320 CSS px equivalent, text reflows without two dimensional page scrolling. Only documented data regions may scroll horizontally.
8. Do not reduce opacity on essential content. Use text, structure, and visible state alongside semantic color.
9. Keep task instructions procedural and short. Keep method definitions separate from data entry while placing validation rules near the field that needs them.
10. Do not auto dismiss scientific warnings, reset values after errors, impose time limits, or make stale results easy to mistake for current results.

## 8. Implementation Governance, QA, and Release

### 8.1 Streamlit constraints

1. Prefer documented Streamlit theme settings, `st.columns` ratios, native controls, uploaders, and dataframes before custom markup.
2. A native dataframe may support sorting and search. Its source values, default order, and raw export must remain deterministic.
3. CSS is scoped to stable application containers and token mapping. Do not rely on generated class names, undocumented DOM depth, or unsupported sticky APIs.
4. `st.bottom` must not be represented as a sticky side results implementation.
5. Plotly tokens and no animation configuration do not by themselves make a chart accessible. The exact dataframe remains required.
6. Any verified framework limitation belongs in Accepted debt before release if it cannot be fixed without violating a higher priority.

### 8.2 Primitive showcase and QA conditions

Before release, a development only primitive showcase must demonstrate these states for every applicable primitive: default, selected, focus, disabled, error, warning, loading, empty, stale, unavailable, and valid.

QA must capture and inspect screenshots at 375, 768, and 1280 px. The 1280 px capture must show the roughly 40/60 desktop columns. The narrow captures must show the stacked source order, no clipped method selection, usable upload control, readable active method panels, and only permitted horizontal table overflow.

QA must also prove:

1. Markov Chain appears selected initially with First-order Markov active; Hidden Markov Model and Observed Shannon Entropy can be selected together; only selected method controls appear.
2. The HMM editor permits one editable value and shows a read only derived complement, with no unlock row control.
3. Single text, batch text, TXT upload, and CSV upload all show consistent parsing, accepted records, rejected records, and stale state after change.
4. Pooled and per-sequence result scope, method comparison, method sections, tables, charts, warnings, MLE unavailable state, and evaluation-only target semantics are distinguishable.
5. Dataframe display, metric blocks, chart labels, and tooltips show exactly three decimals while the corresponding raw export preserves at least 12 decimals or exact retained representation.
6. Keyboard only operation reaches and visibly focuses method selection, upload controls, dataframes or overflow wrappers, downloads, and expanders.
7. Document scrolling remains the only vertical scroll owner. If sticky CSS is proposed, separate browser QA proves every condition in Section 4.3. If not proven, release the non sticky layout.
8. First-order Markov uses the immediately preceding symbol. Higher-order Markov uses the previous k symbols with an explicit k. Variable-order context analysis examines progressively longer preceding contexts and identifies itself as the recommended advanced workflow.
9. MLE and additive smoothing alpha values are visible. An unseen MLE context shows the exact unavailable text, no implicit 0.5 distribution, hidden smoothing, or silent backoff. When explicit suffix backoff is selected, requested depth, actual depth, and reason appear in the table and export.
10. Training and separate evaluation records never cross boundaries. Evaluation data cannot alter fit, counts, support, smoothing, context selection, or backoff. Training-data evaluation is labeled `In-sample evaluation, not held out`.
11. Context-depth tables expose occurrences, A and B counts, support and sparse status, probabilities, entropy, and defined target surprisal. The entropy chart uses a fixed 0 to 1 bits axis; any probability chart is optional; neither implies monotonicity.
12. Experimental Context model, Context evidence, and Evaluation exports contain their required stimulus fields and retain at least 12 decimals or exact retained representation.
13. No gradient, glass effect, decorative shadow, imagery, emoji, marketing section, unsupported API claim, or decorative motion appears.

### 8.3 Release checklist

1. The priority order in Section 1 resolved every disputed visual choice.
2. The rendered theme is dark, high contrast, compact, rounded, and uses one restrained interactive accent.
3. Every custom visual value resolves to a token.
4. Every selected method has an explicit assumption, result meaning, validation path, warning path, and reproducibility record.
5. Markov Chain is the default selection with First-order Markov as the initial workflow. Hidden Markov Model and Observed Shannon Entropy can run simultaneously with it.
6. No unselected method controls occupy the configuration column.
7. Text, batch, TXT, and CSV inputs produce clear accepted, rejected, pooled, and per sequence states.
8. Predictive and descriptive methods are never presented as interchangeable. Evaluation targets are clearly evaluation only.
9. MLE unavailable results are explicit and never replaced by invented values. An unseen MLE context uses the stated unavailable text and never receives an implicit 0.5 probability, hidden smoothing, or silent backoff.
10. Markov Chain exposes exactly First-order Markov, Higher-order Markov, and Variable-order context analysis as nested workflows. First-order uses the immediately preceding symbol, Higher-order uses the previous k symbols, and Variable-order examines progressively longer preceding contexts. Variable-order is the recommended advanced workflow. No simulation or arbitrary order cap exists.
11. Training data alone fits Markov contexts. No context crosses records, and optional separate evaluation data does not contribute to fit, counts, support, smoothing, selection, or backoff. In-sample evaluation is labeled not held out.
12. Context evidence tables report occurrence count, A and B counts, support and sparse status, requested and actual depth, explicit suffix-backoff reason when used, probabilities, entropy, and target surprisal when defined. Context-depth entropy charts use a fixed 0 to 1 bits range, probability charts are optional, and no monotonicity is implied.
13. Experimental Context model, Context evidence, and Evaluation exports preserve the required stimulus fields, float64 source values, and at least 12 decimal places or exact retained representation.
14. HMM complements are derived read-only values. No unlock row control exists.
15. The visual layer uses exactly three decimal places for finite decimal values. Raw exports retain at least 12 decimals or exact retained representation.
16. Exact tables remain available for all charted values, and raw exports remain separate from display formatting.
17. The document owns vertical scroll. Horizontal table overflow is the only allowed internal overflow and is accessible.
18. Screenshots and keyboard checks at 375, 768, and 1280 px pass, including 200 percent zoom and reduced motion.
19. WCAG 2.2 AA contrast, focus, reflow, table, chart, and status requirements pass in the rendered application.
20. Sticky behavior is not claimed unless the scoped CSS implementation and all specified browser QA evidence exist.

### 8.4 Accepted debt

No accepted debt is recorded initially.

Any accepted debt must name the affected token or primitive, user impact, accessibility impact, scientific impact, reason it cannot be resolved now, owner, exit criterion, review date, and whether non sticky document flow is the conforming fallback. Accessibility failures, mathematically incorrect output, hidden assumptions, stale scientific output, contradictory display and export precision, silent input normalization, unsupported API claims, and nested vertical scrolling are never acceptable debt.
