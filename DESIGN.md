# Binary Sequence Predictive Entropy Calculator Design Contract

Status: Greenfield implementation contract

Product surface: Minimal single-page Streamlit scientific application

Design read: This is a scientific research tool for cognitive scientists. It must feel precise, trustworthy, quiet, and reproducible. Mathematical correctness, scientific transparency, reproducibility, and usability take precedence over visual polish, in that order.

## 0. Research Log

### 0.1 Research lanes

| Lane | Work recorded | Decision |
| --- | --- | --- |
| Embedded references | Shortlisted IBM, ClickHouse, and Sentry. | Chose the neutral operational taste discipline plus IBM as the primary reference because Carbon's structured data language fits scientific analysis. Use the grammar of clear hierarchy, tonal layers, compact controls, and rigorous tables without copying IBM branding, logos, content, or Carbon components. |
| UI/UX database | The lookup recommended a scientific data typography direction with restrained sans-serif text, tabular numerals, dense tables, and explicit units. | Accepted the typography and data-density direction. Rejected its dark/orange landing-page palette because it is inappropriate for a quiet research calculator and conflicts with the required light, neutral, single-blue-accent system. |
| Lazyweb | Skipped because no suitable Lazyweb tool is available in this environment. | No substitute screen research is required. Visual polish is explicitly secondary to correctness and transparency. |
| Imagen | Skipped because no suitable image-generation tool is available in this environment. | No concept image is required. Visual polish is explicitly secondary, and the application should not introduce imagery. |

### 0.2 Synthesis

- Reference qualities to retain: strict alignment, compact information density, visible labels, clear status language, strong focus treatment, and neutral tonal separation.
- Reference qualities to reject: corporate branding, marketing-page composition, oversized display type, decorative illustration, gradients, glass effects, dark mode, and motion for atmosphere.
- The product is not presented as using IBM Carbon. Carbon is an inspiration for information structure only.
- The first implementation is one calculator for one sequence. Batch workflows, dashboards, and multi-analysis navigation are outside this contract.

## 1. Product Character and Experience Principles

### 1.1 Audience and job

The primary user is a cognitive scientist configuring a two-state hidden Markov model, entering one binary sequence, and inspecting predictive entropy across sequence prefixes. The interface must support careful parameter entry, verification, calculation, inspection, and export without suggesting more certainty than the mathematics provides.

### 1.2 First implementation scope

The single-page application includes:

1. Customizable labels for the two hidden states and two observed symbols.
2. Editable initial, transition, and emission probabilities for a two-state HMM.
3. One sequence input with a documented accepted syntax.
4. An explicit calculation action.
5. Summary outputs with units and definitions.
6. A prefix result table.
7. A predictive entropy graph.
8. JSON preset import and export for model inputs.
9. CSV export for prefix results.

No batch mode, account flow, saved-project library, dashboard, marketing content, or decorative media belongs in the first implementation.

### 1.3 Experience principles

1. **Correct before clever.** Visual treatment must never obscure inputs, assumptions, precision, units, warnings, or failure states.
2. **Show the model.** Labels, matrices, sequence parsing, and calculation assumptions remain inspectable. Do not silently normalize or repair invalid probabilities.
3. **Make state explicit.** Users can distinguish uncalculated, calculating, valid, invalid, stale, empty, and export-ready states without relying on color.
4. **Keep the surface quiet.** Use neutral layers, one blue interactive accent family, near-sharp geometry, and borders instead of decorative shadows.
5. **Preserve reproducibility.** Presets, exports, displayed values, tooltips, and tables use stable ordering, named units, and one documented numeric precision policy.
6. **Prefer recognition over recall.** Labels stay visible, notation is explained near first use, and advanced explanation is available in help expanders.
7. **Respect Streamlit.** Prefer native, accessible Streamlit controls. Custom CSS expresses tokens and hierarchy but does not pretend that generated DOM or every widget state can be controlled exactly.

### 1.4 Voice and content

- Use sentence case and direct scientific language.
- Name quantities before showing notation: for example, `Predictive entropy, H(X_next | prefix)`.
- Put units in labels and headers, not only in help text.
- State assumptions and validation rules without promotional language.
- Error copy states what is wrong, where it is wrong, and how to correct it.
- Success copy confirms the completed operation and names the artifact or result affected.
- Never use emojis, slogans, anthropomorphic language, or claims of intelligence.

## 2. Foundations and Design Tokens

All custom visual values must resolve to the tokens below. If implementation needs a value not represented here, extend this section first. Do not place one-off colors, spacing, font sizes, radii, shadows, or transition values in application code.

### 2.1 Color tokens

The theme is light only. Neutral layers carry structure. Blue is reserved for interactive controls, links, focus, selected states, and the primary chart series. Red, green, and amber appear only for semantic status and are always paired with text or a shape.

| Token | Value | Use |
| --- | --- | --- |
| `--color-canvas` | `#f4f4f4` | Document background |
| `--color-surface` | `#ffffff` | Primary work surface, inputs, chart plot |
| `--color-surface-subtle` | `#fafafa` | Alternating or inset neutral layer |
| `--color-surface-strong` | `#e8e8e8` | Selected neutral row, disabled well, section separation |
| `--color-text-primary` | `#161616` | Headings, labels, values, primary body text |
| `--color-text-secondary` | `#525252` | Helper text, metadata, axis labels |
| `--color-text-placeholder` | `#6f6f6f` | Placeholder text only |
| `--color-text-disabled` | `#8d8d8d` | Disabled control text only; never essential content |
| `--color-border-subtle` | `#c6c6c6` | Dividers, table rules, quiet boundaries |
| `--color-border-strong` | `#8d8d8d` | Input boundaries and emphasized dividers |
| `--color-interactive` | `#0f62fe` | Primary action, links, focus ring, selected control, entropy series |
| `--color-interactive-hover` | `#0043ce` | Hover state within the same blue action family |
| `--color-interactive-active` | `#002d9c` | Pressed state within the same blue action family |
| `--color-on-interactive` | `#ffffff` | Text and simple glyphs on blue controls |
| `--color-focus` | `#0f62fe` | Visible keyboard focus ring |
| `--color-error` | `#b42318` | Error text, border, and status rule |
| `--color-error-surface` | `#fff1f0` | Error notice background |
| `--color-success` | `#166534` | Success text, border, and status rule |
| `--color-success-surface` | `#eef8f0` | Success notice background |
| `--color-warning` | `#7a4c00` | Warning text, border, and status rule |
| `--color-warning-surface` | `#fff7e6` | Warning notice background |

Implementation must verify WCAG contrast in the rendered Streamlit theme. Disabled styling is not a substitute for removing essential information.

### 2.2 Typography family tokens

No remote font request is required. Use the first locally available family in each stack.

| Token | Value | Use |
| --- | --- | --- |
| `--font-ui` | `"Source Sans 3", "Noto Sans", system-ui, sans-serif` | Interface copy and headings |
| `--font-data` | `"Source Sans 3", "Noto Sans", system-ui, sans-serif` | Numeric values with tabular and lining numeral features enabled |
| `--font-mono` | `"Cascadia Mono", "Liberation Mono", ui-monospace, monospace` | Sequence text, compact notation, and JSON examples |
| `--font-weight-regular` | `400` | Body, helper, and table cells |
| `--font-weight-semibold` | `600` | Labels, headings, table headers, and emphasized values |

Apply `font-variant-numeric: tabular-nums lining-nums` to matrices, metrics, axes, tooltips, tables, and numeric controls. Do not use font weight below 400.

### 2.3 Type scale tokens

| Token | Size | Line height | Use |
| --- | --- | --- | --- |
| `--text-caption` | `0.75rem` | `1rem` | Nonessential provenance and compact metadata |
| `--text-small` | `0.875rem` | `1.25rem` | Helper, validation, table cells, axis labels |
| `--text-body` | `1rem` | `1.5rem` | Body copy and controls |
| `--text-lead` | `1.125rem` | `1.625rem` | Short page purpose statement |
| `--text-heading-3` | `1.25rem` | `1.75rem` | Primitive or result group heading |
| `--text-heading-2` | `1.5rem` | `2rem` | Major section heading |
| `--text-heading-1` | `2rem` | `2.5rem` | Page title at standard and wide widths |
| `--text-heading-1-compact` | `1.75rem` | `2.25rem` | Page title at 375 px |

There is no display type tier. Headings use sentence case and align to the content grid.

### 2.4 Spacing tokens

The spacing system uses a 4 px base with larger composition steps drawn from the same scale.

| Token | Value | Typical use |
| --- | --- | --- |
| `--space-0` | `0` | Reset |
| `--space-1` | `0.25rem` | Label-to-required marker, compact inline separation |
| `--space-2` | `0.5rem` | Label-to-control, compact cell padding |
| `--space-3` | `0.75rem` | Standard cell padding, icon-to-text |
| `--space-4` | `1rem` | Control groups, compact page gutter |
| `--space-5` | `1.5rem` | Standard page gutter, primitive separation |
| `--space-6` | `2rem` | Wide page gutter, section interior |
| `--space-7` | `3rem` | Major section separation |
| `--space-8` | `4rem` | Maximum page section separation |

### 2.5 Geometry, border, and focus tokens

| Token | Value | Use |
| --- | --- | --- |
| `--radius-none` | `0` | Tables, notices, section layers |
| `--radius-control` | `0.125rem` | Inputs and buttons; near-sharp, never pill-shaped |
| `--border-width` | `1px` | Standard boundary |
| `--border-width-emphasis` | `2px` | Focus and status emphasis |
| `--focus-width` | `2px` | Focus outline |
| `--focus-offset` | `2px` | Space between control and focus outline |
| `--control-min-block` | `2.75rem` | Minimum 44 px interactive height |
| `--target-min-inline` | `2.75rem` | Minimum 44 px interactive width when the control is icon-sized |
| `--shadow-none` | `none` | All surfaces; use tone and borders instead of decorative shadow |
| `--motion-none` | `0ms` | All custom transitions and animations |

### 2.6 Layout tokens

| Token | Value | Use |
| --- | --- | --- |
| `--layout-min-supported` | `23.4375rem` | 375 px minimum target width |
| `--breakpoint-standard` | `48rem` | 768 px behavior boundary |
| `--breakpoint-wide` | `80rem` | 1280 px behavior boundary |
| `--content-max` | `70rem` | 1120 px maximum content width |
| `--gutter-compact` | `1rem` | 375 px page gutter |
| `--gutter-standard` | `1.5rem` | 768 px page gutter |
| `--gutter-wide` | `2rem` | 1280 px page gutter |
| `--grid-columns` | `12` | Wide composition grid |
| `--config-column-span` | `4` | Wide input/configuration region |
| `--results-column-span` | `8` | Wide results region |
| `--metric-min-inline` | `10rem` | Minimum metric width before stacking |

### 2.7 Table density tokens

| Token | Value | Use |
| --- | --- | --- |
| `--table-row-compact` | `2.25rem` | Minimum data row height at 768 px and above |
| `--table-row-touch` | `2.75rem` | Minimum row height at 375 px |
| `--table-cell-block` | `0.5rem` | Vertical cell padding |
| `--table-cell-inline` | `0.75rem` | Horizontal cell padding |
| `--table-header-block` | `0.75rem` | Header vertical padding |
| `--table-rule` | `1px solid var(--color-border-subtle)` | Row and column separation |

### 2.8 Chart tokens

| Token | Value | Use |
| --- | --- | --- |
| `--chart-series-primary` | `var(--color-interactive)` | Predictive entropy line and marker stroke |
| `--chart-marker-fill` | `var(--color-surface)` | Marker center so points remain distinct |
| `--chart-grid` | `var(--color-border-subtle)` | Major grid lines only |
| `--chart-axis` | `var(--color-text-secondary)` | Axis titles, ticks, and baseline |
| `--chart-plot` | `var(--color-surface)` | Plot region |
| `--chart-height-compact` | `18rem` | 375 px chart region |
| `--chart-height-standard` | `20rem` | 768 px chart region |
| `--chart-height-wide` | `22rem` | 1280 px chart region |
| `--chart-line-width` | `2px` | Series line |
| `--chart-marker-size` | `0.5rem` | Point marker diameter |
| `--chart-entropy-min` | `0` | Fixed binary entropy axis minimum |
| `--chart-entropy-max` | `1` | Fixed binary entropy axis maximum, in bits |

## 3. Typography and Scientific Content

### 3.1 Hierarchy

- Use one `h1` for `Binary Sequence Predictive Entropy Calculator`.
- Use `h2` for `Model`, `Sequence`, and `Results`.
- Use `h3` for matrix groups, table and chart regions, and help topics.
- Labels are semibold body or small text; they are never replaced by placeholders.
- Helper text appears directly below the relevant label or control and precedes an error message.
- Long methodological text belongs in a help expander, not in the main calculation path.

### 3.2 Numeric presentation

- Use a leading zero for values between -1 and 1.
- Right-align numeric table cells and matrix inputs where Streamlit permits.
- Use tabular numerals for every changing numeric value.
- Show units in metric labels, axis titles, table headers, and export headers.
- Use one canonical precision policy supplied by the calculation specification. The visible metric, table, Plotly tooltip, and CSV must not disagree because of independent formatting rules.
- Do not display more decimal places than the calculation can justify. Do not hide meaningful differences through premature rounding.
- Preserve full machine-readable values in exports according to the calculation specification, even when the interface uses a shorter display format.
- Never silently render `NaN`, positive infinity, or negative infinity as a normal result. Present an error notice with a reproducible diagnostic instead.

### 3.3 Scientific notation and labels

- Introduce notation with a plain-language name at first use.
- Render configured state and symbol labels as user data, never as trusted markup.
- Show the current state-label and symbol-label mapping near the matrices and in reproducibility details.
- The entropy unit is `bits` everywhere.
- Prefix indices and sequence positions must state whether counting begins at zero or one. The interface, table, chart, and CSV must use the same convention.
- Formulae are supporting explanations, not decorative display elements.

### 3.4 Reproducibility language

The result region must make the following inspectable without requiring memory:

- The configured state and symbol labels.
- Initial, transition, and emission probabilities.
- The parsed sequence length and accepted token count.
- The probability validation rule and numeric tolerance defined by the calculation specification.
- The output precision policy.
- The application or calculation schema version when one exists.

Do not claim a seed, method version, or dependency version unless the implementation actually records it.

## 4. Layout and Responsive Behavior

### 4.1 Page anatomy

This is a one-page research document, not a marketing page.

1. **Header:** compact title, one-sentence purpose, and optional version metadata.
2. **Model:** label controls, initial probabilities, transition matrix, emission matrix, and a concise scientific notice about row sums and validation.
3. **Sequence:** sequence input, syntax help, JSON preset input, and validation feedback.
4. **Calculate:** one explicit primary calculation action followed by status text.
5. **Results:** summary metrics, stale or result status, prefix table, entropy chart, reproducibility details, JSON preset download, and CSV result download.
6. **Help:** restrained expanders for input format, definitions, method, and reproducibility.

Source order follows this sequence at every width. A wide visual grid must not change keyboard or screen-reader order.

### 4.2 Tonal layering

- The canvas uses `--color-canvas`.
- Input and result work surfaces use `--color-surface` with borders, not floating cards.
- Inset matrix headers, table headers, and selected neutral regions may use `--color-surface-subtle` or `--color-surface-strong`.
- Major sections are separated by spacing, a tonal change, or one rule. Do not combine all three without a functional reason.
- No gradients, glass, backdrop blur, texture, decorative shadow, illustration, or image treatment is permitted.

### 4.3 Scroll ownership

The browser document is the single vertical scroll owner.

- Inputs, help content, result tables, and charts must not create independent vertical scrolling regions.
- Result tables may use horizontal overflow only when their semantic columns cannot reflow at narrow widths.
- Charts should resize to the available width. Horizontal overflow is allowed only if preserving readable axes and exact point inspection is otherwise impossible at 375 px.
- Any horizontal overflow region receives a visible or screen-reader instruction, a keyboard-reachable wrapper, and a clear focus outline.
- Do not use fixed-height result panels, sticky result panes, modal workflows, or nested scrolling.

### 4.4 Width-specific behavior

| Viewport | Composition | Component behavior |
| --- | --- | --- |
| 375 px | One column with `--gutter-compact`; compact page title; full-width fields and actions. | Matrix cells remain usable in a two-column numeric grid. Metrics stack. Downloads stack and fill available width. Table may scroll horizontally. Chart uses compact height and responsive width. |
| 768 px | One column with `--gutter-standard`; content remains in document order. | Related label fields may share a row. Metrics use two columns when `--metric-min-inline` is preserved. Downloads may sit inline. Table uses compact density and horizontal overflow only if required. |
| 1280 px | Centered `--content-max` area with `--gutter-wide` and a 12-column grid. Model and sequence configuration span 4 columns; results span 8 columns. | Metrics may use up to three columns. Table and chart use the results width. Help and downloads align to the results column. No element stretches beyond the content maximum. |

At intermediate widths, layout changes only when the minimum readable width of a primitive would be violated. Do not shrink labels, controls, targets, or numeric type below their tokens to preserve a column count.

### 4.5 Density and rhythm

- Use `--space-7` between major sections and `--space-5` between primitives.
- Use `--space-4` inside compact groups and `--space-2` between a label and its control.
- Keep tables dense but not compressed below the row-height tokens.
- Keep form groups visually closer to their own help and error text than to the next group.
- Do not use empty decorative bands, oversized top padding, or hero spacing.

## 5. Component Primitives

Use native Streamlit widgets where they satisfy the semantic requirement. The anatomy and states below describe product intent; they do not claim unrestricted control over Streamlit's generated DOM.

### 5.1 Shared state language

| State | Visual and textual contract |
| --- | --- |
| Default | Primary text, standard border, surface background, and a visible persistent label. |
| Hover | Only interactive elements change, using `--color-interactive-hover` or a stronger neutral boundary. No lift, glow, or motion. |
| Focus | `--focus-width` solid `--color-focus` with `--focus-offset`; never removed or replaced by color fill alone. |
| Disabled | Strong neutral surface and disabled text; control remains legible and an adjacent reason explains why it is unavailable. |
| Error | Error border or rule plus concise error text and programmatic association. Color is never the only signal. |
| Success | Success rule or text plus a plain-language confirmation. Do not use success styling for ordinary calculated values. |
| Loading | Static `Calculating...` or `Preparing file...` text. No spinner, shimmer, pulse, or animated skeleton. |
| Empty | Explain what input or action will produce content. Do not render an empty chart frame or a blank table. |
| Stale | Replace current outputs with a notice that inputs changed and recalculation is required. Do not present old values as current. |

### 5.2 Labeled field

**Use for:** state labels, symbol labels, sequence text, numeric values, and JSON preset input.

**Anatomy:** persistent label, optional notation or unit, control, helper text, and validation message. Required status appears as text, not color alone. Sequence text uses `--font-mono`; numeric controls use `--font-data`.

**States:**

- Default: label and helper text are visible; placeholder is an example, never the only instruction.
- Focus: control receives the focus token without layout movement.
- Disabled: preserve the value, dim only the control treatment, and state the reason nearby.
- Error: retain the user's input, mark the specific field, associate the error text, and move focus to the first invalid field after submission when Streamlit permits.
- Loading: fields may be temporarily disabled during calculation, with the page-level static loading message remaining visible.
- Empty: required empty fields show instruction after submission; before submission they remain neutral.

### 5.3 Probability matrix editor

**Use for:** initial probabilities, the 2 x 2 transition matrix, and the 2 x 2 emission matrix.

**Anatomy:** fieldset-like group, descriptive legend, persistent row and column labels, native numeric inputs, row-sum status, helper text, and group-level validation summary. Custom state and symbol labels update matrix headers as text.

**Behavior and states:**

- Inputs follow a row-major tab order that matches visual reading order.
- Each value exposes its probability range and accepted input format.
- Row-sum feedback names the row and displays its current sum with tabular numerals.
- Invalid cells receive an error boundary and specific message; the matrix also receives a concise group summary.
- Do not silently clamp, round, renormalize, or redistribute values.
- Default presets may populate an otherwise empty matrix. If required values are absent, show an empty instruction rather than inferred values.
- During calculation, preserve all entered values and use the shared disabled and loading treatments.
- A horizontal fallback is permitted only if customized labels make the matrix wider than the 375 px content area. The matrix must never gain vertical scrolling.

### 5.4 Scientific notice

**Use for:** model assumptions, method notes, validation summaries, stale results, warnings, errors, and operation success.

**Anatomy:** explicit text label such as `Method`, `Warning`, `Error`, or `Success`; short title; concise body; optional actionable text link. Use a tonal surface and a `--border-width-emphasis` leading rule. No decorative icon is required.

**States:** neutral method notices use neutral tokens; warning, error, and success notices use their semantic surface, text, and rule tokens. A notice is not focusable unless it contains a link or control. Empty notices are not rendered. Status notices persist until resolved or deliberately dismissed; they do not disappear on a timer.

### 5.5 Metric block

**Use for:** a small set of primary calculated outputs.

**Anatomy:** metric label, tabular value, unit, and one short definition or qualification. Use a tonal layer with a top rule, not a floating card or decorative shadow.

**States:**

- Default: value and unit are visually inseparable and the definition remains available.
- Loading: show static `Calculating...` text in place of the value.
- Empty: show `Not calculated` and the action required; do not use a lone dash.
- Error: show `Unavailable` with a linked or adjacent error explanation.
- Stale: remove the numeric value and show `Recalculation required`.
- Metric blocks are not interactive and therefore do not receive hover or focus styling.

### 5.6 Result table

**Use for:** one row per analyzed prefix in deterministic prefix order.

**Anatomy:** visible heading, concise caption, semantic column headers with units, body rows, and an optional horizontal-overflow instruction. Expected data includes prefix position, observed symbol where applicable, predictive probabilities labeled with the configured symbols, and predictive entropy in bits. Final column names must follow the calculation specification.

**Behavior and states:**

- Preserve prefix order; do not add sorting that can detach rows from sequence progression.
- Right-align numeric columns and use tabular numerals.
- Use row rules and neutral header tone rather than card borders around every cell.
- Default rows use compact density at 768 px and above and touch density at 375 px.
- Loading and empty states are textual replacements for the table, not placeholder rows.
- Error and stale states replace the table with a scientific notice.
- If horizontal scrolling is required, the wrapper is keyboard focusable and visibly focused. There is no vertical table scroll owner.

### 5.7 Plotly chart region

**Use for:** predictive entropy by prefix position.

**Anatomy:** visible heading, one-sentence chart summary, responsive Plotly plot, exact-value interaction, and a text/table fallback that contains the same data.

**Behavior and states:**

- Plot one blue line using `--chart-series-primary` with a visible marker at every calculated prefix. The line alone is not the only point cue.
- The x-axis is prefix position or prefix length, named according to the calculation convention.
- The y-axis title is `Predictive entropy (bits)` and its visible domain is fixed from 0 to 1.
- Major grid lines use `--chart-grid`; minor decorative grids are omitted.
- Tooltips identify the exact prefix and show the same canonical numeric value as the result table. They do not imply interpolation between markers.
- The accessible text/table fallback is present in document order, not hidden behind pointer-only interaction. It may be placed in an expander if its presence and purpose are announced before the chart.
- Loading, empty, error, and stale states replace the plot with static text or a scientific notice. Do not render empty axes.
- Disable Plotly transition animation and animated redraw.

### 5.8 Download action

**Use for:** `Download preset JSON` and `Download prefix CSV`.

**Anatomy:** secondary action label naming the artifact and format, optional short scope text, and adjacent readiness or error status. Use an outline or neutral button treatment; the calculation action remains the sole primary filled action.

**Behavior and states:**

- Default: action names the exact file type and content.
- Hover, focus, and active states follow the shared action tokens.
- Disabled: action remains visible before valid content exists and explains what must be calculated or corrected.
- Loading: show static `Preparing file...` text if generation is not immediate.
- Error: retain the action and show a specific export failure notice.
- Stale: disable both downloads until recalculation so exports cannot contradict current inputs.
- File contents use stable field and column order. File names are descriptive and deterministic; do not rely on a timestamp to identify scientific content.

### 5.9 Expander and help panel

**Use for:** `Input format`, `Method and definitions`, `Chart data table`, and `Reproducibility details`.

**Anatomy:** native expander button with a descriptive heading, expanded content, and any links in normal document order.

**Behavior and states:**

- Closed is the default for supporting explanation; critical validation and status information must never be hidden in an expander.
- The trigger is keyboard operable, exposes expanded/collapsed state, and uses the shared focus treatment where Streamlit permits.
- Expanded content has no nested vertical scrolling and no decorative open/close animation.
- If a help panel has no content, omit it rather than render an empty trigger.
- Error content within help uses a scientific notice; the trigger itself is not styled as an error unless opening it is required to resolve the error.

### 5.10 Primary calculation action

The single filled action is `Calculate entropy`. It submits the complete model and sequence as one explicit operation. It is disabled only when submission cannot produce a meaningful validation attempt or while a calculation is active. Focus, disabled, loading, and error behavior follows the shared states. Changing an input never launches an implicit calculation.

## 6. Interaction, Validation, and Motion

### 6.1 Calculation lifecycle

1. **Initial:** fields contain an explicit starter preset or clearly labeled empty values. Results show one empty-state instruction.
2. **Editing:** changing any submitted input invalidates existing results. Replace metrics, table, chart, and download readiness with the stale state.
3. **Validation:** validate on explicit calculation. Preserve all entries and identify both the first invalid field and the complete error summary.
4. **Calculating:** show persistent static `Calculating...` status. Do not use a spinner, shimmer, pulse, progress theater, or animated placeholder.
5. **Success:** announce completion, render results in the existing document flow, and enable applicable downloads.
6. **Failure:** preserve model and sequence inputs, show a reproducible error description, and do not leave partial outputs styled as valid.

JSON preset import populates fields only after schema and value validation. Import does not silently calculate. A failed import must not overwrite the current valid form.

### 6.2 Validation contract

- Probability fields accept only values permitted by the calculation specification and visibly state the range.
- Each required probability row must satisfy the documented sum rule and tolerance.
- State labels are nonempty and distinct; observed symbol labels are nonempty and distinct.
- Sequence parsing accepts only the documented delimiter and label syntax.
- An invalid sequence error identifies the first invalid token and its position, then summarizes any additional invalid tokens without flooding the page.
- Never silently coerce labels, discard sequence tokens, reorder values, or normalize probabilities.
- Validation messages remain until the responsible input changes or a valid submission succeeds.

### 6.3 Action hierarchy

- One primary action: calculate.
- Secondary actions: import preset, download preset JSON, and download prefix CSV.
- Tertiary actions: links that reveal definitions or move to related content.
- No floating actions, icon-only scientific actions, destructive action styling, or hidden context menus.
- Tooltips are supplemental only. Instructions required to complete the task stay visible or keyboard-reachable in help content.

### 6.4 Focus and status

- Focus order follows document order: labels, model inputs, sequence and preset input, calculate, results interactions, downloads, then help.
- Submission errors focus the error summary or first invalid field when Streamlit offers a stable accessible mechanism.
- Calculation completion uses a polite status announcement when Streamlit offers a stable accessible mechanism; it must not force an unexpected focus jump.
- Focus is never obscured by sticky content because the design uses no sticky controls or panels.

### 6.5 Motion policy

There is no decorative motion. Custom animation and transition duration is `--motion-none`.

- No entrance reveals, hover lift, parallax, animated gradients, pulsing status, chart tweening, skeleton shimmer, or decorative micro-interaction.
- State changes are immediate and communicated through text, border, tone, and semantic status.
- Native browser or Streamlit motion that cannot be removed must remain nonessential. Under `prefers-reduced-motion: reduce`, disable any optional smooth scrolling or framework transition that remains.
- Reduced-motion behavior does not remove information or interaction feedback.

## 7. Accessibility and Cognitive Usability

### 7.1 Standard

The target is WCAG 2.2 AA at 375, 768, and 1280 px, at 200 percent browser zoom, and under text-spacing overrides. Accessibility is a release requirement, not accepted debt.

### 7.2 Visual accessibility

- Normal text meets at least 4.5:1 contrast; large text and essential graphical objects meet at least 3:1.
- Focus indicators meet WCAG 2.2 focus appearance expectations and remain visible against every surface token.
- Error, success, warning, selection, and chart meaning are never conveyed by color alone.
- Text can reflow without two-dimensional page scrolling at 320 CSS px equivalent. Only documented data regions may scroll horizontally.
- Controls and targets meet the 44 px token where practical; tightly grouped native numeric fields still require adequate separation and a clear label.
- Do not reduce opacity on essential text to create hierarchy.

### 7.3 Keyboard and screen reader

- Every control is reachable and operable with a keyboard, with no custom keyboard trap.
- Matrix navigation follows row-major tab order; do not override native arrow-key behavior in numeric inputs.
- Expander triggers and download actions have visible text names and visible focus.
- Inputs expose labels, descriptions, units, required state, and errors programmatically where Streamlit permits.
- Matrices expose group names and row/column context. Tables use semantic headers and a caption or equivalent description.
- A horizontal overflow wrapper is keyboard reachable, named, and does not trap focus.
- The chart has a concise accessible name, summary, exact-value fallback table, and no pointer-only information.
- Status changes use persistent visible text first and an appropriate live announcement when the framework supports it reliably.

### 7.4 Chart accessibility

- The result table is the authoritative text fallback for every charted prefix and entropy value.
- The entropy series uses both a line and a marker at every point.
- Exact tooltips state prefix position and predictive entropy in bits using the canonical display precision.
- The y-axis is visibly and semantically labeled from 0 to 1 bits.
- A short text summary identifies the minimum, maximum, and final entropy only if those values are actually calculated and labeled by the implementation.
- Users can understand and export all chart data without hovering, perceiving blue, or operating a pointer.

### 7.5 Cognitive accessibility

- Keep the page in the stable order `Model`, `Sequence`, `Calculate`, `Results`.
- Keep labels visible while values are entered. Do not require users to remember matrix orientation or symbol mapping.
- Separate method explanation from task instructions, but keep validation rules near the relevant inputs.
- Use one term for each quantity throughout labels, table headers, chart axes, help, JSON fields, and CSV headers.
- Avoid unexplained abbreviations. Define HMM and predictive entropy at first use.
- Do not calculate on each keystroke, auto-dismiss messages, impose time limits, or reset inputs after an error.
- Make stale results impossible to mistake for current results.
- Keep paragraphs short, instructions procedural, and error recovery local.

### 7.6 Reduced motion and user settings

- Honor `prefers-reduced-motion` even though custom motion is already disabled.
- Respect browser zoom, platform contrast settings where Streamlit allows, and user font rendering.
- Do not add a dark theme in the first implementation. A future theme must be specified and contrast-tested as a complete token extension before use.

## 8. Governance, Streamlit Constraints, and Handoff

### 8.1 Contract authority

This document governs visual and interaction decisions for the first implementation. The priority order is:

1. Mathematical correctness.
2. Scientific transparency.
3. Reproducibility.
4. Usability and accessibility.
5. Visual polish.

When a visual choice conflicts with a higher priority, the higher priority wins and the contract is updated to record the resolution.

### 8.2 Streamlit implementation constraints

- This contract specifies intent, hierarchy, tokens, and states. It does not claim complete control over Streamlit's generated DOM, widget internals, browser accessibility tree, or responsive behavior.
- Prefer Streamlit theme settings and native controls before custom markup.
- If custom CSS is needed, centralize the tokens, scope selectors narrowly, and avoid brittle selectors tied to generated class names or undocumented DOM depth.
- Native widget limitations must not be hidden behind a claim of pixel parity. Record any verified limitation as design debt before release.
- Use Plotly configuration to consume chart tokens and disable animation where supported. Do not claim accessibility from Plotly alone; the result table remains required.
- Do not import, imitate, or claim IBM Carbon components. Do not use IBM logos, names as decoration, or brand-specific copy.
- Do not add a dependency solely to create decorative styling, imagery, or motion.

### 8.3 Token and primitive governance

- Every custom color resolves to a color token.
- Every custom spacing value resolves to the spacing or layout scale.
- Every custom type treatment resolves to the family, size, line-height, weight, and numeral tokens.
- Every border, radius, target size, chart dimension, table density, and focus treatment resolves to a token.
- New visual requirements extend this document before application code uses them.
- Compose the page from the primitives in Section 5. Do not create one-off card, alert, table, metric, or action variants.
- Semantic status colors are reserved for actual status. Blue is reserved for interaction, focus, selection, and the primary data series.

### 8.4 Implementation handoff sequence

1. Configure the light Streamlit theme and central token mapping.
2. Render a development-only primitive showcase covering the applicable default, focus, disabled, error, loading, empty, and stale states.
3. Verify matrix labels, row-major focus order, and validation associations before composing the page.
4. Build the page in the source order defined in Section 4.
5. Bind table, chart, tooltip, JSON, and CSV formatting to one canonical calculation precision policy.
6. Verify actual behavior at 375, 768, and 1280 px with keyboard-only use, browser zoom, reduced motion, invalid input, stale results, valid calculation, and both downloads.
7. Confirm the browser document is the only vertical scroll owner and that unavoidable horizontal overflow is labeled and keyboard reachable.

The primitive showcase is a development and QA artifact, not a second user-facing page or feature.

### 8.5 Release review checklist

- All model and sequence inputs have persistent visible labels and specific errors.
- Probability values and row sums are never silently altered.
- Current, stale, loading, empty, error, and valid result states are unambiguous.
- Every numeric surface uses tabular numerals, named units, and the canonical precision policy.
- Table, chart, tooltip, and exports agree for the same prefix.
- The entropy chart uses markers, exact tooltips, a 0 to 1 bit axis, and a complete text/table fallback.
- The document owns vertical scrolling; no panel, table, chart, or expander owns a nested vertical scroll.
- The layout works at 375, 768, and 1280 px without shrinking controls or type below tokens.
- Keyboard focus is visible and logical; errors and status are programmatically exposed where Streamlit supports it.
- WCAG 2.2 AA contrast and reflow checks pass in the rendered application.
- No gradient, glass effect, decorative shadow, dark mode, oversized display type, imagery, emoji, marketing section, or animation has been introduced.
- No implementation copy or artifact implies that IBM Carbon components are in use.

### 8.6 Accepted debt

None initially.

Any future accepted debt must name the affected primitive or token, user impact, accessibility impact, reason it cannot be resolved now, owner, exit criterion, and review date. Accessibility failures, incorrect or stale scientific output, contradictory export precision, silent input normalization, and nested vertical scrolling are not acceptable debt.
