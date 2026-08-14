import re
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).parents[2]
APP_PATH: Final = ROOT / "streamlit_app.py"
MODEL_INPUTS_PATH: Final = ROOT / "src/binary_entropy/ui/model_inputs.py"
RESULTS_VIEW_PATH: Final = ROOT / "src/binary_entropy/ui/results_view.py"
SUMMARY_PATH: Final = ROOT / "src/binary_entropy/ui/summary.py"
STYLES_PATH: Final = ROOT / "assets/styles.css"
SUMMARY_GRID_SELECTOR: Final = (
    '.st-key-results-region .st-key-summary-metrics [data-testid="stHorizontalBlock"]'
)
ACTUAL_TARGET_GRID_SELECTOR: Final = (
    ".st-key-results-region .st-key-actual-target-metrics "
    '[data-testid="stHorizontalBlock"]'
)
DOWNLOAD_GRID_SELECTOR: Final = (
    '.st-key-results-region .st-key-download-actions [data-testid="stHorizontalBlock"]'
)
METRIC_VALUE_SELECTOR: Final = '.st-key-results-region [data-testid="stMetricValue"]'
METRIC_LABEL_SELECTOR: Final = '.st-key-results-region [data-testid="stMetricLabel"]'
CAPTION_SELECTOR: Final = '.st-key-results-region [data-testid="stCaptionContainer"]'
PROBABILITY_ROW_GRID_SELECTOR: Final = (
    '.st-key-configuration-region [class*="st-key-probability-row-"] '
    '[data-testid="stHorizontalBlock"]'
)
PROBABILITY_ROW_COLUMN_SELECTOR: Final = (
    f'{PROBABILITY_ROW_GRID_SELECTOR} > [data-testid="stColumn"]'
)
PROBABILITY_NUMBER_SELECTOR: Final = (
    '.st-key-configuration-region [class*="st-key-probability-row-"] '
    '[data-testid="stNumberInput"]'
)


def _normalized(path: Path) -> str:
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


def _rule(css: str, selector: str) -> str:
    start = css.index(selector)
    return css[start : css.index("}", start)]


def _responsive_sections() -> tuple[str, str, str, str]:
    css = _normalized(STYLES_PATH)
    standard_start = css.index("@media (min-width: 48rem)")
    wide_start = css.index("@media (min-width: 80rem)")
    return css, css[:standard_start], css[standard_start:wide_start], css[wide_start:]


def test_app_regions_when_rendered_preserve_keyed_source_order() -> None:
    # Given
    app_source = APP_PATH.read_text(encoding="utf-8")
    results_source = RESULTS_VIEW_PATH.read_text(encoding="utf-8")
    summary_source = SUMMARY_PATH.read_text(encoding="utf-8")

    # When
    app_keys = tuple(re.findall(r'st\.container\(key="([a-z-]+)"\)', app_source))
    results_keys = tuple(
        re.findall(r'st\.container\(key="([a-z-]+)"\)', results_source)
    )
    summary_keys = tuple(
        re.findall(r'st\.container\(key="([a-z-]+)"\)', summary_source)
    )

    # Then
    assert app_keys == (
        "calculator-layout",
        "header-region",
        "configuration-region",
        "results-region",
        "interpretation-region",
    )
    assert results_keys == ("entropy-chart", "download-actions")
    assert summary_keys == ("summary-metrics", "actual-target-metrics")


def test_probability_rows_when_rendered_expose_stable_keyed_layout_hooks() -> None:
    # Given / When
    source = MODEL_INPUTS_PATH.read_text(encoding="utf-8")

    # Then
    assert 'row_key = keys[0].replace("_", "-")' in source
    assert 'st.container(key=f"probability-row-{row_key}")' in source
    assert "(INITIAL_0_KEY, INITIAL_1_KEY)" in source
    assert "TRANSITION_KEYS[row]" in source
    assert "EMISSION_KEYS[row]" in source


def test_responsive_css_when_viewport_is_compact_applies_result_contract() -> None:
    # Given
    css, compact, _, _ = _responsive_sections()

    # When
    compact_metric_grid = _rule(compact, SUMMARY_GRID_SELECTOR)
    compact_actual_target_grid = _rule(compact, ACTUAL_TARGET_GRID_SELECTOR)
    compact_download_grid = _rule(compact, DOWNLOAD_GRID_SELECTOR)
    compact_metric_value = _rule(compact, METRIC_VALUE_SELECTOR)
    compact_metric_label = _rule(compact, METRIC_LABEL_SELECTOR)
    compact_caption = _rule(compact, CAPTION_SELECTOR)
    compact_table = _rule(compact, ".st-key-results-region .prefix-table-overflow")

    # Then
    assert '.stApp [data-testid="stMainBlockContainer"] {' in compact
    assert ".main .block-container" not in css
    assert "padding-inline: var(--gutter-compact);" in compact
    assert ".st-key-calculator-layout .st-key-header-region h1 {" in compact
    assert "\nh1 {" not in css
    assert "font-size: var(--text-heading-1-compact);" in compact
    assert "block-size: var(--chart-height-compact);" in compact
    assert ".st-key-calculator-layout { display: grid;" in compact
    assert "grid-template-columns: minmax(0, 1fr);" in compact
    assert '.st-key-calculator-layout > [data-testid="stVerticalBlock"]' not in css
    assert "display: grid;" in compact_metric_grid
    assert "grid-template-columns: minmax(0, 1fr);" in compact_metric_grid
    assert "grid-template-columns: minmax(0, 1fr);" in compact_actual_target_grid
    assert "grid-template-columns: minmax(0, 1fr);" in compact_download_grid
    assert "white-space: normal;" in compact_metric_value
    assert "text-overflow: clip;" in compact_metric_value
    assert "overflow-wrap: anywhere;" in compact_metric_value
    assert "overflow: visible;" in compact_metric_label
    assert "white-space: normal;" in compact_metric_label
    assert "text-overflow: clip;" in compact_metric_label
    assert "overflow-wrap: anywhere;" in compact_metric_label
    assert "color: var(--color-text-secondary);" in compact_caption
    assert "font-size: var(--text-small);" in compact_caption
    assert "line-height: var(--line-small);" in compact_caption
    assert "max-inline-size: 100%;" in compact_table
    assert "overflow-x: scroll;" in compact_table
    assert ".prefix-table-overflow::-webkit-scrollbar" in compact
    assert ".prefix-table-overflow::-webkit-scrollbar-thumb" in compact
    assert "overflow-y: hidden;" in compact_table
    assert "block-size:" not in compact_table


def test_compact_css_when_probability_rows_render_preserves_two_columns() -> None:
    # Given
    _, compact, _, _ = _responsive_sections()

    # When
    probability_row_grid = _rule(compact, PROBABILITY_ROW_GRID_SELECTOR)
    probability_row_column = _rule(compact, PROBABILITY_ROW_COLUMN_SELECTOR)
    probability_number = _rule(compact, PROBABILITY_NUMBER_SELECTOR)

    # Then
    assert "display: grid;" in probability_row_grid
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in probability_row_grid
    assert "inline-size: 100% !important;" in probability_row_column
    assert "min-inline-size: 0 !important;" in probability_row_column
    assert "inline-size: 100%;" in probability_number


def test_responsive_css_when_viewport_is_standard_uses_two_result_columns() -> None:
    # Given
    _, _, standard, _ = _responsive_sections()

    # When
    metric_grid = _rule(standard, SUMMARY_GRID_SELECTOR)
    actual_target_grid = _rule(standard, ACTUAL_TARGET_GRID_SELECTOR)
    download_grid = _rule(standard, DOWNLOAD_GRID_SELECTOR)

    # Then
    assert "padding-inline: var(--gutter-standard);" in standard
    assert "font-size: var(--text-heading-1);" in standard
    assert "block-size: var(--chart-height-standard);" in standard
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in metric_grid
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in actual_target_grid
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in download_grid
    assert PROBABILITY_ROW_GRID_SELECTOR not in standard


def test_responsive_css_when_viewport_is_wide_uses_three_result_columns() -> None:
    # Given
    _, _, _, wide = _responsive_sections()

    # When
    metric_grid = _rule(wide, SUMMARY_GRID_SELECTOR)
    actual_target_grid = _rule(wide, ACTUAL_TARGET_GRID_SELECTOR)
    probability_row_grid = _rule(wide, PROBABILITY_ROW_GRID_SELECTOR)

    # Then
    assert "padding-inline: var(--gutter-wide);" in wide
    assert "repeat(var(--grid-columns), minmax(0, 1fr))" in wide
    assert '[data-testid="stLayoutWrapper"]:has(> .st-key-header-region)' in wide
    assert '[data-testid="stLayoutWrapper"]:has(> .st-key-configuration-region)' in wide
    assert '[data-testid="stLayoutWrapper"]:has(> .st-key-results-region)' in wide
    assert (
        '[data-testid="stLayoutWrapper"]:has(> .st-key-interpretation-region)' in wide
    )
    assert "grid-column: span var(--config-column-span);" in wide
    assert "grid-column: span var(--results-column-span);" in wide
    assert "grid-column: 1 / -1;" in wide
    assert "block-size: var(--chart-height-wide);" in wide
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in metric_grid
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in actual_target_grid
    assert "grid-template-columns: minmax(0, 1fr);" in probability_row_grid


def test_results_table_when_rendered_uses_safe_html_instead_of_native_table() -> None:
    # Given / When
    results_source = RESULTS_VIEW_PATH.read_text(encoding="utf-8")

    # Then
    assert "st.table(" not in results_source
    assert "st.html(prefix_table_html(" in results_source
