import re
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).parents[2]
APP_PATH: Final = ROOT / "streamlit_app.py"
COMPARISON_PATH: Final = ROOT / "src/bspe/ui/comparison.py"
MODEL_INPUTS_PATH: Final = ROOT / "src/bspe/ui/model_inputs.py"
RESULTS_VIEW_PATH: Final = ROOT / "src/bspe/ui/results_view.py"
SIDEBAR_PATH: Final = ROOT / "src/bspe/ui/sidebar.py"
FORM_PATH: Final = ROOT / "src/bspe/ui/form.py"
MARKOV_VIEW_PATH: Final = ROOT / "src/bspe/ui/markov_view.py"
MARKOV_MODEL_VIEW_PATH: Final = ROOT / "src/bspe/ui/markov_model_view.py"
SUMMARY_PATH: Final = ROOT / "src/bspe/ui/summary.py"
SHANNON_RESULTS_PATH: Final = ROOT / "src/bspe/ui/shannon_results.py"
STYLES_PATH: Final = ROOT / "assets/styles.css"
MAIN_CONTAINER: Final = '.stApp [data-testid="stMainBlockContainer"]'
PROBABILITY_GRID: Final = (
    '[class*="st-key-probability-row-"] [data-testid="stHorizontalBlock"]'
)


def _normalized(path: Path) -> str:
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


def _rule(css: str, selector: str) -> str:
    start = css.index(selector)
    return css[start : css.index("}", start)]


def _responsive_sections() -> tuple[str, str, str, str]:
    css = _normalized(STYLES_PATH)
    standard_start = css.index("@media (min-width: 48rem)")
    desktop_start = css.index("@media (min-width: 80rem)")
    return (
        css,
        css[:standard_start],
        css[standard_start:desktop_start],
        css[desktop_start:],
    )


def test_app_shell_when_rendered_uses_native_sidebar_and_tracked_tabs() -> None:
    # Given / When
    app_source = APP_PATH.read_text(encoding="utf-8")
    results_source = RESULTS_VIEW_PATH.read_text(encoding="utf-8")

    # Then
    assert 'initial_sidebar_state="expanded"' in app_source
    assert "st.sidebar" in app_source
    assert "workbench-columns" not in app_source
    assert "st.columns(\n            (4, 6)," not in app_source
    assert "st.tabs(" in results_source
    assert 'key="workspace-method-tabs"' in results_source
    assert 'on_change="rerun"' in results_source
    assert ".open" in results_source
    calculated_path = results_source.index(
        "current = current_submission(form, record.success)"
    )
    active_tab_context = results_source.index(
        "with tabs[active_index]:", calculated_path
    )
    selection = results_source.index("selection = render_workspace_selection(")
    subview = results_source.index("subview = render_subview_control()")
    projection = results_source.index("projected = project_record(")
    dispatch = results_source.index("match active_tab:")
    assert active_tab_context < selection < subview < projection < dispatch


def test_sidebar_sections_when_rendered_default_collapse_exactly_five() -> None:
    # Given / When
    sidebar_source = SIDEBAR_PATH.read_text(encoding="utf-8")
    form_source = FORM_PATH.read_text(encoding="utf-8")
    source = sidebar_source + form_source

    # Then
    collapsed_labels = (
        "General settings",
        "Markov workflow settings",
        "Hidden Markov Model settings",
        "Training data intake",
        "Calculation status",
    )
    assert all(
        f'with st.expander("{label}", expanded=False):' in source
        for label in collapsed_labels
    )
    assert source.count("expanded=False") == len(collapsed_labels)
    assert 'with st.expander("Complete exports"):' in sidebar_source
    assert sidebar_source.index('st.button(\n        "Calculate selected methods"') < (
        sidebar_source.index('with st.expander("Calculation status"')
    )


def test_hmm_rows_when_rendered_have_one_editable_and_one_disabled_complement() -> None:
    # Given / When
    source = MODEL_INPUTS_PATH.read_text(encoding="utf-8")

    # Then
    assert source.count("columns[0].number_input(") == 1
    assert source.count("columns[1].number_input(") == 1
    assert "complement = 1.0 - source" in source
    assert "disabled=True" in source
    assert "unlock" not in source.lower()


def test_design_tokens_when_loaded_match_dark_scientific_contract() -> None:
    # Given / When
    css = _normalized(STYLES_PATH)

    # Then
    required_tokens = (
        "--color-canvas: #07111f",
        "--color-sidebar: #0a1628",
        "--color-surface: #0d1b2a",
        "--color-surface-raised: #122338",
        "--color-surface-strong: #183047",
        "--color-text-primary: #f3f7fc",
        "--color-accent: #22d3ee",
        "--radius-control: 0.25rem",
        "--radius-panel: 0.5rem",
        "--sidebar-width: 20rem",
        "--workspace-max: 96rem",
        "--text-heading-1: 1.875rem / 2.25rem",
    )
    assert all(token in css for token in required_tokens)
    assert "gradient" not in css
    assert "backdrop-filter" not in css
    assert "box-shadow" not in css
    assert "position: sticky" not in css


def test_responsive_css_when_narrow_keeps_one_native_workspace_flow() -> None:
    # Given
    _, compact, standard, _ = _responsive_sections()

    # When / Then
    assert "padding-inline: var(--gutter-narrow);" in compact
    assert "padding-inline: var(--gutter-standard);" in standard
    assert "workbench-columns" not in compact
    assert "workbench-columns" not in standard


def test_probability_rows_when_responsive_stack_then_split_at_standard() -> None:
    # Given
    _, compact, standard, desktop = _responsive_sections()

    # When
    assert PROBABILITY_GRID in compact
    assert PROBABILITY_GRID in standard
    compact_grid = _rule(compact, PROBABILITY_GRID)
    standard_grid = _rule(standard, PROBABILITY_GRID)

    # Then
    assert "grid-template-columns: minmax(0, 1fr);" in compact_grid
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in standard_grid
    assert PROBABILITY_GRID not in desktop


def test_main_container_when_toolbar_is_visible_composes_global_clearance() -> None:
    # Given
    css, compact, standard, desktop = _responsive_sections()

    # When
    main_container = _rule(compact, MAIN_CONTAINER)

    # Then
    assert "--toolbar-clearance: 3.75rem;" in css
    assert (
        "padding-block: calc(var(--toolbar-clearance) + var(--space-5)) var(--space-6);"
    ) in main_container
    assert "padding-block:" not in standard
    assert "padding-block:" not in desktop


def test_responsive_css_when_desktop_uses_sidebar_without_four_six_split() -> None:
    # Given
    css, _, _, desktop = _responsive_sections()

    # When / Then
    assert "padding-inline: var(--gutter-desktop);" in desktop
    assert "4fr" not in css
    assert "6fr" not in css
    assert "workbench-columns" not in css


def test_css_when_released_keeps_document_as_only_vertical_scroll_owner() -> None:
    # Given / When
    css = _normalized(STYLES_PATH)

    # Then
    assert "overflow-y: auto" not in css
    assert "overflow-y: scroll" not in css
    assert "overflow: auto" not in css
    assert "max-block-size:" not in css
    assert "position: sticky" not in css


def test_workspace_title_when_styled_uses_compact_heading_token() -> None:
    # Given / When
    css = _normalized(STYLES_PATH)

    # Then
    assert "--text-heading-1: 1.875rem / 2.25rem" in css
    title_font = (
        "font: var(--font-weight-semibold) var(--text-heading-1) var(--font-ui);"
    )
    assert title_font in css


def test_dataframes_when_wide_use_only_streamlit_native_horizontal_scroll() -> None:
    # Given / When
    css = _normalized(STYLES_PATH)

    # Then
    assert "overflow-x: auto" not in css


def test_results_when_rendered_use_native_dataframes_not_custom_html_tables() -> None:
    # Given / When
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            COMPARISON_PATH,
            MARKOV_MODEL_VIEW_PATH,
            MARKOV_VIEW_PATH,
            RESULTS_VIEW_PATH,
            SHANNON_RESULTS_PATH,
            SUMMARY_PATH,
        )
    )

    # Then
    assert "st.dataframe(" in sources
    assert "st.table(" not in sources
    assert "prefix_table_html" not in sources
    assert sources.count("st.dataframe(") == 7
    assert sources.count('height="content"') == 7


def test_markov_information_metrics_when_narrow_wrap_nested_text_nodes() -> None:
    # Given
    css = _normalized(STYLES_PATH)
    selector = '.st-key-results-region [data-testid="stMetricLabel"]'

    # When
    assert selector in css
    metric_rule = _rule(css, selector)

    # Then
    assert '[data-testid="stMetricLabel"] p' in metric_rule
    assert '[data-testid="stMetricValue"] p' in metric_rule
    assert "white-space: normal !important;" in metric_rule
    assert "overflow: visible !important;" in metric_rule
    assert "text-overflow: clip !important;" in metric_rule


def test_results_when_tables_can_overflow_explain_horizontal_scrolling() -> None:
    # Given / When
    source = RESULTS_VIEW_PATH.read_text(encoding="utf-8")

    # Then
    assert "Wide comparison and result tables scroll horizontally." in source
