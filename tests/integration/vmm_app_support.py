"""Shared Streamlit driver for VMM integration scenarios."""

from pathlib import Path
from typing import Final

from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import ButtonGroup, Radio, Selectbox

APP_PATH: Final = Path(__file__).parents[2] / "streamlit_app.py"
TAB_KEY: Final = "workspace-method-tabs"
SCOPE_KEY: Final = "workspace-sequence-scope"
SUBVIEW_KEY: Final = "workspace-subview"


def workspace() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert not app.exception
    return app


def select_markov_chain(app: AppTest) -> AppTest:
    app.session_state[TAB_KEY] = "Markov Chain"
    _ = app.run()
    assert not app.exception
    return app


def _workspace_choice(
    app: AppTest, key: str
) -> Selectbox[str] | Radio[str] | ButtonGroup[str]:
    for item in app.selectbox:
        if item.key == key:
            return item
    for item in app.radio:
        if item.key == key:
            return item
    for item in app.button_group:
        if item.key == key:
            return item
    message = f"missing workspace choice: {key}"
    raise AssertionError(message)


def select_scope(app: AppTest, scope: str) -> AppTest:
    _ = _workspace_choice(app, SCOPE_KEY).set_value(scope)
    _ = app.run()
    assert not app.exception
    return select_markov_chain(app)


def select_subview(app: AppTest, subview: str) -> AppTest:
    _ = _workspace_choice(app, SUBVIEW_KEY).set_value(subview)
    _ = app.run()
    assert not app.exception
    return select_markov_chain(app)


def calculate(app: AppTest) -> AppTest:
    """Run the explicit calculation action and return the settled app."""
    _ = next(
        button for button in app.button if button.label == "Calculate selected methods"
    ).click()
    _ = app.run()
    assert not app.exception
    return select_markov_chain(app)
