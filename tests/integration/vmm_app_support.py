"""Shared Streamlit driver for VMM integration scenarios."""

from pathlib import Path
from typing import Final

from streamlit.testing.v1 import AppTest

APP_PATH: Final = Path(__file__).parents[2] / "streamlit_app.py"


def workspace() -> AppTest:
    """Open the configured workbench and advance to its main surface."""
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert not app.exception
    _ = next(button for button in app.button if button.label == "Continue").click()
    _ = app.run()
    assert not app.exception
    return app


def calculate(app: AppTest) -> AppTest:
    """Run the explicit calculation action and return the settled app."""
    _ = next(
        button for button in app.button if button.label == "Calculate selected methods"
    ).click()
    _ = app.run()
    assert not app.exception
    return app
