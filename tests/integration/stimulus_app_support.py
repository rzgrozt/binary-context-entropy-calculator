from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import ButtonGroup, Radio, Selectbox

from bspe import MatchingResult, SearchResult
from bspe.ui.session import WorkbenchCalculationRecord
from bspe.ui.stimulus_session import StimulusSearchFailure

APP_PATH: Final = Path(__file__).parents[2] / "streamlit_app.py"
MODE_KEY: Final = "workbench-workspace-mode"
SEARCH_SNAPSHOT_KEY: Final = "_stimulus_search_snapshot"
CANDIDATE_KEY: Final = "stimulus-search-candidate-id"
ANALYZER_TABS: Final = [
    "Method Comparison",
    "Markov Chain",
    "Hidden Markov Model",
    "Observed Shannon Entropy",
]


@dataclass(frozen=True, slots=True)
class SearchFixture:
    sequence_length: int = 8
    desired_count: int = 8
    candidate_limit: int = 32
    seed: int = 17


class _AnalyzerSession(Protocol):
    def __getitem__(self, key: str) -> WorkbenchCalculationRecord: ...


class _StimulusSession(Protocol):
    def __getitem__(self, key: str) -> SearchResult: ...


class _StimulusFailureSession(Protocol):
    def __getitem__(self, key: str) -> StimulusSearchFailure: ...


class _MatchingSession(Protocol):
    def __getitem__(self, key: str) -> MatchingResult: ...


def workbench() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert not app.exception
    return app


def choice_by_key(
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


def select_mode(app: AppTest, mode: str) -> AppTest:
    _ = choice_by_key(app, MODE_KEY).set_value(mode)
    _ = app.run()
    assert not app.exception
    return app


def configure_search(app: AppTest, fixture: SearchFixture) -> AppTest:
    app = select_mode(app, "Stimulus Search")
    numbers = {item.label: item for item in app.number_input}
    _ = numbers["Sequence length"].set_value(fixture.sequence_length)
    _ = numbers["Desired count"].set_value(fixture.desired_count)
    _ = numbers["Candidate limit"].set_value(fixture.candidate_limit)
    search_seed = next(item for item in app.text_input if item.label == "Search seed")
    _ = search_seed.set_value(str(fixture.seed))
    _ = app.run()
    assert not app.exception
    return app


def click_action(app: AppTest, label: str) -> AppTest:
    _ = next(item for item in app.button if item.label == label).click()
    _ = app.run()
    assert not app.exception
    return app


def analyzer_record(state: _AnalyzerSession) -> WorkbenchCalculationRecord:
    record = state["_calculation_record"]
    assert isinstance(record, WorkbenchCalculationRecord)
    return record


def search_snapshot(state: _StimulusSession) -> SearchResult:
    snapshot = state[SEARCH_SNAPSHOT_KEY]
    assert isinstance(snapshot, SearchResult)
    return snapshot


def search_failure(state: _StimulusFailureSession) -> StimulusSearchFailure:
    failure = state["_stimulus_search_failure"]
    assert isinstance(failure, StimulusSearchFailure)
    return failure


def matching_result(state: _MatchingSession) -> MatchingResult:
    result = state["_stimulus_matching_result"]
    assert isinstance(result, MatchingResult)
    return result
