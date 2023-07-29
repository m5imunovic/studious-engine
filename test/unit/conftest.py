from pathlib import Path

import pytest

import utils.path_helpers as ph


@pytest.fixture
def four_faces_path() -> Path:
    return ph.get_data_root() / "four_faces"
