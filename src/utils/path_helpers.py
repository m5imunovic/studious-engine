from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def get_data_root() -> Path:
    return get_project_root() / "data"


def get_config_root() -> Path:
    return get_project_root() / "configs"


def get_test_root() -> Path:
    return get_project_root() / "test"


def project_root_append(path: str) -> Path:
    return get_project_root() / path
