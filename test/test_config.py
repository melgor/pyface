import pytest

from pyface.config import load_config


def test_config():
    """Test if config can be loaded"""
    config_path = "test/assets/test_config.yaml"
    config = load_config(config_path)
    assert config is not None


def test_config_bad():
    """Test if exception will be raised"""
    config_path = "test/assets/test_config_bad.yaml"
    with pytest.raises(AssertionError):
        config = load_config(config_path)  # noqa: F841
