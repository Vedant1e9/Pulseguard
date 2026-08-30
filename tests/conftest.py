"""Shared fixtures. The pipeline loads a trained bundle, so it is built once
per session and reused rather than reloaded for every test."""
import pytest

from engine.rule_pack import RulePack
from engine.safety_engine import SafetyEngine
from engine.triage_pipeline import TriagePipeline


@pytest.fixture(scope="session")
def rule_pack():
    return RulePack.load()


@pytest.fixture
def engine(rule_pack):
    return SafetyEngine(rule_pack)


@pytest.fixture(scope="session")
def pipeline():
    p = TriagePipeline()
    p.initialize(verbose=False)
    p.triage_all_patients()
    return p
