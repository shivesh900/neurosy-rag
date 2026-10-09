import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def engine():
    from neurosy.model import train
    from neurosy.pipeline import NeuroSyRAG

    clf = train(per_disease=80, save=False, verbose=False)  # small + fast for CI
    return NeuroSyRAG(classifier=clf)
