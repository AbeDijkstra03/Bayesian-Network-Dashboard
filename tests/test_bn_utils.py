import sys
import os
import pytest
from pgmpy.factors.discrete import TabularCPD

# Ensure repo root is on sys.path so tests can import bn_utils
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bn_utils import create_guilty_cpd
import numpy as np


def test_create_guilty_cpd_default():
    state_map = {'Guilty': ['Innocent', 'Guilty']}
    prior = 1 / 200000
    cpd = create_guilty_cpd(prior, state_map)
    assert isinstance(cpd, TabularCPD)
    # Check values (handle scalar or array storage across pgmpy versions)
    vals = np.ravel(np.array(cpd.values))
    assert pytest.approx(vals[1], rel=1e-12) == prior
    assert pytest.approx(vals[0] + vals[1], rel=1e-12) == 1.0


def test_create_guilty_cpd_higher_prior():
    state_map = {'Guilty': ['Innocent', 'Guilty']}
    prior = 1 / 1000
    cpd = create_guilty_cpd(prior, state_map)
    assert isinstance(cpd, TabularCPD)
    vals = np.ravel(np.array(cpd.values))
    assert pytest.approx(vals[1], rel=1e-12) == prior
    assert pytest.approx(vals[0] + vals[1], rel=1e-12) == 1.0
