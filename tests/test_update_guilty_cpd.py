import sys
import os
import numpy as np
import pytest
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork

# Ensure repo root is on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bn_utils import create_guilty_cpd, update_model_guilty_cpd


def make_model_with_guilty(state_map, prior):
    # Create an empty BayesianNetwork (no edges) and add the Guilty CPD
    model = BayesianNetwork([])
    model.add_nodes_from(['Guilty'])
    cpd = create_guilty_cpd(prior, state_map)
    model.add_cpds(cpd)
    return model, {'Guilty': cpd}


def test_update_applies_when_not_edited():
    state_map = {'Guilty': ['Innocent', 'Guilty']}
    prior_old = 1 / 200000
    prior_new = 1 / 1000
    model, cpds = make_model_with_guilty(state_map, prior_old)

    edited = {}
    updated = update_model_guilty_cpd(model, prior_new, state_map, cpds_dict=cpds, edited_cpds=edited)
    assert updated is True
    vals = np.ravel(np.array(cpds['Guilty'].values))
    assert vals[1] == pytest.approx(prior_new)


def test_update_skipped_when_edited():
    state_map = {'Guilty': ['Innocent', 'Guilty']}
    prior_old = 1 / 200000
    prior_new = 1 / 1000
    model, cpds = make_model_with_guilty(state_map, prior_old)

    edited = {'Guilty': True}
    updated = update_model_guilty_cpd(model, prior_new, state_map, cpds_dict=cpds, edited_cpds=edited)
    assert updated is False
    vals = np.ravel(np.array(cpds['Guilty'].values))
    assert vals[1] == pytest.approx(prior_old)
