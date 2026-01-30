from pgmpy.factors.discrete import TabularCPD


def create_guilty_cpd(prior_prob, state_map):
    """Return a TabularCPD for the `Guilty` node using given prior and state map."""
    return TabularCPD(
        variable='Guilty', variable_card=2,
        values=[[1 - prior_prob], [prior_prob]],
        state_names=state_map
    )


def update_model_guilty_cpd(model, prior_prob, node_states, cpds_dict=None, edited_cpds=None):
    """Update the `Guilty` CPD on `model` using `prior_prob` unless it is marked edited.

    Parameters:
    - model: pgmpy BayesianNetwork
    - prior_prob: float
    - node_states: mapping of state names
    - cpds_dict: optional dict to update with new CPD
    - edited_cpds: optional dict-like marking edited CPDs

    Returns: True if updated, False if skipped because edited.
    """
    if edited_cpds is None:
        edited_cpds = {}

    if 'Guilty' in edited_cpds:
        return False

    cpd = create_guilty_cpd(prior_prob, node_states)
    try:
        model.remove_cpds('Guilty')
    except Exception:
        pass
    model.add_cpds(cpd)
    if cpds_dict is not None:
        cpds_dict['Guilty'] = cpd
    return True
