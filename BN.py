import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from fractions import Fraction

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
PRIOR_GUILTY = 1 / 200000
RMP = 1 / 200000000  # Random Match Probability
NETWORKS_DIR = Path("bayesian_networks")
NETWORKS_DIR.mkdir(exist_ok=True)

# Default CPT tolerance for validation
CPT_TOLERANCE = 1e-9

# ==========================================
# NETWORK PERSISTENCE FUNCTIONS
# ==========================================

def serialize_cpd(cpd: TabularCPD) -> Dict[str, Any]:
    """Convert a TabularCPD to a serializable format"""
    return {
        'variable': cpd.variable,
        'variable_card': int(cpd.cardinality[0]),
        'values': cpd.values.tolist(),
        'evidence': cpd.variables[1:] if len(cpd.variables) > 1 else None,
        'evidence_card': [int(c) for c in cpd.cardinality[1:]] if len(cpd.cardinality) > 1 else None,
        'state_names': {k: list(v) for k, v in cpd.state_names.items()}
    }

def deserialize_cpd(cpd_dict: Dict[str, Any]) -> TabularCPD:
    """Reconstruct a TabularCPD from serialized format"""
    return TabularCPD(
        variable=cpd_dict['variable'],
        variable_card=cpd_dict['variable_card'],
        values=cpd_dict['values'],
        evidence=cpd_dict['evidence'],
        evidence_card=cpd_dict['evidence_card'],
        state_names=cpd_dict['state_names']
    )

def serialize_network(network_name: str, model: BayesianNetwork, 
                     cpds_dict: Dict[str, TabularCPD], 
                     node_states: Dict[str, List[str]]) -> str:
    """Save a network structure and CPTs to JSON"""
    assert network_name, "Network name cannot be empty"
    
    network_data = {
        'name': network_name,
        'created': datetime.now().isoformat(),
        'nodes': list(model.nodes()),
        'edges': list(model.edges()),
        'node_states': node_states,
        'cpds': {name: serialize_cpd(cpd) for name, cpd in cpds_dict.items()}
    }
    
    file_path = NETWORKS_DIR / f"{network_name}.json"
    with open(file_path, 'w') as f:
        json.dump(network_data, f, indent=2)
    
    return str(file_path)

def deserialize_network(network_name: str) -> Optional[Dict[str, Any]]:
    """Load a network structure and CPTs from JSON"""
    file_path = NETWORKS_DIR / f"{network_name}.json"
    
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r') as f:
            network_data = json.load(f)
        
        # Recreate model
        model = BayesianNetwork(network_data['edges'])
        
        # Recreate CPDs
        cpds = {}
        for cpd_name, cpd_dict in network_data['cpds'].items():
            cpds[cpd_name] = deserialize_cpd(cpd_dict)
        
        model.add_cpds(*cpds.values())
        
        # Validate if possible
        try:
            model.check_model()
        except Exception as e:
            st.warning(f"Loaded model may have validation issues: {e}")
        
        return {
            'model': model,
            'cpds': cpds,
            'node_states': network_data['node_states']
        }
    except Exception as e:
        st.error(f"Error loading network: {e}")
        return None

def list_networks() -> List[Dict[str, Any]]:
    """Get list of saved networks"""
    if not NETWORKS_DIR.exists():
        return []
    
    networks = []
    for file in NETWORKS_DIR.glob("*.json"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                networks.append({
                    'name': data['name'],
                    'created': data.get('created', 'Unknown'),
                    'nodes': len(data['nodes']),
                    'edges': len(data['edges'])
                })
        except Exception:
            pass  # Skip invalid files
    
    return sorted(networks, key=lambda x: x['created'], reverse=True)

def delete_network(network_name: str) -> bool:
    """Delete a saved network"""
    file_path = NETWORKS_DIR / f"{network_name}.json"
    if file_path.exists():
        try:
            file_path.unlink()
            return True
        except Exception:
            return False
    return False 

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_node_parents(model: BayesianNetwork, node: str) -> List[str]:
    """Get parent nodes of a given node"""
    return list(model.get_parents(node))

def cpd_to_dataframe(cpd: TabularCPD, states: List[str]) -> pd.DataFrame:
    """Convert a TabularCPD to a DataFrame for display"""
    values = cpd.values
    parents = cpd.variables[1:] if len(cpd.variables) > 1 else []
    
    if not parents:
        df = pd.DataFrame(values, index=states, columns=["Probability"])
    else:
        parent_states = [cpd.state_names[p] for p in parents]
        combinations = list(itertools.product(*parent_states))
        cols = []
        for c in combinations:
            if len(c) == 1:
                cols.append(str(c[0]))
            else:
                cols.append(str(c).replace("'", "").replace("(", "").replace(")", "").replace(", ", "|"))
        df = pd.DataFrame(values, index=states, columns=cols)
    
    return df

def generate_cpt_template(model: BayesianNetwork, node: str, 
                         states: List[str], 
                         parent_states: Dict[str, List[str]]) -> pd.DataFrame:
    """Generate a template CPT DataFrame with uniform probabilities"""
    parents = get_node_parents(model, node)
    if not parents:
        cols = ["Probability"]
    else:
        p_lists = [parent_states[p] for p in parents]
        combinations = list(itertools.product(*p_lists))
        cols = [str(c).replace("'", "").replace("(", "").replace(")", "").replace(", ", "|") 
                for c in combinations]
    
    default_val = 1.0 / len(states)
    df = pd.DataFrame(default_val, index=states, columns=cols)
    return df

def validate_cpt_columns(df: pd.DataFrame, tolerance: float = CPT_TOLERANCE) -> Tuple[bool, List[Tuple[str, float]]]:
    """
    Validate that all columns in a CPT sum to 1.0
    
    Returns:
        Tuple of (is_valid, list of invalid columns with their sums)
    """
    sums = df.sum(axis=0)
    invalid_cols = []
    
    for col_name, sum_val in zip(df.columns, sums.values):
        if abs(sum_val - 1.0) > tolerance:
            invalid_cols.append((col_name, sum_val))
    
    return len(invalid_cols) == 0, invalid_cols

def update_cpd_from_df(model: BayesianNetwork, node: str, df: pd.DataFrame, 
                      states: List[str], parent_states: Dict[str, List[str]]) -> Tuple[bool, str]:
    """
    Update a CPD in the model from a DataFrame
    
    Returns:
        Tuple of (success, message)
    """
    values = df.values.tolist()
    parents = get_node_parents(model, node)
    card = len(states)
    evidence = parents if parents else None
    evidence_card = [len(parent_states[p]) for p in parents] if parents else None

    try:
        # Validate column sums
        is_valid, invalid_cols = validate_cpt_columns(df)
        
        if not is_valid:
            error_msg = "Invalid CPT: Columns must sum to exactly 1.0\n"
            for col_name, sum_val in invalid_cols:
                error_msg += f"  - Column '{col_name}': {sum_val:.15f} (error: {sum_val - 1.0:.15f})\n"
            return False, error_msg
        
        # Create the CPD
        cpd = TabularCPD(
            variable=node, variable_card=card, values=values,
            evidence=evidence, evidence_card=evidence_card,
            state_names={node: states, **{p: parent_states[p] for p in parents}}
        )
        
        # Validate CPD
        if not cpd.is_valid_cpd():
            return False, "CPD failed validation"
        
        # Update the model
        model.remove_cpds(node)
        model.add_cpds(cpd)
        
        return True, "CPT Updated Successfully!"
        
    except Exception as e:
        return False, f"Error creating CPD: {str(e)}"

# ==========================================
# NETWORK STRUCTURE EDITOR FUNCTIONS
# ==========================================

def add_node_to_network(model: BayesianNetwork, node_states: Dict[str, List[str]], 
                       node_name: str, num_states: int = 2) -> Tuple[bool, str]:
    """Add a new node to the network"""
    if not node_name or not node_name.strip():
        return False, "Node name cannot be empty"
    
    node_name = node_name.strip()
    
    if node_name in model.nodes():
        return False, f"Node '{node_name}' already exists"
    
    if num_states < 2:
        return False, "Node must have at least 2 states"
    
    model.add_node(node_name)
    node_states[node_name] = [f"State_{i}" for i in range(num_states)]
    
    return True, f"Added node '{node_name}' with {num_states} states"

def remove_node_from_network(model: BayesianNetwork, cpds: Dict[str, TabularCPD],
                            node_states: Dict[str, List[str]], 
                            node_name: str) -> Tuple[bool, str]:
    """Remove a node from the network"""
    if node_name not in model.nodes():
        return False, f"Node '{node_name}' not found"
    
    # Remove associated CPD
    if node_name in cpds:
        try:
            model.remove_cpds(node_name)
        except Exception:
            pass
        del cpds[node_name]
    
    # Remove node
    model.remove_node(node_name)
    if node_name in node_states:
        del node_states[node_name]
    
    return True, f"Removed node '{node_name}'"

def add_edge_to_network(model: BayesianNetwork, parent: str, child: str) -> Tuple[bool, str]:
    """Add an edge from parent to child"""
    if parent not in model.nodes() or child not in model.nodes():
        return False, "One or both nodes don't exist"
    
    if parent == child:
        return False, "Cannot add self-loop"
    
    if model.has_edge(parent, child):
        return False, f"Edge from '{parent}' to '{child}' already exists"
    
    # Check for cycles - pgmpy raises ValueError if edge would create a cycle
    try:
        model.add_edge(parent, child)
    except ValueError as e:
        # pgmpy detected a cycle or self-loop
        if "loop" in str(e).lower():
            return False, f"Adding this edge would create a cycle: {e}"
        else:
            return False, f"Cannot add edge: {e}"
    
    # Double-check with networkx (belt and suspenders)
    if not nx.is_directed_acyclic_graph(model):
        model.remove_edge(parent, child)
        return False, "Adding this edge would create a cycle"
    
    return True, f"Added edge from '{parent}' to '{child}'"

def remove_edge_from_network(model: BayesianNetwork, cpds: Dict[str, TabularCPD],
                            parent: str, child: str) -> Tuple[bool, str]:
    """Remove an edge"""
    if not model.has_edge(parent, child):
        return False, f"Edge from '{parent}' to '{child}' not found"
    
    model.remove_edge(parent, child)
    
    # Remove associated CPDs for affected children
    if child in cpds:
        try:
            model.remove_cpds(child)
        except Exception:
            pass
        del cpds[child]
    
    return True, f"Removed edge from '{parent}' to '{child}'"

def create_default_cpd(node: str, node_states: Dict[str, List[str]], 
                      p_match_given_innocent: float) -> TabularCPD:
    """Create default CPD based on node name"""
    states = node_states
    
    if node == 'Guilty':
        return TabularCPD(
            variable='Guilty', variable_card=2,
            values=[[1 - PRIOR_GUILTY], [PRIOR_GUILTY]],
            state_names=states
        )
    elif node == 'Alibi':
        return TabularCPD(
            variable='Alibi', variable_card=2,
            values=[
                [0.50, 0.75],
                [0.50, 0.25]
            ],
            evidence=['Guilty'], evidence_card=[2],
            state_names=states
        )
    elif node == 'Desc_Match':
        return TabularCPD(
            variable='Desc_Match', variable_card=2,
            values=[
                [0.90, 0.10],
                [0.10, 0.90]
            ],
            evidence=['Guilty'], evidence_card=[2],
            state_names=states
        )
    elif node == 'DNA_Match':
        return TabularCPD(
            variable='DNA_Match', variable_card=2,
            values=[
                [1 - p_match_given_innocent, 0.0],
                [p_match_given_innocent, 1.0]
            ],
            evidence=['Guilty'], evidence_card=[2],
            state_names=states
        )
    else:
        raise ValueError(f"No default CPD defined for node '{node}'")

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'current_network_name' not in st.session_state:
        st.session_state['current_network_name'] = 'DNA Case'
    
    if 'networks' not in st.session_state:
        st.session_state['networks'] = {}
    
    if 'cpt_save_message' not in st.session_state:
        st.session_state['cpt_save_message'] = None
    
    if 'cpt_original_values' not in st.session_state:
        st.session_state['cpt_original_values'] = {}
    
    if 'edited_cpds' not in st.session_state:
        st.session_state['edited_cpds'] = {}
    
    # Initialize model if not exists
    if 'model' not in st.session_state:
        # Create default network structure
        model = BayesianNetwork([
            ('Guilty', 'DNA_Match'),
            ('Guilty', 'Alibi'),
            ('Guilty', 'Desc_Match')
        ])
        
        # Define states
        state_map = {
            'Guilty': ['Innocent', 'Guilty'],
            'DNA_Match': ['No_Match', 'Match'],
            'Alibi': ['No_Alibi', 'Yes_Alibi'],
            'Desc_Match': ['No_Match', 'Match']
        }
        
        # Create CPDs with initial values
        p_match_initial = RMP  # Will be updated by slider
        
        cpd_guilty = create_default_cpd('Guilty', state_map, p_match_initial)
        cpd_alibi = create_default_cpd('Alibi', state_map, p_match_initial)
        cpd_desc = create_default_cpd('Desc_Match', state_map, p_match_initial)
        cpd_dna = create_default_cpd('DNA_Match', state_map, p_match_initial)
        
        model.add_cpds(cpd_guilty, cpd_alibi, cpd_desc, cpd_dna)
        
        # Validate model
        try:
            model.check_model()
        except Exception as e:
            st.error(f"Model validation failed: {e}")
        
        st.session_state['model'] = model
        st.session_state['node_states'] = state_map
        st.session_state['cpds'] = {
            'Guilty': cpd_guilty,
            'Alibi': cpd_alibi,
            'Desc_Match': cpd_desc,
            'DNA_Match': cpd_dna
        }

# ==========================================
# MAIN APPLICATION
# ==========================================

def main():
    """Main application entry point"""
    st.set_page_config(page_title="Bayesian Network Lab", layout="wide")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar: DNA Evidence Settings
    st.sidebar.title("🧬 DNA Evidence Settings")
    st.sidebar.markdown("Adjust the reliability of the DNA test to see how it impacts the verdict.")
    
    # Use unique key for lab error slider
    lab_error_rate = st.sidebar.slider(
        "Lab Error Rate (False Positive)", 
        min_value=0.0, 
        max_value=0.01, 
        value=0.0, 
        step=0.0001,
        format="%.4f",
        key="lab_error_slider",  # UNIQUE KEY
        help="Probability that an Innocent person matches due to lab error (contamination, swap)."
    )
    
    p_match_given_innocent = RMP + lab_error_rate
    
    # Update DNA CPD if not manually edited
    model = st.session_state['model']
    node_states = st.session_state['node_states']
    
    if 'DNA_Match' in model.nodes() and 'DNA_Match' not in st.session_state.get('edited_cpds', {}):
        try:
            cpd_dna_new = create_default_cpd('DNA_Match', node_states, p_match_given_innocent)
            model.remove_cpds('DNA_Match')
            model.add_cpds(cpd_dna_new)
            st.session_state['cpds']['DNA_Match'] = cpd_dna_new
        except Exception as e:
            st.sidebar.warning(f"Could not update DNA CPD: {e}")
    
    # Network Management Sidebar
    with st.sidebar:
        st.divider()
        st.subheader("📁 Network Management")
        
        st.write(f"**Current Network:** `{st.session_state['current_network_name']}`")
        
        # Create New Network
        st.markdown("#### Create New Network")
        new_net_name = st.text_input("Network name:", key="new_network_name_input")
        if st.button("Create Network", key="create_new_network_btn"):
            if new_net_name and new_net_name.strip():
                st.session_state['model'] = BayesianNetwork()
                st.session_state['cpds'] = {}
                st.session_state['edited_cpds'] = {}
                st.session_state['node_states'] = {}
                st.session_state['cpt_original_values'] = {}
                st.session_state['current_network_name'] = new_net_name.strip()
                st.success(f"✅ Created network '{new_net_name}'")
                st.rerun()
            else:
                st.error("Please enter a network name")
        
        # List Saved Networks
        st.markdown("#### Saved Networks")
        saved_networks = list_networks()
        
        if saved_networks:
            for net in saved_networks:
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"📊 {net['name']}", key=f"load_network_{net['name']}"):
                        loaded_data = deserialize_network(net['name'])
                        if loaded_data:
                            st.session_state['model'] = loaded_data['model']
                            st.session_state['cpds'] = loaded_data['cpds']
                            st.session_state['node_states'] = loaded_data['node_states']
                            st.session_state['edited_cpds'] = {}
                            st.session_state['cpt_original_values'] = {}
                            st.session_state['current_network_name'] = net['name']
                            st.success(f"✅ Loaded '{net['name']}'")
                            st.rerun()
                        else:
                            st.error(f"Failed to load '{net['name']}'")
                with col2:
                    if st.button("🗑️", key=f"delete_network_{net['name']}", help="Delete network"):
                        if delete_network(net['name']):
                            st.success(f"Deleted '{net['name']}'")
                            if st.session_state['current_network_name'] == net['name']:
                                st.session_state['current_network_name'] = 'DNA Case'
                            st.rerun()
        else:
            st.info("No saved networks yet. Create one or save the current network.")
        
        # Save Current Network
        st.markdown("#### Save Current Network")
        save_name = st.text_input("Save as:", value=st.session_state['current_network_name'], 
                                  key="save_network_name_input")
        if st.button("💾 Save Network", key="save_current_network_btn"):
            if save_name and save_name.strip():
                try:
                    serialize_network(save_name.strip(), st.session_state['model'], 
                                    st.session_state['cpds'], st.session_state['node_states'])
                    st.success(f"✅ Saved as '{save_name}'")
                    st.session_state['current_network_name'] = save_name.strip()
                except Exception as e:
                    st.error(f"Failed to save: {e}")
            else:
                st.error("Please enter a name")
    
    # Main Content
    st.title("Bayesian Network Lab: The DNA Case ⚖️")
    
    # Understanding Section
    with st.expander("📚 Understanding the Assumptions & Calculations", expanded=False):
        st.markdown("""
        ## The Mathematics Behind the Verdict
        
        This Bayesian Network models a criminal case using **Bayes' Theorem** to calculate the probability of guilt given evidence.
        
        ### Core Formula
        
        We calculate **Posterior Odds** using:
        
        $$\\text{Posterior Odds} = \\text{Prior Odds} \\times \\prod_{i} \\text{Likelihood Ratio}_i$$
        
        Then convert to probability:
        
        $$P(\\text{Guilty} | \\text{Evidence}) = \\frac{\\text{Odds}}{1 + \\text{Odds}}$$
        
        ---
        
        ### Key Assumptions in This Model
        
        #### 1. **Prior Probability (Base Rate)**
        - **Assumption**: The suspect is randomly selected from 200,000 local males aged 15-60
        - **Formula**: $P(\\text{Guilty}) = 1/200,000 = 0.0005\\%$
        - **⚠️ Critical Issue**: This assumes a "cold hit" scenario. In reality, if police had other reasons to suspect this person (motive, opportunity), the prior should be much higher (e.g., 1/10 or 1/100).
        
        #### 2. **DNA Evidence**
        - **Assumption**: $P(\\text{Match}|\\text{Guilty}) = 1.0$ (guilty person always matches)
        - **Assumption**: $P(\\text{Match}|\\text{Innocent}) = 1/200,000,000$ (Random Match Probability)
        - **⚠️ Critical Issue**: **Lab Error is NOT included by default!**
        
        **The "Zero Lab Error" Fallacy:**
        - While random genetic matches are extremely rare (1 in 200 million), **human errors** (sample mix-ups, contamination) occur at rates of 1/1,000 to 1/10,000
        - If lab error rate = 0.001 (1/1,000), the DNA likelihood ratio drops from 200,000,000 to just 1,000!
        - **Use the slider** in the sidebar to see this effect
        
        #### 3. **Conditional Independence (Naive Bayes)**
        - **Assumption**: Evidence types are independent given guilt status
        - **Formula**: $P(DNA \\cap Alibi \\cap Desc | G) = P(DNA|G) \\times P(Alibi|G) \\times P(Desc|G)$
        - **⚠️ Issue**: In reality, having a solid alibi might correlate with not matching witness descriptions
        
        #### 4. **Description Match**
        - **Assumption**: $P(\\text{Match}|\\text{Guilty}) = 0.90$, $P(\\text{Match}|\\text{Innocent}) = 0.10$
        - **Meaning**: Witness descriptions are fairly reliable but not perfect
        
        #### 5. **Alibi**
        - **Assumption**: $P(\\text{Alibi}|\\text{Guilty}) = 0.25$ (guilty person might fabricate)
        - **Assumption**: $P(\\text{Alibi}|\\text{Innocent}) = 0.50$ (innocent person might have verifiable alibi)
        
        ---
        
        ### Example Calculation (Default Scenario)
        
        **Evidence**: DNA Match ✓, Has Alibi ✓, Does NOT match description ✗
        
        **Likelihood Ratios:**
        - DNA: $LR = 1.0 / (5 \\times 10^{-9}) = 200,000,000$ (strongly favors guilt)
        - Alibi: $LR = 0.25 / 0.50 = 0.5$ (weakly favors innocence)
        - Description: $LR = 0.10 / 0.90 = 0.11$ (strongly favors innocence by factor of 9)
        
        **Calculation:**
        $$\\text{Posterior Odds} = \\frac{1}{200,000} \\times 200,000,000 \\times 0.5 \\times 0.11$$
        $$= \\frac{200,000,000}{200,000 \\times 2 \\times 9} = \\frac{200,000,000}{3,600,000} \\approx 55.5$$
        
        $$P(\\text{Guilty}) = \\frac{55.5}{56.5} \\approx 98.2\\%$$
        
        ---
        
        ### 🧪 Explore the Impact of Assumptions
        
        **Try these experiments:**
        
        1. **Lab Error Impact**: Move the "Lab Error Rate" slider to 0.001 (0.1%). Watch the probability of guilt plummet!
        
        2. **Prior Probability**: Edit the "Guilty" CPT to change the base rate from 1/200,000 to 1/100 (if suspect was identified through investigation, not random)
        
        3. **Evidence Reliability**: Change the Description or Alibi probabilities to see how witness reliability affects the verdict
        
        ---
        
        ### ⚖️ Legal Implications
        
        - **"Beyond Reasonable Doubt"** typically requires >95-99% certainty
        - Small changes in assumptions (especially lab error and prior probability) can drastically change verdicts
        - This model shows why **transparent probability assumptions** are crucial in forensic science
        """)
    
    # Network Visualization
    with st.expander("📊 Network Visualization", expanded=True):
        if len(model.nodes()) > 0 and model.number_of_edges() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            try:
                # Manual hierarchical layout for DNA case
                if all(node in ['Guilty', 'DNA_Match', 'Alibi', 'Desc_Match'] for node in model.nodes()):
                    pos = {
                        'Guilty': (0.5, 1.0),
                        'DNA_Match': (0.15, 0.0),
                        'Alibi': (0.5, 0.0),
                        'Desc_Match': (0.85, 0.0)
                    }
                else:
                    pos = nx.spring_layout(model, seed=42, k=1, iterations=50)
            except Exception as e:
                st.warning(f"Layout failed: {e}. Using circular layout.")
                pos = nx.circular_layout(model)
            
            if all(node in pos for node in model.nodes()):
                nx.draw_networkx_nodes(model, pos, node_color="#ff4b4b", node_size=4500, ax=ax)
                
                labels = {node: node.replace('_', '\n') for node in model.nodes()}
                nx.draw_networkx_labels(model, pos, labels=labels, font_color="white", 
                                       font_weight="bold", font_size=9, ax=ax)
                
                if model.number_of_edges() > 0:
                    try:
                        nx.draw_networkx_edges(
                            model, pos, 
                            edge_color="gray", 
                            arrows=True, 
                            arrowsize=25, 
                            arrowstyle='-|>', 
                            connectionstyle='arc3,rad=0.0',
                            width=2,
                            node_size=4500,
                            ax=ax
                        )
                    except Exception:
                        nx.draw_networkx_edges(
                            model, pos, 
                            edge_color="gray", 
                            arrows=True,
                            arrowsize=20,
                            width=2,
                            ax=ax
                        )
                
                ax.axis('off')
                ax.margins(0.2)
                st.pyplot(fig)
            else:
                st.error("Could not generate valid node positions.")
        else:
            st.warning("No nodes or edges in network.")
    
    # Network Structure Editor
    with st.expander("🔧 Edit Network Structure", expanded=False):
        st.markdown("""
        **Modify the network by adding/removing nodes and edges.**
        Warning: Changes will affect CPT requirements.
        """)
        
        st.subheader("Nodes")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            new_node = st.text_input("New node name:", key="new_node_name_input")
        with col2:
            num_states = st.number_input("States:", min_value=2, max_value=10, value=2, 
                                        key="new_node_states_input")
        
        if st.button("➕ Add Node", key="add_node_action_btn"):
            if new_node and new_node.strip():
                success, msg = add_node_to_network(model, node_states, new_node.strip(), int(num_states))
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.error("Please enter a node name")
        
        # List existing nodes
        if len(model.nodes()) > 0:
            st.markdown("**Existing Nodes:**")
            for node in model.nodes():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"• **{node}** ({len(node_states.get(node, []))} states)")
                with col2:
                    if st.button(f"Edit", key=f"edit_node_states_{node}"):
                        st.session_state[f'edit_states_{node}'] = True
                with col3:
                    if st.button(f"🗑️", key=f"remove_node_action_{node}"):
                        cpds = st.session_state['cpds']
                        success, msg = remove_node_from_network(model, cpds, node_states, node)
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                
                # Edit states if requested
                if st.session_state.get(f'edit_states_{node}', False):
                    current_states = node_states.get(node, [])
                    st.write(f"Edit states for {node}:")
                    new_states = []
                    for i, state in enumerate(current_states):
                        new_state = st.text_input(f"State {i}:", value=state, 
                                                 key=f"edit_state_{node}_{i}")
                        new_states.append(new_state)
                    
                    if st.button(f"Save states for {node}", key=f"save_node_states_{node}"):
                        st.session_state['node_states'][node] = new_states
                        st.session_state[f'edit_states_{node}'] = False
                        st.success(f"Updated states for {node}")
                        st.rerun()
        
        st.divider()
        st.subheader("Edges")
        
        if len(model.nodes()) > 1:
            col1, col2 = st.columns(2)
            with col1:
                parent_node = st.selectbox("Parent:", options=list(model.nodes()), 
                                          key="edge_parent_select")
            with col2:
                child_node = st.selectbox("Child:", options=list(model.nodes()), 
                                         key="edge_child_select")
            
            if st.button("➕ Add Edge", key="add_edge_action_btn"):
                if parent_node != child_node:
                    success, msg = add_edge_to_network(model, parent_node, child_node)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.error("Parent and child must be different nodes")
            
            # List existing edges
            if model.number_of_edges() > 0:
                st.markdown("**Existing Edges:**")
                for parent, child in model.edges():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"• **{parent}** → **{child}**")
                    with col2:
                        edge_key = f"{parent}_{child}".replace(" ", "_")
                        if st.button(f"🗑️", key=f"remove_edge_action_{edge_key}"):
                            cpds = st.session_state['cpds']
                            success, msg = remove_edge_from_network(model, cpds, parent, child)
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
    
    # Inference Engine
    st.divider()
    st.subheader("🔍 Verdict Calculator (Exact Inference)")
    
    target_var = 'Guilty'
    st.markdown(f"**Target Variable:** `{target_var}`")
    
    st.write("### Set Evidence:")
    evidence_dict = {}
    
    # Filter out target variable
    evidence_nodes = [n for n in model.nodes() if n != target_var]
    
    if evidence_nodes:
        cols = st.columns(min(3, len(evidence_nodes)))
        
        for i, node in enumerate(evidence_nodes):
            with cols[i % min(3, len(evidence_nodes))]:
                states = ["Unknown"] + node_states.get(node, [])
                # Set smart defaults
                default_idx = 0
                if node == 'DNA_Match': 
                    default_idx = states.index('Match') if 'Match' in states else 0
                if node == 'Alibi': 
                    default_idx = states.index('Yes_Alibi') if 'Yes_Alibi' in states else 0
                if node == 'Desc_Match': 
                    default_idx = states.index('No_Match') if 'No_Match' in states else 0
                
                selection = st.selectbox(f"{node} is:", options=states, index=default_idx,
                                       key=f"evidence_select_{node}")
                if selection != "Unknown":
                    evidence_dict[node] = selection
    
    if st.button("Calculate Probability of Guilt", key="calculate_inference_btn"):
        try:
            # Validate model before inference
            model.check_model()
            
            infer = VariableElimination(model)
            q = infer.query(variables=[target_var], evidence=evidence_dict)
            
            # Display Results
            target_states = node_states.get(target_var, [])
            res_df = pd.DataFrame({
                "State": target_states,
                "Probability": q.values
            })
            
            # Find guilty probability
            if 'Guilty' in target_states:
                prob_guilty = res_df.loc[res_df['State'] == 'Guilty', 'Probability'].values[0]
            else:
                prob_guilty = res_df.iloc[-1]['Probability']  # Last state
            
            c1, c2 = st.columns([1, 2])
            c1.metric("Probability of Guilt", f"{prob_guilty:.6%}")
            c2.bar_chart(res_df.set_index("State"))
            
            # Interpretation
            if prob_guilty > 0.95:
                st.error("Verdict: Highly Likely Guilty (Beyond Reasonable Doubt?)")
            elif prob_guilty < 0.05:
                st.success("Verdict: Highly Likely Innocent")
            else:
                st.warning("Verdict: Inconclusive")
            
            st.markdown("---")
            st.markdown(f"**Current Parameters:**\n- Lab Error Rate: `{lab_error_rate:.4f}`\n- Prior (Random Person): `1/200,000`")
            
        except Exception as e:
            st.error(f"Inference Failed: {e}\n\nMake sure all nodes have valid CPTs defined.")
    
    # CPT Editor
    with st.expander("📝 Advanced: Edit CPTs Manually"):
        st.markdown("""
        **Warning**: Editing CPTs will override automated updates (like the DNA lab error slider).
        Each column must sum to 1.0.
        """)
        
        # Show persistent save message
        if st.session_state.get('cpt_save_message'):
            if st.session_state['cpt_save_message'][0] == 'success':
                st.success(st.session_state['cpt_save_message'][1])
            else:
                st.error(st.session_state['cpt_save_message'][1])
            st.session_state['cpt_save_message'] = None
        
        # Format toggle
        display_format = st.radio(
            "Display Format:",
            options=["Fractions", "Decimals"],
            horizontal=True,
            key="cpt_display_format_radio",
            help="Choose how to display probabilities. You can edit in either format."
        )
        
        edit_node = st.selectbox("Select Node to Edit:", options=model.nodes(), 
                                key="cpt_edit_node_select")
        
        if edit_node:
            parents = get_node_parents(model, edit_node)
            
            # Get current CPD
            if edit_node in st.session_state['cpds']:
                cpd = st.session_state['cpds'][edit_node]
                current_df = cpd_to_dataframe(cpd, node_states[edit_node])
            else:
                current_df = generate_cpt_template(model, edit_node, node_states[edit_node], node_states)
            
            st.write(f"**Editing {edit_node}** (Parents: {parents if parents else 'None'})")
            st.write("Columns must sum to 1.0. Edit values directly in the table below.")
            
            # Store original values
            original_key = f"original_{edit_node}"
            if original_key not in st.session_state['cpt_original_values']:
                st.session_state['cpt_original_values'][original_key] = current_df.copy()
            
            # Format display
            if display_format == "Fractions":
                display_df = current_df.map(
                    lambda x: str(Fraction(x).limit_denominator(1000000000))
                )
            else:
                display_df = current_df.map(lambda x: f"{x:.10f}")
            
            # Editable table
            edited_df = st.data_editor(
                display_df, 
                key=f"cpt_data_editor_{edit_node}"
            )
            
            # Validation
            try:
                if display_format == "Fractions":
                    edited_decimal_df = edited_df.map(lambda x: float(Fraction(x)))
                else:
                    edited_decimal_df = edited_df.map(lambda x: float(x))
                
                original_df = st.session_state['cpt_original_values'][original_key]
                has_changes = not np.allclose(edited_decimal_df.values, original_df.values, rtol=1e-12)
                
                is_valid, invalid_cols = validate_cpt_columns(edited_decimal_df)
                
                if has_changes:
                    st.warning("⚠️ **You have unsaved changes!**")
                
                if not is_valid:
                    for col_name, sum_val in invalid_cols:
                        st.error(f"❌ **Column doesn't sum to 1.0:** {col_name}: {sum_val:.15f}")
                else:
                    st.success("✅ **All columns sum to 1.0**")
                    
            except Exception as e:
                st.warning(f"Unable to validate: {e}")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("💾 Save CPT", key=f"save_cpt_{edit_node}"):
                    try:
                        if display_format == "Fractions":
                            decimal_df = edited_df.map(lambda x: float(Fraction(x)))
                        else:
                            decimal_df = edited_df.map(lambda x: float(x))
                        
                        is_valid, invalid_cols = validate_cpt_columns(decimal_df)
                        
                        if not is_valid:
                            error_msg = "❌ Cannot save: Columns don't sum to exactly 1.0"
                            st.session_state['cpt_save_message'] = ('error', error_msg)
                        else:
                            success, msg = update_cpd_from_df(model, edit_node, decimal_df, 
                                                            node_states[edit_node], node_states)
                            if success:
                                st.session_state['cpds'][edit_node] = model.get_cpds(edit_node)
                                st.session_state['edited_cpds'][edit_node] = True
                                st.session_state['cpt_original_values'][original_key] = decimal_df.copy()
                                st.session_state['cpt_save_message'] = ('success', f"✅ {msg}")
                            else:
                                st.session_state['cpt_save_message'] = ('error', msg)
                        
                        st.rerun()
                    except ValueError as e:
                        st.session_state['cpt_save_message'] = ('error', f"❌ Invalid format: {e}")
                        st.rerun()
            
            with col2:
                if st.button("🔄 Reset to Default", key=f"reset_cpt_{edit_node}"):
                    if edit_node in st.session_state.get('edited_cpds', {}):
                        del st.session_state['edited_cpds'][edit_node]
                    
                    if original_key in st.session_state['cpt_original_values']:
                        del st.session_state['cpt_original_values'][original_key]
                    
                    try:
                        cpd_default = create_default_cpd(edit_node, node_states, p_match_given_innocent)
                        model.remove_cpds(edit_node)
                        model.add_cpds(cpd_default)
                        st.session_state['cpds'][edit_node] = cpd_default
                        st.session_state['cpt_save_message'] = ('success', f"🔄 Reset {edit_node} to default values!")
                        st.rerun()
                    except ValueError:
                        st.warning(f"No default CPD defined for '{edit_node}'. Cannot reset.")

if __name__ == "__main__":
    main()