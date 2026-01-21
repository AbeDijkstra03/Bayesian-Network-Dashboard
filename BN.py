import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import numpy as np

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
# Default probabilities from the Case
PRIOR_GUILTY = 1 / 200000
# DNA Random Match Probability (approx 1 in 200 million)
RMP = 1 / 200000000 

# ==========================================
# SIDEBAR: LAB ERROR ADJUSTMENT
# ==========================================
st.sidebar.title("🧬 DNA Evidence Settings")
st.sidebar.markdown("Adjust the reliability of the DNA test to see how it impacts the verdict.")

# Allow user to inject Lab Error (False Positive Probability)
# Range: 0 (Impossible) to 1% (High error rate)
lab_error_rate = st.sidebar.slider(
    "Lab Error Rate (False Positive)", 
    min_value=0.0, 
    max_value=0.01, 
    value=0.0, 
    step=0.0001,
    format="%.4f",
    help="Probability that an Innocent person matches due to lab error (contamination, swap)."
)

# Total Probability of Match | Innocent = Random Match + Lab Error
p_match_given_innocent = RMP + lab_error_rate

# ==========================================
# SESSION STATE MANAGEMENT
# ==========================================
if 'model' not in st.session_state:
    # 1. Initialize the "Crime" Network
    # Structure: Guilt causes the Evidence (DNA, Alibi, Description)
    model = BayesianNetwork([
        ('Guilty', 'DNA_Match'),
        ('Guilty', 'Alibi'),
        ('Guilty', 'Desc_Match')
    ])
    
if 'cpt_save_message' not in st.session_state:
    st.session_state['cpt_save_message'] = None
if 'cpt_original_values' not in st.session_state:
    st.session_state['cpt_original_values'] = {}
    
    # 2. Define States
    # We store this in session state so the helper functions can access it
    state_map = {
        'Guilty': ['Innocent', 'Guilty'],
        'DNA_Match': ['No_Match', 'Match'],
        'Alibi': ['No_Alibi', 'Yes_Alibi'],
        'Desc_Match': ['No_Match', 'Match']
    }
    st.session_state['node_states'] = state_map

    # 3. Define Conditional Probability Distributions (CPDs)
    
    # Node: Guilty (Prior)
    # P(Guilty) = 1/200,000
    cpd_guilty = TabularCPD(
        variable='Guilty', variable_card=2,
        values=[[1 - PRIOR_GUILTY], [PRIOR_GUILTY]],
        state_names=state_map
    )

    # Node: Alibi
    # P(Alibi|Innocent) = 0.50, P(Alibi|Guilty) = 0.25
    cpd_alibi = TabularCPD(
        variable='Alibi', variable_card=2,
        values=[
            [0.50, 0.75],  # Row 0: No_Alibi (1 - P(Yes))
            [0.50, 0.25]   # Row 1: Yes_Alibi
        ],
        evidence=['Guilty'], evidence_card=[2],
        state_names=state_map
    )

    # Node: Description
    # P(NoMatch|Innocent) = 0.90 -> P(Match|Inn) = 0.10
    # P(NoMatch|Guilty) = 0.10 -> P(Match|Gui) = 0.90
    cpd_desc = TabularCPD(
        variable='Desc_Match', variable_card=2,
        values=[
            [0.90, 0.10],  # Row 0: No_Match
            [0.10, 0.90]   # Row 1: Match
        ],
        evidence=['Guilty'], evidence_card=[2],
        state_names=state_map
    )
    
    # Node: DNA (Initial - will be updated by slider)
    # P(Match|Innocent) = RMP, P(Match|Guilty) = 1.0
    cpd_dna = TabularCPD(
        variable='DNA_Match', variable_card=2,
        values=[
            [1 - RMP, 0.0],  # Row 0: No_Match
            [RMP,     1.0]   # Row 1: Match
        ],
        evidence=['Guilty'], evidence_card=[2],
        state_names=state_map
    )

    model.add_cpds(cpd_guilty, cpd_alibi, cpd_desc, cpd_dna)
    model.check_model()
    
    st.session_state['model'] = model
    st.session_state['cpds'] = {
        'Guilty': cpd_guilty,
        'Alibi': cpd_alibi, 
        'Desc_Match': cpd_desc,
        'DNA_Match': cpd_dna
    }
    st.session_state['edited_cpds'] = {}  # Track which CPDs have been manually edited

# Helper to access model
model = st.session_state['model']
node_states = st.session_state.get('node_states', {})

# --- DYNAMIC UPDATE OF DNA CPT BASED ON SLIDER ---
# We must update the DNA CPD every rerun because the slider changes
# But only if it hasn't been manually edited
if 'DNA_Match' not in st.session_state.get('edited_cpds', {}):
    cpd_dna_new = TabularCPD(
        variable='DNA_Match', variable_card=2,
        values=[
            [1 - p_match_given_innocent, 0.0],  # Row 0: No_Match
            [p_match_given_innocent,     1.0]   # Row 1: Match
        ],
        evidence=['Guilty'], evidence_card=[2],
        state_names=node_states
    )
    model.remove_cpds('DNA_Match')
    model.add_cpds(cpd_dna_new)
    st.session_state['cpds']['DNA_Match'] = cpd_dna_new

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_node_parents(node):
    return list(model.get_parents(node))

def cpd_to_dataframe(cpd, states):
    """Convert a TabularCPD to a DataFrame for display"""
    values = cpd.values
    parents = cpd.variables[1:] if len(cpd.variables) > 1 else []
    
    if not parents:
        # No parents - just show single column
        df = pd.DataFrame(values, index=states, columns=["Probability"])
    else:
        # Has parents - create column labels from evidence combinations
        evidence_card = cpd.cardinality[1:]
        parent_states = [cpd.state_names[p] for p in parents]
        combinations = list(itertools.product(*parent_states))
        cols = []
        for c in combinations:
            if len(c) == 1:
                cols.append(str(c[0]))  # Just use the single value
            else:
                cols.append(str(c).replace("'", "").replace("(", "").replace(")", "").replace(", ", "|"))
        df = pd.DataFrame(values, index=states, columns=cols)
    
    return df

def generate_cpt_template(node, states, parent_states):
    parents = get_node_parents(node)
    if not parents:
        cols = ["Probability"]
    else:
        p_lists = [parent_states[p] for p in parents]
        combinations = list(itertools.product(*p_lists))
        cols = [str(c).replace("'", "").replace("(", "").replace(")", "").replace(", ", "|") for c in combinations]
    
    default_val = 1.0 / len(states)
    df = pd.DataFrame(default_val, index=states, columns=cols)
    return df

def update_cpd_from_df(node, df, states, parent_states):
    values = df.values.tolist()
    parents = get_node_parents(node)
    card = len(states)
    evidence = parents if parents else None
    evidence_card = [len(parent_states[p]) for p in parents] if parents else None

    try:
        # STRICT VALIDATION: Check column sums BEFORE creating CPD
        sums = df.sum(axis=0)
        tolerance = 1e-9  # Very strict tolerance (0.000000001)
        
        invalid_cols = []
        for col_name, sum_val in zip(df.columns, sums.values):
            if abs(sum_val - 1.0) > tolerance:
                invalid_cols.append((col_name, sum_val))
        
        if invalid_cols:
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
        
        # Double-check with pgmpy's validation (secondary check)
        if not cpd.is_valid_cpd():
            return False, "CPD failed pgmpy validation (this shouldn't happen after our checks)"
        
        # Update the model
        st.session_state['model'].remove_cpds(node)
        st.session_state['model'].add_cpds(cpd)
        st.session_state['cpds'][node] = cpd
        
        # Mark as manually edited
        if 'edited_cpds' not in st.session_state:
            st.session_state['edited_cpds'] = {}
        st.session_state['edited_cpds'][node] = True
        
        return True, "CPT Updated Successfully!"
        
    except Exception as e:
        return False, f"Error creating CPD: {str(e)}"

# ==========================================
# MAIN INTERFACE
# ==========================================
st.title("Bayesian Network Lab: The DNA Case ⚖️")

# ==========================================
# EDUCATIONAL SECTION: ASSUMPTIONS
# ==========================================
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

# --- VISUALIZATION ---
with st.expander("📊 Network Visualization", expanded=True):
    if len(model.nodes()) > 0 and model.number_of_edges() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use a more reliable layout with error handling
        try:
            # Manual hierarchical layout for this specific network structure
            pos = {
                'Guilty': (0.5, 1.0),
                'DNA_Match': (0.15, 0.0),
                'Alibi': (0.5, 0.0),
                'Desc_Match': (0.85, 0.0)
            }
            # Fallback to spring layout if nodes don't match
            if not all(node in pos for node in model.nodes()):
                pos = nx.spring_layout(model, seed=42, k=1, iterations=50)
        except Exception as e:
            st.warning(f"Layout failed: {e}. Using circular layout.")
            pos = nx.circular_layout(model)
        
        # Verify all nodes have positions
        if all(node in pos for node in model.nodes()):
            # Draw nodes with larger size
            nx.draw_networkx_nodes(model, pos, node_color="#ff4b4b", node_size=4500, ax=ax)
            
            # Draw labels with smaller font and line breaks for long names
            labels = {node: node.replace('_', '\n') for node in model.nodes()}
            nx.draw_networkx_labels(model, pos, labels=labels, font_color="white", 
                                   font_weight="bold", font_size=9, ax=ax)
            
            # Draw edges with proper arrows
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
                        node_size=4500,  # Must match node size for proper arrow placement
                        ax=ax
                    )
                except Exception as e:
                    # Fallback: draw with simpler arrow style
                    st.warning(f"Arrow rendering issue, using fallback: {e}")
                    nx.draw_networkx_edges(
                        model, pos, 
                        edge_color="gray", 
                        arrows=True,
                        arrowsize=20,
                        width=2,
                        ax=ax
                    )
            
            ax.axis('off')
            ax.margins(0.2)  # Add margin so arrows aren't cut off
            st.pyplot(fig)
        else:
            st.error("Could not generate valid node positions.")
    else:
        st.warning("No nodes or edges in network.")

# --- INFERENCE ENGINE ---
st.divider()
st.subheader("🔍 Verdict Calculator (Exact Inference)")

# Setup Inference
target_var = 'Guilty' # Hardcode target for this case
st.markdown(f"**Target Variable:** `{target_var}`")

# Evidence Selectors
st.write("### Set Evidence:")
evidence_dict = {}
cols = st.columns(3)

# Filter out the target variable so we only show evidence inputs
evidence_nodes = [n for n in model.nodes() if n != target_var]

for i, node in enumerate(evidence_nodes):
    with cols[i % 3]:
        states = ["Unknown"] + node_states[node]
        # Set smart defaults based on your prompt's scenario
        default_idx = 0
        if node == 'DNA_Match': default_idx = 2 # Match
        if node == 'Alibi': default_idx = 2     # Yes_Alibi
        if node == 'Desc_Match': default_idx = 1 # No_Match
        
        selection = st.selectbox(f"{node} is:", options=states, index=default_idx)
        if selection != "Unknown":
            evidence_dict[node] = selection

if st.button("Calculate Probability of Guilt"):
    try:
        infer = VariableElimination(model)
        q = infer.query(variables=[target_var], evidence=evidence_dict)
        
        # Display Results
        res_df = pd.DataFrame({
            "State": node_states[target_var],
            "Probability": q.values
        })
        
        # Highlight the "Guilty" probability
        prob_guilty = res_df.loc[res_df['State'] == 'Guilty', 'Probability'].values[0]
        
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
        st.error(f"Inference Failed: {e}")

# --- CPT EDITOR (Hidden by default to keep UI clean) ---
with st.expander("📝 Advanced: Edit CPTs Manually"):
    st.markdown("""
    **Warning**: Editing CPTs will override automated updates (like the DNA lab error slider).
    Each column must sum to 1.0.
    """)
    
    # Show persistent save message if exists
    if st.session_state.get('cpt_save_message'):
        if st.session_state['cpt_save_message'][0] == 'success':
            st.success(st.session_state['cpt_save_message'][1])
        else:
            st.error(st.session_state['cpt_save_message'][1])
        # Clear message after displaying
        st.session_state['cpt_save_message'] = None
    
    # Add format toggle
    display_format = st.radio(
        "Display Format:",
        options=["Fractions", "Decimals"],
        horizontal=True,
        help="Choose how to display probabilities. You can edit in either format."
    )
    
    edit_node = st.selectbox("Select Node to Edit:", options=model.nodes())
    if edit_node:
        parents = get_node_parents(edit_node)
        
        # Get current CPD and convert to DataFrame
        if edit_node in st.session_state['cpds']:
            cpd = st.session_state['cpds'][edit_node]
            current_df = cpd_to_dataframe(cpd, node_states[edit_node])
        else:
            current_df = generate_cpt_template(edit_node, node_states[edit_node], node_states)
        
        st.write(f"**Editing {edit_node}** (Parents: {parents if parents else 'None'})")
        st.write("Columns must sum to 1.0. Edit values directly in the table below.")
        
        # Store original values for change detection (use a key specific to the node)
        original_key = f"original_{edit_node}"
        if original_key not in st.session_state['cpt_original_values']:
            st.session_state['cpt_original_values'][original_key] = current_df.copy()
        
        # Format display based on selection
        if display_format == "Fractions":
            from fractions import Fraction
            # Convert to fractions for display
            display_df = current_df.map(
                lambda x: str(Fraction(x).limit_denominator(1000000000))
            )
        else:  # Decimals
            # Show high precision to avoid rounding confusion
            display_df = current_df.map(lambda x: f"{x:.10f}")
        
        # Editable table
        edited_df = st.data_editor(
            display_df, 
            key=f"editor_{edit_node}"
        )
        
        # Check for unsaved changes AND validate column sums automatically
        try:
            if display_format == "Fractions":
                from fractions import Fraction
                edited_decimal_df = edited_df.map(lambda x: float(Fraction(x)))
            else:
                edited_decimal_df = edited_df.map(lambda x: float(x))
            
            original_df = st.session_state['cpt_original_values'][original_key]
            
            # Compare with small tolerance for floating point comparison
            has_changes = not np.allclose(edited_decimal_df.values, original_df.values, rtol=1e-12)
            
            # Check column sums automatically
            sums = edited_decimal_df.sum(axis=0)
            tolerance = 1e-9
            
            invalid_cols = []
            for col_name, sum_val in zip(edited_decimal_df.columns, sums.values):
                if abs(sum_val - 1.0) > tolerance:
                    invalid_cols.append((col_name, sum_val))
            
            # Show appropriate messages
            if has_changes:
                st.warning("⚠️ **You have unsaved changes!**")
            
            if invalid_cols:
                for col_name, sum_val in invalid_cols:
                    error = sum_val - 1.0
                    st.error(f"❌ **Columns don't sum to 1.0:** \n\n {col_name}: {sum_val:.15f}")
            else:
                st.success("✅ **All columns sum to 1.0**")
                
        except Exception as e:
            st.warning(f"Unable to validate: {e}")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("💾 Save CPT", key=f"save_{edit_node}"):
                # Convert back to decimals if needed
                try:
                    if display_format == "Fractions":
                        from fractions import Fraction
                        # Parse fractions back to decimals
                        decimal_df = edited_df.map(lambda x: float(Fraction(x)))
                    else:
                        # Parse decimal strings back to floats
                        decimal_df = edited_df.map(lambda x: float(x))
                    
                    # STRICT validation with detailed error messages
                    sums = decimal_df.sum(axis=0)
                    tolerance = 1e-9  # Very strict: 0.000000001
                    
                    invalid_cols = []
                    for col_name, sum_val in zip(decimal_df.columns, sums.values):
                        if abs(sum_val - 1.0) > tolerance:
                            invalid_cols.append((col_name, sum_val))
                    
                    if invalid_cols:
                        error_msg = "❌ Cannot save: Columns don't sum to exactly 1.0"
                        st.session_state['cpt_save_message'] = ('error', error_msg)
                    else:
                        success, msg = update_cpd_from_df(edit_node, decimal_df, node_states[edit_node], node_states)
                        if success: 
                            # Update the original values to the new saved values
                            st.session_state['cpt_original_values'][original_key] = decimal_df.copy()
                            st.session_state['cpt_save_message'] = ('success', f"✅ {msg}")
                            st.rerun()
                        else: 
                            st.session_state['cpt_save_message'] = ('error', msg)
                except ValueError as e:
                    st.session_state['cpt_save_message'] = ('error', f"❌ Invalid format. Please enter valid numbers or fractions: {e}")
                
                # Rerun to show the message
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset to Default", key=f"reset_{edit_node}"):
                # Remove from edited_cpds
                if edit_node in st.session_state.get('edited_cpds', {}):
                    del st.session_state['edited_cpds'][edit_node]
                
                # Clear the original values so it resets on next load
                if original_key in st.session_state['cpt_original_values']:
                    del st.session_state['cpt_original_values'][original_key]
                
                # Recreate the default CPD based on the node
                if edit_node == 'Guilty':
                    cpd_default = TabularCPD(
                        variable='Guilty', variable_card=2,
                        values=[[1 - PRIOR_GUILTY], [PRIOR_GUILTY]],
                        state_names=node_states
                    )
                elif edit_node == 'Alibi':
                    cpd_default = TabularCPD(
                        variable='Alibi', variable_card=2,
                        values=[
                            [0.50, 0.75],
                            [0.50, 0.25]
                        ],
                        evidence=['Guilty'], evidence_card=[2],
                        state_names=node_states
                    )
                elif edit_node == 'Desc_Match':
                    cpd_default = TabularCPD(
                        variable='Desc_Match', variable_card=2,
                        values=[
                            [0.90, 0.10],
                            [0.10, 0.90]
                        ],
                        evidence=['Guilty'], evidence_card=[2],
                        state_names=node_states
                    )
                elif edit_node == 'DNA_Match':
                    cpd_default = TabularCPD(
                        variable='DNA_Match', variable_card=2,
                        values=[
                            [1 - p_match_given_innocent, 0.0],
                            [p_match_given_innocent, 1.0]
                        ],
                        evidence=['Guilty'], evidence_card=[2],
                        state_names=node_states
                    )
                
                # Update the model with default CPD
                model.remove_cpds(edit_node)
                model.add_cpds(cpd_default)
                st.session_state['cpds'][edit_node] = cpd_default
                
                st.session_state['cpt_save_message'] = ('success', f"🔄 Reset {edit_node} to default values!")
                st.rerun()