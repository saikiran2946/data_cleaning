import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from backend.agents.root_agent import RootAgent
import backend.agents.value_standardization_agent as value_standardization_module
import backend.agents.feature_generation_agent as feature_generation_module
from utils.openai_client import llm
import backend.agents.duplicate_agent as duplicate_agent_module
import re
from pandas.api.types import CategoricalDtype

# Add a helper to ensure Arrow compatibility before displaying DataFrames
def ensure_arrow_compatible(df):
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype).startswith('category'):
            df[col] = df[col].astype(str)
    return df

# Place safe_fillna_unknown near the top of the file, after imports and before main()
def safe_fillna_unknown(df):
    """
    For each column:
    - If categorical and contains 'Unknown' and any category is numeric, convert the column to string.
    - If object and contains both numbers and strings, convert all to string.
    - If numeric, replace 'Unknown' with median and convert to numeric.
    - If categorical and all categories are strings, ensure 'Unknown' is a category.
    """
    for col in df.columns:
        # If column contains 'Unknown'
        if (df[col] == 'Unknown').any():
            # Numeric columns: replace 'Unknown' with median and convert to numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                median = df[col].median()
                df[col] = df[col].replace('Unknown', median)
            # Categorical columns
            elif isinstance(df[col].dtype, CategoricalDtype):
                cats = df[col].cat.categories
                # If any category is numeric, convert the whole column to string
                if any(isinstance(x, (int, float, np.integer, np.floating)) for x in cats):
                    df[col] = df[col].astype(str)
                else:
                    # If categories are all strings, ensure 'Unknown' is a category
                    if 'Unknown' not in df[col].cat.categories:
                        df[col] = df[col].cat.add_categories(['Unknown'])
            # Object columns: if mixed types, convert all to string
            elif pd.api.types.is_object_dtype(df[col]):
                # If both numbers and strings, convert all to string
                if df[col].apply(lambda x: isinstance(x, (int, float, np.integer, np.floating))).any():
                    df[col] = df[col].astype(str)
    return df

# --- Load Environment Variables ---
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB = os.getenv("MONGODB_DB", "agent_memory")
MEMORY_STORAGE_TYPE = os.getenv("MEMORY_STORAGE_TYPE", "file")

# --- Streamlit Page Config & Styling ---
st.set_page_config(page_title="AI Data Cleaner", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .stApp { background: linear-gradient(120deg, #f8fafc 0%, #e2e8f0 100%); color: #222; }
    section[data-testid="stSidebar"] { background: linear-gradient(120deg, #f1f5f9 0%, #e2e8f0 100%); border-radius: 0 20px 20px 0; }
    .profile-column-label { font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #4F8BF9;'>AI Data Cleaning Agent</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Effortlessly clean, standardize, and enhance your datasets with AI-powered agents.</p>", unsafe_allow_html=True)

def execute_code(df, code):
    """Executes a code string on a DataFrame copy and returns the result."""
    df_copy = df.copy()
    try:
        if code and 'from scipy.stats import winsorize' in code:
            code = code.replace('from scipy.stats import winsorize', 'from scipy.stats.mstats import winsorize')
        exec(code, {'df': df_copy, 'pd': pd, 'np': np})
        return df_copy
    except Exception as e:
        st.error(f"Error executing code: {e}")
        return df

def find_actual_column_name(df, col_name):
    """Find the actual column name in the DataFrame (case-insensitive)."""
    for actual_col in df.columns:
        if actual_col.lower() == col_name.lower():
            return actual_col
        if actual_col.strip().lower() == col_name.strip().lower():
            return actual_col
    return col_name if col_name in df.columns else None

# =========================
# UI Rendering Functions
# =========================
def display_ui_for_agent(agent):
    """Display the UI for the given agent based on its type."""
    agent_name = agent.__class__.__name__
    ui_function_map = {
        "DataTypeAgent": display_data_type_ui,
        "MissingValueAgent": display_missing_value_ui,
        "DuplicateAgent": display_duplicate_ui,
        "OutlierAgent": display_outlier_ui,
        "NormalizationAgent": display_normalization_ui,
        "ValueStandardizationAgent": display_value_standardization_ui,
        "FeatureGenerationAgent": display_feature_generation_ui,
        "ValidatingAgent": display_validation_results,
    }
    ui_function = ui_function_map.get(agent_name)
    if ui_function:
        ui_function(agent)

def display_data_type_ui(agent):
    """Display the UI for data type suggestions and user selection."""
    step = st.session_state.get("current_step", 0)
    cache_key = f"actions_DataTypeAgent_{step}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = agent.generate_actions()
    actions = st.session_state[cache_key]
    if not actions: return st.info("No data type suggestions available.")
    for col_action in actions:
        col_name = col_action.get('name', 'Unknown')
        dtype = col_action.get('dtype', 'N/A')
        reason = col_action.get('reason', '')
        expander_title = f"**{col_name}** - {reason}"
        with st.expander(expander_title, expanded=True):
            options = ["skip", "int64", "float64", "datetime64[ns]", "category", "string"]
            suggestion = col_action.get("suggested_dtype", "skip")
            idx = options.index(suggestion) if suggestion in options else 0
            user_dtype = st.selectbox("New data type:", options, index=idx, key=f"dtype_{col_action['name']}_{step}")
            st.session_state.user_choices[col_action['name']] = {"agent": agent, "choice": {"suggested_action": user_dtype, "reason": "User selected data type."}}

def display_missing_value_ui(agent):
    """Display the UI for missing value handling and user selection."""
    step = st.session_state.get("current_step", 0)
    cache_key = f"actions_MissingValueAgent_{step}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = agent.generate_actions()
    actions = st.session_state[cache_key]
    if not actions: return st.info("No missing values found.")
    for col_action in actions:
        col_name = col_action.get('name', 'Unknown')
        reason = col_action.get('reason', '')
        expander_title = f"**{col_name}** - {reason}"
        with st.expander(expander_title, expanded=True):
            options = ["skip", "drop_rows_with_missing_values", "drop_column", "fillna_mean", "fillna_median", "fillna_mode", "fillna_constant"]
            suggestion = col_action.get("suggested_action", "skip")
            idx = options.index(suggestion) if suggestion in options else 0
            user_action = st.radio("Action:", options, index=idx, key=f"mv_action_{col_action['name']}_{step}")
            choice = {"suggested_action": user_action, "reason": "User selected missing value treatment."}
            if user_action == "fillna_constant":
                # Pre-fill with agent suggestion if available
                suggested_val = col_action.get("constant_value", "")
                val = st.text_input("Constant value:", value=suggested_val, key=f"mv_const_{col_action['name']}_{step}")
                choice["constant_value"] = val
            st.session_state.user_choices[col_action['name']] = {"agent": agent, "choice": choice}

def display_duplicate_ui(agent):
    """Display the UI for duplicate row handling and user selection."""
    step = st.session_state.get("current_step", 0)
    cache_key = f"actions_DuplicateAgent_{step}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = agent.generate_actions()
    actions = st.session_state[cache_key]
    if not actions or not actions[0].get("suggested_action"): return st.info("No duplicate rows found.")
    action = actions[0]
    col_name = action.get('name', 'Unknown')
    reason = action.get('reason', '')
    expander_title = f"**{col_name}** - {reason}"
    with st.expander(expander_title, expanded=True):
        options = ["skip", "drop_duplicates"]
        suggestion = action.get("suggested_action", "skip")
        idx = options.index(suggestion) if suggestion in options else 0
        user_action = st.radio("Action:", options, index=idx, key=f"dup_action_{step}")
        choice = dict(action)
        choice["suggested_action"] = user_action
        st.session_state.user_choices["duplicates"] = {"agent": agent, "choice": choice}

def display_outlier_ui(agent):
    """Display the UI for outlier detection and user selection."""
    step = st.session_state.get("current_step", 0)
    cache_key = f"actions_OutlierAgent_{step}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = agent.generate_actions()
    actions = st.session_state[cache_key]
    if not actions: return st.info("No numeric columns for outlier detection.")
    for col_action in actions:
        col_name = col_action.get('name', 'Unknown')
        reason = col_action.get('reason', '')
        expander_title = f"**{col_name}** - {reason}"
        with st.expander(expander_title, expanded=True):
            options = ["skip", "clip_to_bounds", "winsorize", "remove_outliers", "flag_outliers"]
            suggestion = col_action.get("suggested_action", "skip")
            idx = options.index(suggestion) if suggestion in options else 0
            user_action = st.radio("Action:", options, index=idx, key=f"outlier_{col_action['name']}_{step}")
            choice = dict(col_action)
            choice["suggested_action"] = user_action
            st.session_state.user_choices[col_action['name']] = {"agent": agent, "choice": choice}

def display_normalization_ui(agent):
    """Display the UI for normalization strategy selection."""
    step = st.session_state.get("current_step", 0)
    cache_key = f"actions_NormalizationAgent_{step}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = agent.generate_actions()
    actions = st.session_state[cache_key]
    if not actions: return st.info("No columns suitable for normalization.")
    for col_action in actions:
        col_name = col_action.get('name', 'Unknown')
        reason = col_action.get('reason', '')
        expander_title = f"**{col_name}** - {reason}"
        with st.expander(expander_title, expanded=True):
            st.markdown(f"**Column: {col_name}** - {reason}", unsafe_allow_html=True)
            options = ["skip", "StandardScaler", "MinMaxScaler", "Log-Transform"]
            suggestion = col_action.get("suggested_strategy", "skip")
            idx = options.index(suggestion) if suggestion in options else 0
            user_strategy = st.radio("Strategy:", options, index=idx, key=f"norm_{col_action['name']}_{step}")
            choice = dict(col_action)
            choice["suggested_strategy"] = user_strategy
            st.session_state.user_choices[col_action['name']] = {"agent": agent, "choice": choice}

def display_value_standardization_ui(agent):
    """Display the UI for value standardization and mapping selection."""
    step = st.session_state.get("current_step", 0)
    cache_key = f"actions_ValueStandardizationAgent_{step}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = agent.generate_actions()
    actions = st.session_state[cache_key]
    if not actions: 
        st.info("No value standardization needed.")
        return
    st.write(f"Value standardization found {len(actions)} columns to process")
    for col_action in actions:
        column_name = col_action.get('name', 'Unknown')
        reason = col_action.get('reason', '')
        mappings = col_action.get("mappings", [])
        expander_title = f"**{column_name}** - {reason}"
        with st.expander(expander_title, expanded=True):
            st.write("**Current unique values in this column:**")
            if column_name in st.session_state.df.columns:
                unique_vals = st.session_state.df[column_name].dropna().unique()
                st.write(f"Found {len(unique_vals)} unique values: {unique_vals[:10].tolist()}")
                if len(unique_vals) > 10:
                    st.write(f"... and {len(unique_vals) - 10} more values")
            else:
                st.warning(f"Column '{column_name}' not found in current dataset")
            st.write("**Suggested Mappings:**")
            st.json(mappings)
            st.session_state.user_choices[column_name] = {"agent": agent, "choice": col_action}

def display_feature_generation_ui(agent):
    """Display the UI for feature generation suggestions and user selection."""
    step = st.session_state.get("current_step", 0)
    cache_key = f"actions_FeatureGenerationAgent_{step}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = agent.generate_actions()
    actions = st.session_state[cache_key]
    if not actions: 
        st.info("No new features suggested.")
        return
    st.write(f"Feature generation found {len(actions)} features to create")
    for feature in actions:
        feature_name = feature.get('name', 'unnamed_feature')
        reason = feature.get('reason', 'No reason provided')
        formula = feature.get('formula', '')
        expander_title = f"**{feature_name}**"
        with st.expander(expander_title, expanded=True):
            st.write("**Formula:**")
            st.code(formula, language="python")
            st.write("**Reason:**", reason)
            st.session_state.user_choices[feature_name] = {"agent": agent, "choice": feature}

def display_validation_results(agent):
    """Display the UI for final data validation results."""
    step = st.session_state.get("current_step", 0)
    cache_key = f"actions_ValidatingAgent_{step}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = agent.generate_actions()
    results = st.session_state[cache_key]
    if not results: return st.error("Validation agent failed to produce a result.")
    result = results[0]
    st.subheader("Final Validation")
    st.json(result)
    if result.get("status") == "completed":
        st.success("Data validation passed!")
    else:
        st.warning("Data validation found issues.")

def initialize_session_state():
    """Initialize Streamlit session state variables if not already set."""
    if "df" not in st.session_state: st.session_state.df = None
    if "root_agent" not in st.session_state: st.session_state.root_agent = None
    if "user_choices" not in st.session_state: st.session_state.user_choices = {}
    if "cleaning_histories" not in st.session_state:
        st.session_state.cleaning_histories = {}
    if "current_step" not in st.session_state: st.session_state.current_step = 0
    if "cleaning_logs" not in st.session_state: st.session_state.cleaning_logs = []
    if "cleaning_plan" not in st.session_state: st.session_state.cleaning_plan = None
    if "cleaning_history" not in st.session_state: st.session_state.cleaning_history = []

def parse_data_dictionary(file):
    """Parse a data dictionary file (CSV or Excel) and return it as a DataFrame."""
    if file is None:
        return None
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Unsupported data dictionary file type.")
        return None

def generate_column_description(llm, col, dtype, sample_values):
    """Generate a column description using the LLM based on column name, type, and sample values."""
    prompt = f"""
You are a data analyst. Given the column name, data type, and sample values, write a short, clear description of what this column likely represents in business or domain terms.
Column name: {col}
Data type: {dtype}
Sample values: {sample_values}
Description:
"""
    response = llm.invoke(prompt)
    return response.content.strip()

def generate_data_dictionary(df, llm):
    """Generate a data dictionary DataFrame for the given DataFrame using the LLM."""
    data = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_values = df[col].dropna().unique()[:5].tolist()
        description = generate_column_description(llm, col, dtype, sample_values)
        unique_vals = df[col].dropna().unique()
        allowed_values = unique_vals.tolist() if (dtype in ["object", "category"] and len(unique_vals) <= 10) else ""
        nullable = "Yes" if df[col].isnull().any() else "No"
        is_primary_key = "Yes" if df[col].is_unique and not df[col].isnull().any() else "No"
        constraints = []
        if df[col].is_unique:
            constraints.append("No duplicates")
        if np.issubdtype(df[col].dtype, np.number):
            if (df[col].dropna() > 0).all():
                constraints.append("Must be > 0")
        constraints_str = "; ".join(constraints) if constraints else ""
        data.append({
            "Column Name": col,
            "Data Type": dtype,
            "Description": description,
            "Allowed Values": ", ".join(map(str, allowed_values)) if isinstance(allowed_values, list) else allowed_values,
            "Example Values": ", ".join(map(str, sample_values)),
            "Nullable": nullable,
            "Default Value": "",
            "Is Primary Key": is_primary_key,
            "Foreign Key Reference": "",
            "Units": "",
            "Source": "",
            "Constraints/Rules": constraints_str
        })
    return pd.DataFrame(data)

def apply_user_notes_to_data_dict(data_dict, user_notes):
    """Apply user notes to the data dictionary DataFrame."""
    col_names = [str(col).lower() for col in data_dict["Column Name"]]
    # Helper to add unmatched notes to User Notes
    def add_note_to_all(note):
        """Add a note to the 'User Notes' column for all columns in the data dictionary."""
        if "User Notes" not in data_dict.columns:
            data_dict["User Notes"] = ""
        data_dict["User Notes"] = [((val + "\n") if val else "") + note for val in data_dict["User Notes"]]

    # Primary Key
    pk_matches = re.findall(r'primary key is ([\w_]+)', user_notes, re.IGNORECASE)
    pk_matches += re.findall(r'([\w_]+) is the primary key', user_notes, re.IGNORECASE)
    pk_col_found = None
    for pk_col in pk_matches:
        if pk_col.lower() in col_names:
            pk_col_found = pk_col.lower()
            break
    if pk_matches:
        if pk_col_found:
            data_dict["Is Primary Key"] = [
                "Yes" if str(col).lower() == pk_col_found else "No"
                for col in data_dict["Column Name"]
            ]
        else:
            add_note_to_all(f"Primary key instruction: {pk_matches[0]}")

    # Nullable
    nullable_matches = re.findall(r'([\w_]+) is nullable', user_notes, re.IGNORECASE)
    for colname in nullable_matches:
        if colname.lower() in col_names:
            data_dict["Nullable"] = [
                "Yes" if str(col).lower() == colname.lower() else val
                for col, val in zip(data_dict["Column Name"], data_dict["Nullable"])
            ]
        else:
            add_note_to_all(f"Nullable instruction: {colname} is nullable")
    not_nullable_matches = re.findall(r'([\w_]+) is not nullable', user_notes, re.IGNORECASE)
    for colname in not_nullable_matches:
        if colname.lower() in col_names:
            data_dict["Nullable"] = [
                "No" if str(col).lower() == colname.lower() else val
                for col, val in zip(data_dict["Column Name"], data_dict["Nullable"])
            ]
        else:
            add_note_to_all(f"Nullable instruction: {colname} is not nullable")
    # Units
    units_matches = re.findall(r'units? for ([\w_]+) (?:is|are) ([\w%$]+)', user_notes, re.IGNORECASE)
    for colname, unit in units_matches:
        if colname.lower() in col_names:
            data_dict["Units"] = [
                unit if str(col).lower() == colname.lower() else val
                for col, val in zip(data_dict["Column Name"], data_dict["Units"])
            ]
        else:
            add_note_to_all(f"Units instruction: units for {colname} are {unit}")
    # Default Value
    default_matches = re.findall(r'default value for ([\w_]+) is ([^.,;\n]+)', user_notes, re.IGNORECASE)
    for colname, default_val in default_matches:
        if colname.lower() in col_names:
            data_dict["Default Value"] = [
                default_val.strip() if str(col).lower() == colname.lower() else val
                for col, val in zip(data_dict["Column Name"], data_dict["Default Value"])
            ]
        else:
            add_note_to_all(f"Default value instruction: default value for {colname} is {default_val}")
    # Constraints/Rules
    constraint_matches = re.findall(r'([\w_]+) must be ([^.,;\n]+)', user_notes, re.IGNORECASE)
    for colname, rule in constraint_matches:
        if colname.lower() in col_names:
            data_dict["Constraints/Rules"] = [
                (val + "; " if val else "") + rule.strip() if str(col).lower() == colname.lower() else val
                for col, val in zip(data_dict["Column Name"], data_dict["Constraints/Rules"])
            ]
        else:
            add_note_to_all(f"Constraint instruction: {colname} must be {rule}")
    # Foreign Key
    fk_matches = re.findall(r'([\w_]+) references ([\w_]+)\.([\w_]+)', user_notes, re.IGNORECASE)
    for colname, table, field in fk_matches:
        if colname.lower() in col_names:
            data_dict["Foreign Key Reference"] = [
                f"{table}.{field}" if str(col).lower() == colname.lower() else val
                for col, val in zip(data_dict["Column Name"], data_dict["Foreign Key Reference"])
            ]
        else:
            add_note_to_all(f"Foreign key instruction: {colname} references {table}.{field}")
    fk_matches2 = re.findall(r'foreign key for ([\w_]+) is ([\w_]+)\.([\w_]+)', user_notes, re.IGNORECASE)
    for colname, table, field in fk_matches2:
        if colname.lower() in col_names:
            data_dict["Foreign Key Reference"] = [
                f"{table}.{field}" if str(col).lower() == colname.lower() else val
                for col, val in zip(data_dict["Column Name"], data_dict["Foreign Key Reference"])
            ]
        else:
            add_note_to_all(f"Foreign key instruction: foreign key for {colname} is {table}.{field}")
    return data_dict

# =========================
# Main App Logic
# =========================
def main():
    """Main entry point for the Streamlit app."""
    initialize_session_state()

    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

        # Only show data dictionary options if not already set
        if "data_dictionary" not in st.session_state or st.session_state.data_dictionary is None:
            upload_dict = st.radio(
                "Do you want to upload a data dictionary?",
                ("No, generate automatically", "Yes, I want to upload"),
                index=None,
                key="data_dict_choice"
            )

            # Only proceed if a choice has been made
            if upload_dict is not None:
                data_dict_file = None
                user_notes_dict = {}
                if upload_dict == "Yes, I want to upload":
                    data_dict_file = st.file_uploader("Upload Data Dictionary (CSV or Excel)", type=["csv", "xlsx"], key="data_dict_file")
                    if data_dict_file is not None:
                        temp_data_dict = parse_data_dictionary(data_dict_file)
                        st.session_state.data_dictionary = temp_data_dict
                        st.session_state.user_uploaded_data_dictionary = temp_data_dict.copy()
                
                # After DataFrame is loaded, if generating data dictionary, ask for general user notes
                if upload_dict == "No, generate automatically" and "df" in st.session_state and st.session_state.df is not None:
                    st.subheader("Add any general notes or points for the data dictionary (optional)")
                    general_notes = st.text_area("General Notes", value="", key="general_notes_box")
                    if st.button("Generate Data Dictionary"):
                        temp_data_dict = generate_data_dictionary(st.session_state.df, llm)
                        temp_data_dict["User Notes"] = general_notes
                        temp_data_dict = apply_user_notes_to_data_dict(temp_data_dict, general_notes)
                        st.session_state.data_dictionary = temp_data_dict
        # If data dictionary is set, do not show the above UI again

    if "uploaded_file" in st.session_state and st.session_state.uploaded_file is not None:
        if "df" not in st.session_state or st.session_state.df is None:
            try:
                df = pd.read_csv(st.session_state.uploaded_file) if st.session_state.uploaded_file.name.endswith('.csv') else pd.read_excel(st.session_state.uploaded_file)
                st.session_state.df = df
                # Initialize RootAgent with memory
                st.session_state.root_agent = RootAgent(df, st.session_state.get("data_dictionary"))
                st.session_state.user_choices = {}
                st.session_state.current_step = 0
                st.session_state.cleaning_logs = []
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return

    # Generate data dictionary if not provided
    if "df" in st.session_state and st.session_state.df is not None:
        if "data_dictionary" not in st.session_state or st.session_state.data_dictionary is None:
            st.session_state.data_dictionary = generate_data_dictionary(st.session_state.df, llm)

    if st.session_state.df is not None:
        # --- Add Profile Tab ---
        tabs = st.tabs(["Profile", "Cleaning Workflow"])
        with tabs[0]:
            st.subheader("Dataset Profile Information")
            st.session_state.df = safe_fillna_unknown(st.session_state.df)
            df = st.session_state.df
            st.write("**Shape:**", df.shape)
            st.write("**Columns:**", list(df.columns))
            st.write("**Data Types:**")
            st.dataframe(df.dtypes.astype(str).reset_index().rename(columns={0: 'dtype', 'index': 'column'}))
            st.write("**Missing Values (per column):**")
            st.dataframe(df.isnull().sum().reset_index().rename(columns={0: 'missing', 'index': 'column'}))
            st.write("**Unique Values (per column):**")
            st.dataframe(df.nunique().reset_index().rename(columns={0: 'unique', 'index': 'column'}))
            # Check if there are any numerical columns before displaying statistics
            numerical_df = df.select_dtypes(include=np.number)
            if not numerical_df.empty:
                st.write("**Basic Statistics (numeric columns):**")
                st.dataframe(numerical_df.describe().T)
            else:
                st.info("No numerical columns found in the dataset.")
            # Display both user-provided and generated data dictionaries if available
            if "data_dictionary" in st.session_state and st.session_state.data_dictionary is not None:
                st.subheader("Data Dictionary in Use")
                try:
                    # Convert data dictionary to a displayable format
                    display_dict = st.session_state.data_dictionary.copy()
                    
                    # Convert any problematic columns to strings
                    for col in display_dict.columns:
                        if display_dict[col].dtype == 'object':
                            display_dict[col] = display_dict[col].astype(str)
                    
                    st.dataframe(display_dict)
                except Exception as e:
                    st.warning(f"Could not display data dictionary due to serialization error: {e}")
                    st.write("Data dictionary exists but cannot be displayed in table format.")
            
            if "data_dictionary" in st.session_state and hasattr(st.session_state, 'user_uploaded_data_dictionary') and st.session_state.user_uploaded_data_dictionary is not None:
                st.subheader("User-Provided Data Dictionary")
                try:
                    # Convert data dictionary to a displayable format
                    display_dict = st.session_state.user_uploaded_data_dictionary.copy()
                    
                    # Convert any problematic columns to strings
                    for col in display_dict.columns:
                        if display_dict[col].dtype == 'object':
                            display_dict[col] = display_dict[col].astype(str)
                    
                    st.dataframe(display_dict)
                except Exception as e:
                    st.warning(f"Could not display user data dictionary due to serialization error: {e}")
                    st.write("User data dictionary exists but cannot be displayed in table format.")
            
            if "data_dictionary" in st.session_state and st.session_state.data_dictionary is not None:
                st.subheader("Column Profiles")
                display_dict = st.session_state.data_dictionary.copy()
                for idx, row in display_dict.iterrows():
                    col_name = row.get("Column Name", "")
                    dtype = row.get("Data Type", "")
                    desc = row.get("Description", "")
                    st.markdown(f"**{col_name}** (Current: **{dtype}**) - {desc}", unsafe_allow_html=True)
        
        with tabs[1]:
            root_agent = st.session_state.root_agent
            # --- Always use cached cleaning plan ---
            cleaning_plan = st.session_state.get("cleaning_plan")
            if cleaning_plan is None:
                cleaning_plan = root_agent.get_cleaning_plan()
                st.session_state.cleaning_plan = cleaning_plan
            total_steps = len(cleaning_plan)
            step = st.session_state.get("current_step", 0)

            # --- Add live preview of current data ---
            st.subheader("Current Data Preview")
            st.dataframe(ensure_arrow_compatible(st.session_state.df.head(10)))

            # If all steps are done, show final results
            if step >= total_steps:
                st.success("All cleaning steps completed!")
                st.write("### Cleaned Dataset")
                st.dataframe(ensure_arrow_compatible(st.session_state.df))
                st.write("### Full Cleaning Log (JSON)")
                st.json(st.session_state.cleaning_logs)
                
                # Add download button for cleaned dataset
                csv = st.session_state.df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Cleaned Dataset as CSV",
                    data=csv,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv"
                )
                return

            step_info = cleaning_plan[step]
            agent_name = step_info["agent_name"]
            st.header(f"Step {step+1} of {total_steps}: {agent_name}")
            st.write(step_info["reason"])

            agent = root_agent.get_agent(agent_name)
            if agent:
                display_ui_for_agent(agent)
            else:
                st.error(f"Could not find agent: {agent_name}")

            # Show cleaning log for this step
            st.subheader("Cleaning Log (JSON)")
            if len(st.session_state.cleaning_logs) > step:
                st.json(st.session_state.cleaning_logs[step])
            else:
                st.info("No cleaning actions taken yet for this step.")
            # Optionally, show all logs so far in a collapsible section
            with st.expander("Show All Cleaning Logs So Far"):
                st.json(st.session_state.cleaning_logs)

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Previous", disabled=step == 0):
                    st.session_state.current_step = max(0, step - 1)
                    st.rerun()
            with col2:
                if st.button("Apply Changes and Next", key=f"apply_next_{step}"):
                    cleaned_df = st.session_state.df.copy()
                    step_logs = []
                    try:
                        for column, data in st.session_state.user_choices.items():
                            # Only apply changes for columns relevant to this agent
                            if data.get("agent") != agent:
                                continue
                            choice = data["choice"]
                            
                            # Handle duplicate agent specially as it acts on the whole dataframe and has a special key
                            if column == "duplicates":
                                if choice.get("suggested_action") == "drop_duplicates":
                                    code_to_run = agent.generate_code_from_choice(None, choice)
                                    log_entry = {
                                        "column": "__table__",
                                        "user_choice": choice,
                                        "code": code_to_run,
                                        "status": None,
                                        "error": None
                                    }
                                    try:
                                        if code_to_run and 'from scipy.stats import winsorize' in code_to_run:
                                            code_to_run = code_to_run.replace('from scipy.stats import winsorize', 'from scipy.stats.mstats import winsorize')
                                        exec(code_to_run, {'df': cleaned_df})
                                        log_entry["status"] = "success"
                                    except Exception as e:
                                        log_entry["status"] = "error"
                                        log_entry["error"] = str(e)
                                    step_logs.append(log_entry)
                                # After handling, continue to the next choice to avoid the generic column-based logic.
                                continue
                            
                            # Handle feature generation agent specially
                            if isinstance(agent, feature_generation_module.FeatureGenerationAgent):
                                formula = choice.get("formula")
                                feature_name = choice.get("name", column)
                                log_entry = {
                                    "column": feature_name,
                                    "user_choice": choice,
                                    "code": None,
                                    "status": None,
                                    "error": None
                                }
                                
                                if formula and isinstance(formula, str) and formula.strip():
                                    # Always create the new column, do not check for existence
                                    if not formula.strip().startswith(f"df['{feature_name}']"):
                                        formula_to_run = f"df['{feature_name}'] = {formula}"
                                    else:
                                        formula_to_run = formula
                                    
                                    try:
                                        import datetime
                                        current_date = pd.Timestamp(datetime.datetime.now())
                                        if code_to_run and 'from scipy.stats import winsorize' in code_to_run:
                                            code_to_run = code_to_run.replace('from scipy.stats import winsorize', 'from scipy.stats.mstats import winsorize')
                                        exec(formula_to_run, {'df': cleaned_df, 'pd': pd, 'np': np, 'datetime': datetime, 'current_date': current_date})
                                        log_entry["code"] = formula_to_run
                                        log_entry["status"] = "success"
                                        log_entry["error"] = None
                                    except Exception as e:
                                        log_entry["code"] = formula_to_run
                                        log_entry["status"] = "error"
                                        log_entry["error"] = str(e)
                                else:
                                    log_entry["status"] = "skipped"
                                    log_entry["error"] = "No formula provided or formula is empty."
                                
                                step_logs.append(log_entry)
                            else:
                                # Handle other agents (including value standardization)
                                agent_col = choice.get("name", column)
                                actual_col = find_actual_column_name(cleaned_df, agent_col)
                                
                                if not actual_col or actual_col not in cleaned_df.columns:
                                    log_entry = {
                                        "column": agent_col,
                                        "user_choice": choice,
                                        "code": None,
                                        "status": "error",
                                        "error": f"Column '{agent_col}' does not exist in DataFrame at this step."
                                    }
                                    step_logs.append(log_entry)
                                    continue
                                
                                code_to_run = agent.generate_code_from_choice(actual_col, choice)
                                log_entry = {
                                    "column": actual_col,
                                    "user_choice": choice,
                                    "code": code_to_run,
                                    "status": None,
                                    "error": None
                                }
                                
                                # Execute the code if it exists
                                if code_to_run and code_to_run.strip():
                                    # Check if the code contains actual executable statements (not just comments)
                                    code_lines = [line.strip() for line in code_to_run.split('\n') if line.strip()]
                                    executable_lines = [line for line in code_lines if not line.startswith('#') and line]
                                    
                                    if executable_lines:
                                        try:
                                            if code_to_run and 'from scipy.stats import winsorize' in code_to_run:
                                                code_to_run = code_to_run.replace('from scipy.stats import winsorize', 'from scipy.stats.mstats import winsorize')
                                            exec(code_to_run, {'df': cleaned_df, 'pd': pd, 'np': np})
                                            log_entry["status"] = "success"
                                        except Exception as e:
                                            log_entry["status"] = "error"
                                            log_entry["error"] = str(e)
                                else:
                                    log_entry["status"] = "skipped"
                                    log_entry["error"] = "No code generated or code is empty."
                                
                                step_logs.append(log_entry)
                            
                            # Record to persistent cleaning history for this file
                            st.session_state.cleaning_history.append({
                                "step": step,
                                "agent_name": agent_name,
                                "column": column,
                                "action": choice.get("suggested_action") or choice.get("suggested_strategy") or ("formula" if "formula" in choice else None),
                                "details": choice,
                                "reason": choice.get("reason"),
                                "code": log_entry["code"],
                                "status": log_entry["status"],
                                "error": log_entry.get("error")
                            })
                            
                    except Exception as e:
                        st.error(f"Error during cleaning step: {e}")
                    
                    # Save log for this step
                    if len(st.session_state.cleaning_logs) > step:
                        st.session_state.cleaning_logs[step] = step_logs
                    else:
                        st.session_state.cleaning_logs.append(step_logs)
                    
                    # Always update df for next step with all changes
                    st.session_state.df = cleaned_df
                    st.session_state.current_step = step + 1
                    st.rerun()

# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    main()