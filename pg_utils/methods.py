from typing import Dict, Any, Tuple, List
import json
import math
import glob
import re
import pandas as pd


def identify_player_number(data_dir: str = 'files') -> Tuple[int, str]:
    """
    Identify player number from loss data file in specified directory.

    Args:
        data_dir: Directory to search for p{X}_loss_data.csv file

    Returns:
        Tuple of (player_number, file_path)

    Raises:
        FileNotFoundError: If no player loss data file found
        ValueError: If player number cannot be extracted from filename
    """
    # Find player-specific loss data file
    loss_data_files = glob.glob(f'{data_dir}/p*_loss_data.csv')

    if not loss_data_files:
        raise FileNotFoundError(
            f"❌ No player loss data file found in {data_dir}/ directory!\n"
            f"📋 Please copy your p{{X}}_loss_data.csv file from wds/ to {data_dir}/"
        )

    # Use the first match (should only be one)
    loss_data_file = loss_data_files[0]

    # Extract player number from filename
    player_match = re.search(r'p(\d+)_loss_data\.csv', loss_data_file)
    if not player_match:
        raise ValueError(f"Could not extract player number from {loss_data_file}")

    player_num = int(player_match.group(1))

    return player_num, loss_data_file


def parse_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def _parse_band_key(key: str) -> Tuple[float, float]:
    s = str(key).strip()
    if '-' in s:
        if s.lower().startswith('-inf-'):
            # Handle -inf-y case
            hi = s[5:].strip()  # Remove '-inf-' prefix
            min_val = 0.0
            max_val = float('inf') if hi.lower() == 'inf' else float(hi)
        else:
            lo, hi = s.split('-', 1)
            lo = lo.strip()
            hi = hi.strip()
            min_val = 0.0 if lo.lower() == 'inf' else float(lo)  # This case shouldn't happen but safety
            max_val = float('inf') if hi.lower() == 'inf' else float(hi)
        return (min_val, max_val)
    v = float(s)
    return (v, v)


def get_band_factor(value, var_name: str, factor_table: Dict) -> float:
    if isinstance(value, str):
        try:
            return float(factor_table.get(value, 1.0))
        except Exception:
            return 1.0
    try:
        x = float(value)
    except Exception:
        return 1.0
    base_factor = None
    for key, val in factor_table.items():
        try:
            lo, hi = _parse_band_key(key)
            if float(val) == 1.0 and base_factor is None:
                base_factor = 1.0
            if x >= float(lo) and x <= float(hi):
                return float(val)
        except Exception:
            continue
    return base_factor if base_factor is not None else 1.0


def calculate_means_with_factors(df: pd.DataFrame, factor_tables: Dict, base_value_key: str = 'value') -> pd.Series:
    """
    Calculate means using factor tables with relativities format.
    Supports both raw variables and their banded counterparts (e.g., province vs province_band).
    """
    # Use relativities format directly
    relativities = factor_tables['relativities']

    # Build mapping from JSON key -> dataframe column to read (prefer exact match, else *_band)
    variable_column_map = {}
    for var_name in relativities.keys():
        if var_name in df.columns:
            variable_column_map[var_name] = var_name
        else:
            band_col = f"{var_name}_band"
            if band_col in df.columns:
                variable_column_map[var_name] = band_col
    base_value = float(factor_tables.get('base_values', {}).get(base_value_key, 1.0))
    
    def row_mean(row) -> float:
        product = 1.0
        for var_name, col_name in variable_column_map.items():
            product *= get_band_factor(row[col_name], var_name, relativities[var_name])
        return base_value * product

    mean_series = df.apply(row_mean, axis=1)
    mean_series.name = 'mean_value'
    return mean_series


def flatten_nominal_relativities(rating_structure: Dict[str, Any], bands_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten grouped nominal relativities to per-level entries for competition scoring.

    For each nominal variable where bands_dict[var] maps group_name -> [levels],
    duplicate the group's factor to each atomic level key in relativities.

    Numeric variables (bands as (min, max) tuples) are left unchanged.
    """
    import copy

    flattened = copy.deepcopy(rating_structure)
    rel = flattened.get('relativities', {})

    for var_name, table in list(rel.items()):
        groups = bands_dict.get(var_name)
        if not groups or not isinstance(groups, dict) or len(groups) == 0:
            continue

        first_val = list(groups.values())[0]
        # Numeric bands are tuples; skip flattening for those
        if isinstance(first_val, tuple):
            continue

        # Categorical/grouped: groups is dict[group_name] -> [levels]
        per_level: Dict[str, float] = {}
        for group_name, levels in groups.items():
            factor = float(table.get(group_name, 1.0)) if isinstance(table, dict) else 1.0
            for lvl in levels:
                per_level[str(lvl)] = factor

        flattened['relativities'][var_name] = per_level

    return flattened


def logit(x):
    """Logistic function: 1 / (1 + exp(-x))"""
    import numpy as np
    return 1.0 / (1.0 + np.exp(-x))


def calculate_retention_probability(df: pd.DataFrame, retention_model_path: str = 'files/retention.json') -> pd.Series:
    """
    Calculate retention probability based on premium delta and driver age.
    
    Args:
        df: DataFrame with 'new_comm_premium', 'commercial_premium', and optionally 'driver_age'
        retention_model_path: Path to retention model JSON file
    
    Returns:
        Series with retention probabilities
    """
    import numpy as np
    import json
    import os
    
    # Load retention model
    if not os.path.exists(retention_model_path):
        raise FileNotFoundError(f"Retention model file not found: {retention_model_path}")
    
    with open(retention_model_path, 'r') as f:
        retention_model = json.load(f)
    
    # Extract model parameters
    model = retention_model.get('model', {})
    if model.get('link') != 'logit':
        raise ValueError("Only 'logit' link is supported")
    
    intercept = float(model.get('intercept', 0.0))
    coeffs = model.get('coeffs', {})
    beta_delta = float(coeffs.get('delta_premium_centered', 0.0))
    transform = model.get('transform', {})
    delta_center = float(transform.get('delta_center', 1.0))
    age_bands = model.get('age_bands', {}).get('driver_age', {})
    
    # Calculate premium delta ratio
    delta_ratio = (df['new_comm_premium'] / df['commercial_premium']).values
    
    # Center the delta around delta_center
    x_centered = delta_ratio - delta_center
    
    # Linear predictor: intercept + beta * x_centered
    eta = intercept + beta_delta * x_centered
    
    # Add age adjustments if driver_age column exists
    if 'driver_age' in df.columns and age_bands:
        ages = df['driver_age'].values
        age_adj = np.zeros_like(eta)
        
        # Map ages to bands and apply adjustments
        for i, age in enumerate(ages):
            if pd.isna(age):
                continue
            
            # Find appropriate age band
            for band_key, adjustment in age_bands.items():
                try:
                    if '-' in band_key:
                        lo, hi = band_key.split('-', 1)
                        lo_val = float(lo) if lo.lower() != 'inf' else -np.inf
                        hi_val = float(hi) if hi.lower() != 'inf' else np.inf
                    else:
                        lo_val = hi_val = float(band_key)
                    
                    if lo_val <= age <= hi_val:
                        age_adj[i] = float(adjustment)
                        break
                except (ValueError, TypeError):
                    continue
        
        eta = eta + age_adj
    
    # Apply logistic function to get probabilities
    retention_prob = logit(eta)
    
    return pd.Series(retention_prob, index=df.index, name='retention_prob')


def retention_analysis_by_variable(df: pd.DataFrame, variable: str, top_n: int = 10) -> pd.DataFrame:
    """
    Analyze retention rates and premium deltas by a categorical variable.
    
    Args:
        df: DataFrame with retention_prob, premium_delta, and the variable
        variable: Column name to group by
        top_n: Number of top levels to show (by policy count)
    
    Returns:
        DataFrame with grouped results
    """
    if variable not in df.columns:
        print(f"Variable '{variable}' not found in data")
        return None
    
    # Group by variable and calculate metrics
    grouped = df.groupby(variable, observed=True).agg({
        'retention_prob': ['sum', 'count'],
        'premium_delta': 'mean',
        'new_comm_premium': 'mean',
        'commercial_premium': 'mean'
    }).round(4)
    
    # Flatten column names
    grouped.columns = ['total_retention_prob', 'policy_count', 'avg_premium_delta', 
                      'avg_new_premium', 'avg_old_premium']
    
    # Calculate retention rate and delta ratio
    grouped['retention_rate'] = grouped['total_retention_prob'] / grouped['policy_count']
    grouped['avg_delta_ratio'] = grouped['avg_new_premium'] / grouped['avg_old_premium']
    
    # Sort by policy count and take top N
    grouped = grouped.sort_values('policy_count', ascending=False).head(top_n)
    
    return grouped.reset_index()


def prepare_profitability_data(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Prepare profitability analysis data for a given variable (following univariate pattern).

    Args:
        df: DataFrame with inforce policies
        variable: Variable to group by

    Returns:
        DataFrame with aggregated profitability metrics
    """
    # Calculate earned premiums BEFORE grouping (premium × exposure for each policy)
    df = df.copy()
    df['earned_premium_old'] = df['commercial_premium'] * df['exposure']
    df['earned_premium_new'] = df['new_comm_premium'] * df['exposure']

    # Aggregate data by selected variable (following univariate pattern)
    grouped = df.groupby(variable, observed=True).agg({
        'exposure': 'sum',
        'commercial_premium': 'sum',
        'new_comm_premium': 'sum',
        'earned_premium_old': 'sum',
        'earned_premium_new': 'sum',
        'loss_amt': 'sum',
        'claim_counts': 'sum',
        'policy_id': 'count'
    }).reset_index()

    grouped.columns = [variable, 'exposure', 'total_premium_old', 'total_premium_new',
                       'earned_premium_old', 'earned_premium_new',
                       'loss_amt', 'claim_counts', 'policy_count']

    # Calculate derived metrics for both old and new premiums
    grouped['avg_premium_old'] = grouped['total_premium_old'] / grouped['policy_count']
    grouped['avg_premium_new'] = grouped['total_premium_new'] / grouped['policy_count']
    # Loss ratio = total_losses / earned_premium (already summed correctly)
    grouped['loss_ratio_old'] = (grouped['loss_amt'] / grouped['earned_premium_old'] * 100).round(2)
    grouped['loss_ratio_new'] = (grouped['loss_amt'] / grouped['earned_premium_new'] * 100).round(2)

    return grouped


def calculate_metrics(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Calculate univariate analysis metrics for a given variable.

    Args:
        df: DataFrame with policy data
        variable: Variable to group by

    Returns:
        DataFrame with aggregated metrics
    """
    # Calculate earned premium BEFORE grouping (premium × exposure for each policy)
    df = df.copy()
    df['earned_premium'] = df['commercial_premium'] * df['exposure']

    # Aggregate data by selected variable
    grouped = df.groupby(variable, observed=True).agg({
        'exposure': 'sum',
        'commercial_premium': 'sum',
        'earned_premium': 'sum',
        'loss_amt': 'sum',
        'claim_counts': 'sum',
        'policy_id': 'count'  # Count of policies
    }).reset_index()

    grouped.columns = [variable, 'exposure', 'commercial_premium', 'earned_premium', 'loss_amt', 'claim_counts', 'policy_count']

    # Calculate derived metrics
    grouped['avg_premium'] = grouped['commercial_premium'] / grouped['policy_count']
    # Loss ratio = total_losses / earned_premium (already summed correctly)
    grouped['loss_ratio'] = (grouped['loss_amt'] / grouped['earned_premium'] * 100).round(2)
    grouped['pure_premium'] = (grouped['loss_amt'] / grouped['exposure']).round(2)
    grouped['frequency'] = (grouped['claim_counts'] / grouped['exposure']).round(4)
    grouped['severity'] = (grouped['loss_amt'] / grouped['claim_counts'].replace(0, float('nan'))).round(2)
    
    return grouped


def portfolio_summary(df: pd.DataFrame) -> dict:
    """
    Calculate portfolio-level summary metrics.
    
    Args:
        df: DataFrame with policy data
    
    Returns:
        Dictionary with formatted portfolio metrics
    """
    # Calculate earned premium (commercial_premium × exposure)
    total_earned_premium = (df['commercial_premium'] * df['exposure']).sum()

    metrics = {
        'Policy Count': f"{len(df):,}",
        'Total Exposure': f"{df['exposure'].sum():,.2f}",
        'Total Claim Counts': f"{df['claim_counts'].sum():,}",
        'Total Loss Amount': f"${df['loss_amt'].sum():,.2f}",
        'Total Commercial Premium': f"${df['commercial_premium'].sum():,.2f}",
        'Total Earned Premium': f"${total_earned_premium:,.2f}",
        'Average Commercial Premium': f"${(df['commercial_premium'].sum() / len(df)):.2f}",
        'Loss Ratio': f"{(df['loss_amt'].sum() / total_earned_premium * 100):.2f}%",
        'Pure Premium': f"${(df['loss_amt'].sum() / df['exposure'].sum()):.2f}"
    }
    return metrics


def validate_glm_calculation(model, df: pd.DataFrame, rating_structure: dict, tolerance: float = 1.0) -> str:
    """
    Validate that GLM predictions match calculate_means_with_factors results.
    
    Args:
        model: Fitted GLM model
        df: DataFrame with policy data
        rating_structure: Rating structure JSON
        tolerance: Maximum allowed difference in dollars
    
    Returns:
        Validation message
    """
    glm_predicted_premium = model.predict(df).sum() / df['policy_id'].count()
    calc_predicted_premium = calculate_means_with_factors(df, rating_structure).sum() / df['policy_id'].count()
    difference = abs(glm_predicted_premium - calc_predicted_premium)
    
    if difference < tolerance:
        return f"✅ Validation OK - Difference: ${difference:.4f} (< ${tolerance:.2f})"
    else:
        return f"ℹ️ Validation Note - Difference: ${difference:.4f} (>= ${tolerance:.2f})"


def aggregate_by_group(df: pd.DataFrame, group_var: str, agg_cols: List[str]) -> pd.DataFrame:
    """Aggregate specified columns by grouping variable, summing values."""
    available_cols = {col: 'sum' for col in agg_cols if col in df.columns}
    if not available_cols:
        raise ValueError(f"None of the aggregation columns {agg_cols} found in DataFrame")
    
    grouped = df.groupby(group_var, observed=True).agg(available_cols).reset_index()
    return grouped


def calculate_loss_ratio(grouped_df: pd.DataFrame, premium_col: str = 'commercial_premium', loss_col: str = 'loss_amt') -> pd.DataFrame:
    """Calculate loss ratio percentage from aggregated premium and loss columns."""
    result = grouped_df.copy()
    if premium_col in result.columns and loss_col in result.columns:
        result['lr_pct'] = (result[loss_col] / result[premium_col]).replace([float('inf')], 0.0) * 100.0
        result['lr_pct'] = result['lr_pct'].round(2)
    else:
        result['lr_pct'] = 0.0
    return result


def get_rating_variables(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify rating variables from the dataframe, excluding operational columns.
    Returns a dictionary with categorical, numeric, and all rating variables.
    """
    # Get categorical and numeric columns
    categorical_vars = df.select_dtypes(include=['object']).columns.tolist()
    numeric_vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Define columns to exclude (operational/result columns)
    exclude_numeric = ['exposure', 'claim_counts', 'loss_amt', 'player_id', 'commercial_premium']
    exclude_categorical = ['policy_id', 'effective_date', 'expiration_date', 'cancellation_date']
    
    # Filter out non-rating variables
    rating_numeric = [v for v in numeric_vars if v not in exclude_numeric]
    rating_categorical = [v for v in categorical_vars if v not in exclude_categorical]
    
    # Combine all rating variables
    all_rating = rating_categorical + rating_numeric
    
    return {
        'categorical': rating_categorical,
        'numeric': rating_numeric,
        'all': all_rating
    }


# Removed: consolidate_variables_and_bands - replaced by create_consolidated_config and apply_banding_config
# Removed: prepare_consolidation_predictions - no longer needed with new JSON-based workflow


def glm_to_rating_structure(model, selected_vars: List[str], data: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert GLM model relativities to rating structure JSON format.
    
    Parameters:
        model: Fitted GLM model
        selected_vars: List of variables in the model
        data: DataFrame containing the data (to find all levels)
    
    Returns:
        Dictionary in rating structure JSON format
    """
    import json
    import numpy as np
    
    # Extract relativities and base premium
    relativities = np.exp(model.params)
    base_premium = np.exp(model.params['Intercept'])
    
    # Create rating structure dictionary
    rating_structure = {
        "base_values": {
            "value": round(base_premium, 2)
        },
        "relativities": {},
        "base_group": ""
    }
    
    # Extract relativities for each variable
    for var in selected_vars:
        # Remove '_band' suffix for output variable name (to match dataset columns)
        output_var = var.replace('_band', '')
        
        # Fixed regex: match parameters that start with C(var, 
        var_params = {param: rel for param, rel in relativities.items() 
                      if param.startswith(f'C({var},') and param != 'Intercept'}
        
        if var_params:
            # Get all levels for this variable
            all_levels = sorted(data[var].unique().astype(str))
            var_relativities = {}
            
            # Find base level (not in parameters)
            param_levels = [param.split('[T.')[-1].replace(']', '') for param in var_params.keys()]
            base_level_candidates = [level for level in all_levels if str(level) not in param_levels]
            if base_level_candidates:
                base_level = base_level_candidates[0]
                
                # Set base level to 1.0
                var_relativities[str(base_level)] = 1.0
                
                # Add to base_group string
                if rating_structure["base_group"]:
                    rating_structure["base_group"] += f", {output_var}: {base_level}"
                else:
                    rating_structure["base_group"] = f"{output_var}: {base_level}"
                
                # Add other levels with their relativities
                for param, relativity in var_params.items():
                    level = param.split('[T.')[-1].replace(']', '')
                    var_relativities[level] = round(relativity, 4)
                
                rating_structure["relativities"][output_var] = var_relativities
    
    return rating_structure




def validate_rating_structure(obj: Dict[str, Any]) -> Dict[str, Any]:
    errors = []
    if not isinstance(obj, dict):
        return {"ok": False, "errors": ["Rating structure must be a JSON object."]}
    bv = obj.get('base_values', {})
    if not isinstance(bv, dict) or 'value' not in bv:
        errors.append("Missing base_values.value")
    else:
        try:
            float(bv.get('value'))
        except Exception:
            errors.append("base_values.value must be numeric")
    rel = obj.get('relativities', {})
    if not isinstance(rel, dict):
        errors.append("relativities must be an object")
    else:
        for var, table in rel.items():
            if not isinstance(table, dict):
                errors.append(f"relativities.{var} must be an object")
                continue
            for k, v in table.items():
                try:
                    float(v)
                except Exception:
                    errors.append(f"relativities.{var}['{k}'] must be numeric")
    return {"ok": len(errors) == 0, "errors": errors}
