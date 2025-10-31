import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, List


def compare_lr_test_glm(model_unrestricted, model_restricted):
    """
    Likelihood ratio test for nested GLM models.
    
    Formula: LR = -2 * (LL_restricted - LL_unrestricted)
    where LR ~ Chi-square(df_diff)
    
    Parameters:
        model_unrestricted: Fitted GLM model with more parameters
        model_restricted: Fitted GLM model with fewer parameters (nested)
    
    Returns:
        lr_stat: Likelihood ratio statistic (chi-square distributed)
        p_value: P-value from chi-square distribution  
        df_diff: Degrees of freedom for the test
    """
    # Get log-likelihoods
    llf_unrestricted = model_unrestricted.llf
    llf_restricted = model_restricted.llf
    
    # LR statistic = -2 * (LL_restricted - LL_unrestricted)
    # This follows chi-square distribution
    lr_stat = -2 * (llf_restricted - llf_unrestricted)
    
    # Degrees of freedom = difference in number of parameters
    # df_resid = n - p, so smaller df_resid means more parameters
    df_diff = model_restricted.df_resid - model_unrestricted.df_resid
    
    # P-value from chi-square distribution (survival function)
    p_value = stats.chi2.sf(lr_stat, df_diff)
    
    return lr_stat, p_value, df_diff


# Removed: create_bands_from_quantiles - replaced by apply_variable_banding


def apply_bands(series: pd.Series, bands: Dict[str, Tuple[float, float]]) -> pd.Series:
    """
    Apply bands to a numeric series.
    
    Parameters:
        series: Numeric series to band
        bands: Dictionary with band labels and (min, max) tuples
    
    Returns:
        Categorical series with band labels
    """
    result = pd.Series(index=series.index, dtype='object')
    
    for label, (min_val, max_val) in bands.items():
        if max_val == float('inf'):
            # Last band: include min_val and everything above
            mask = series >= min_val
        else:
            # All other bands: include min_val, exclude max_val (except for exact matches at series max)
            mask = (series >= min_val) & (series < max_val)
            # Special case: include values that equal max_val if it's the actual maximum in the series
            if max_val == series.max():
                mask = mask | (series == max_val)
        
        result[mask] = label
    
    return pd.Categorical(result, categories=list(bands.keys()), ordered=True)


# Removed: band_numeric_variables (first definition) - replaced by apply_variable_banding


def compare_add_one_variable(baseline_model, baseline_formula: str, variables_to_test: List[str], 
                             data: pd.DataFrame, family, offset=None) -> pd.DataFrame:
    """
    Test adding each variable individually to baseline model and return LR test table.
    
    Parameters:
        baseline_model: Fitted baseline GLM model
        baseline_formula: Formula string of baseline model  
        variables_to_test: List of variables to test
        data: DataFrame with the data
        family: GLM family
        offset: Model offset (optional)
    
    Returns:
        DataFrame with variables sorted by significance
    """
    import statsmodels.formula.api as smf
    
    results = []
    baseline_deviance = baseline_model.deviance
    
    for var in variables_to_test:
        # Build extended formula
        if baseline_formula.endswith("~ 1"):
            extended_formula = baseline_formula.replace("~ 1", f"~ C({var})")
        else:
            extended_formula = baseline_formula + f" + C({var})"
        
        try:
            # Fit extended model
            extended_model = smf.glm(
                formula=extended_formula,
                data=data,
                family=family,
                offset=offset
            ).fit()
            
            # Run LR test
            lr_stat, p_value, df = compare_lr_test_glm(extended_model, baseline_model)
            
            results.append({
                'Variable': var,
                'Deviance_Reduction': baseline_deviance - extended_model.deviance,
                'DF': df,
                'P_Value': p_value
            })
        except:
            pass
    
    if results:
        df_results = pd.DataFrame(results).sort_values('P_Value')
        df_results['P_Value'] = df_results['P_Value'].apply(
            lambda x: f"{x:.4e}" if x < 0.0001 else f"{x:.4f}"
        )
        return df_results
    
    return pd.DataFrame()


def compare_add_one_variable_f_test(
    baseline_model, baseline_formula, variables_to_test, data, family, offset=None
):
    import pandas as pd, numpy as np
    import statsmodels.formula.api as smf
    from scipy.stats import f as f_dist

    results = []
    baseline_deviance = baseline_model.deviance
    baseline_df_resid = int(baseline_model.df_resid)

    for var in variables_to_test:
        # choose C(var) only for non-numeric
        is_cat = (data[var].dtype.kind in "OUSb") or str(data[var].dtype).startswith("category")
        term = f"C({var})" if is_cat else var
        extended_formula = f"{baseline_formula} + {term}"

        try:
            extended_model = smf.glm(
                formula=extended_formula,
                data=data,
                family=family,
                offset=offset
            ).fit()

            df_num = int(baseline_df_resid - extended_model.df_resid)
            if df_num <= 0:
                results.append({'Variable': var, 'Deviance_Reduction': np.nan,
                                'DF_Num': df_num, 'DF_Den': int(extended_model.df_resid),
                                'F': np.nan, 'P_Value': np.nan})
                continue

            phi_hat = extended_model.pearson_chi2 / extended_model.df_resid
            dev_drop = baseline_deviance - extended_model.deviance
            F_stat = (dev_drop / df_num) / phi_hat
            df_den = int(extended_model.df_resid)
            p_value = 1.0 - f_dist.cdf(F_stat, df_num, df_den)

            results.append({'Variable': var,
                            'Deviance_Reduction': dev_drop,
                            'DF_Num': df_num, 'DF_Den': df_den,
                            'F': float(F_stat), 'P_Value': float(p_value)})
        except Exception as e:
            results.append({'Variable': var, 'Deviance_Reduction': np.nan,
                            'DF_Num': np.nan, 'DF_Den': np.nan,
                            'F': np.nan, 'P_Value': np.nan, 'Error': str(e)})

    if results:
        df_results = pd.DataFrame(results).sort_values('P_Value', na_position='first')
        def fmt_p(p):
            if pd.isna(p): return p
            return f"{p:.4e}" if p < 1e-4 else f"{p:.4f}"
        df_results['P_Value'] = df_results['P_Value'].apply(fmt_p)
        return df_results

    return pd.DataFrame()

def calculate_observed_vs_predicted(data: pd.DataFrame, group_var: str, 
                                   observed_col: str, predicted_col: str, 
                                   offset_col: str) -> pd.DataFrame:
    """
    Calculate observed and predicted rates by group.
    
    Parameters:
        data: DataFrame with observations
        group_var: Variable to group by
        observed_col: Name of observed response column
        predicted_col: Name of predicted response column  
        offset_col: Name of offset column (exposure or claim_counts)
    
    Returns:
        DataFrame with observed rate, predicted rate, and offset sum by group
    """
    grouped = data.groupby(group_var, observed=True).agg({
        observed_col: 'sum',
        predicted_col: 'sum',
        offset_col: 'sum'
    }).reset_index()
    
    # Calculate rates
    grouped['observed_rate'] = grouped[observed_col] / grouped[offset_col]
    grouped['predicted_rate'] = grouped[predicted_col] / grouped[offset_col]
    
    # Rename columns for clarity
    grouped.columns = [group_var, 'observed_sum', 'predicted_sum', 'offset_sum', 
                       'observed_rate', 'predicted_rate']
    
    return grouped


# Removed: create_manual_bands - replaced by apply_variable_banding


def extract_relativities_table(model, selected_vars: List[str], data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Extract relativities table from a fitted GLM model and identify base levels.
    
    Parameters:
        model: Fitted GLM model
        selected_vars: List of variables in the model
        data: DataFrame containing the data (to find all levels)
    
    Returns:
        Tuple of (DataFrame with relativities and CIs, Dictionary of base levels)
    """
    # Extract relativities and confidence intervals
    relativities = np.exp(model.params)
    ci_lower = np.exp(model.conf_int()[0])
    ci_upper = np.exp(model.conf_int()[1])
    
    # Create DataFrame
    relativities_df = pd.DataFrame({
        'Parameter': relativities.index,
        'Relativity': relativities.values,
        'CI_Lower': ci_lower.values,
        'CI_Upper': ci_upper.values
    })
    
    # Identify base levels
    base_levels = {}
    for var in selected_vars:
        # Check if this variable has parameters in the model
        var_params = [p for p in relativities.index if f'C({var})' in p]
        
        if var_params:
            # Extract the levels that appear in parameters
            param_levels = [p.split('[T.')[-1].replace(']', '') for p in var_params]
            
            # Find the base level
            if var in data.columns:
                all_levels = data[var].unique()
                base_level_list = [level for level in all_levels if str(level) not in param_levels]
                if base_level_list:
                    base_levels[var] = str(base_level_list[0])
    
    return relativities_df, base_levels


# Removed: band_numeric_variables (second definition) - replaced by apply_variable_banding


def get_base_level_by_weight(data: pd.DataFrame, variable: str, weight_col: str = 'exposure') -> str:
    """
    Find the base level for a variable based on highest weight (exposure or claim counts).
    
    Parameters:
        data: DataFrame containing the variable and weight column
        variable: Variable name to find base level for
        weight_col: Weight column ('exposure', 'claim_counts', etc.)
    
    Returns:
        Base level (string) with highest total weight
    """
    if variable not in data.columns or weight_col not in data.columns:
        return None
    
    # Group by variable and sum the weight
    weight_by_level = data.groupby(variable, observed=True)[weight_col].sum().sort_values(ascending=False)
    base_level = str(weight_by_level.index[0])
    
    print(f"Base level for {variable}: {base_level} (weight: {weight_by_level.iloc[0]:,.0f})")
    
    return base_level


def build_glm_formula(response_var: str, 
                     risk_variables: List[str], 
                     data: pd.DataFrame,
                     offset_col: str = 'exposure',
                     reference_level: str = 'highest_weight') -> str:
    """
    Build GLM formula string with proper base level specification.
    
    Parameters:
        response_var: Response variable name (e.g., 'claim_counts', 'loss_amt', 'predicted_pure_premium')
        risk_variables: List of risk variables to include in the model
        data: DataFrame containing the variables
        offset_col: Offset column to determine base levels ('exposure', 'claim_counts', etc.)
        reference_level: Method for selecting reference levels ('highest_weight' only supported for now)
    
    Returns:
        Complete GLM formula string ready for statsmodels
    
    Examples:
        >>> build_glm_formula('claim_counts', ['driver_age_band', 'horsepower_band'], df, 'exposure')
        'claim_counts ~ C(driver_age_band, Treatment(reference="35-44")) + C(horsepower_band, Treatment(reference="106-132"))'
        
        >>> build_glm_formula('loss_amt', [], df)  # No variables
        'loss_amt ~ 1'
    """
    # Validate reference level method
    if reference_level != 'highest_weight':
        raise ValueError(f"reference_level '{reference_level}' not supported. Only 'highest_weight' is currently available.")
    
    # Handle intercept-only model
    if not risk_variables:
        return f"{response_var} ~ 1"
    
    # Filter variables that exist in the data
    available_vars = [var for var in risk_variables if var in data.columns]
    
    if not available_vars:
        print(f"Warning: None of the specified variables {risk_variables} found in data. Using intercept-only model.")
        return f"{response_var} ~ 1"
    
    # Build formula terms with base levels
    formula_parts = []
    
    for var in available_vars:
        if reference_level == 'highest_weight':
            # Get base level with highest offset
            base_level = get_base_level_by_weight(data, var, offset_col)
            
            if base_level:
                # Create factor with explicit base level
                formula_parts.append(f"C({var}, Treatment(reference='{base_level}'))")
            else:
                # Fallback to regular categorical
                formula_parts.append(f"C({var})")
    
    # Combine into full formula
    formula_terms = " + ".join(formula_parts)
    formula = f"{response_var} ~ {formula_terms}"
    
    print(f"GLM formula: {formula}")
    
    return formula


def apply_bands_from_other_dataset(target_data: pd.DataFrame, source_bands: Dict[str, Dict[str, Tuple[float, float]]], suffix: str = '_sev') -> pd.DataFrame:
    """
    Apply bands from another dataset to the target dataset.
    This is useful when you want to use severity bands on frequency dataset for prediction.
    
    Parameters:
        target_data: DataFrame to add bands to (e.g., frequency_dataset)
        source_bands: Band definitions from another dataset (e.g., sev_bands)
        suffix: Suffix to add to band column names (e.g., '_sev' creates 'driver_age_sev_band')
    
    Returns:
        DataFrame with new band columns added
    """
    result_data = target_data.copy()
    
    print(f"🔄 Transferring bands to target dataset with suffix '{suffix}':")
    
    for variable, bands in source_bands.items():
        if variable in result_data.columns:
            # Create new band column name with suffix
            band_col_name = f'{variable}{suffix}_band'
            
            # Apply the bands to the target dataset
            result_data[band_col_name] = apply_bands(result_data[variable], bands)
            
            print(f"  ✅ Created {band_col_name} from {variable}")
            
            # Show sample distribution
            print(f"     Sample distribution:")
            for band_label, count in result_data[band_col_name].value_counts().head(3).items():
                pct = count / len(result_data) * 100
                print(f"       {band_label}: {count:,} ({pct:.1f}%)")
        else:
            print(f"  ⚠️  Variable {variable} not found in target dataset, skipping")
    
    return result_data


# Removed: apply_variable_banding - replaced by apply_banding_config
# Removed: apply_categorical_grouping - replaced by apply_banding_config

# ============================================================================
# JSON-BASED BANDING CONFIGURATION SYSTEM
# ============================================================================

def generate_banding_config(data: pd.DataFrame,
                           numeric_vars: List[str],
                           categorical_vars: List[str],
                           weight_col: str = 'exposure',
                           n_quantiles: int = 4) -> Dict:
    """
    Generate initial banding configuration with automatic quartiles for numeric variables
    and all unique levels for categorical variables.

    Parameters:
        data: DataFrame containing the variables
        numeric_vars: List of numeric variable names
        categorical_vars: List of categorical variable names
        weight_col: Column to use for weighting ('exposure' or 'claim_counts')
        n_quantiles: Number of quantiles for automatic banding

    Returns:
        Dictionary ready for JSON serialization with format:
        {
            "driver_age": [32, 54, 72],  # numeric: cutpoints
            "province": ["RI", "NJ", ...]  # categorical: all levels
        }

    Example:
        >>> config = generate_banding_config(
        ...     df,
        ...     numeric_vars=['driver_age', 'vehicle_age'],
        ...     categorical_vars=['province', 'body_type'],
        ...     weight_col='exposure'
        ... )
        >>> save_banding_config(config, 'outputs/bands_freq.json')
    """
    config = {}

    # Process numeric variables - generate weighted quantile cutpoints
    for var in numeric_vars:
        if var not in data.columns:
            print(f"⚠️  Warning: Numeric variable '{var}' not found in data, skipping")
            continue

        # Sort by variable and calculate cumulative weights
        sorted_data = data.sort_values(var)
        cumulative_weights = sorted_data[weight_col].cumsum()
        total_weight = cumulative_weights.iloc[-1]

        # Find quantile cutpoints based on weighted distribution
        cutpoints = []
        for i in range(1, n_quantiles):
            target_weight = (i / n_quantiles) * total_weight
            idx = (cumulative_weights >= target_weight).idxmax()
            cutpoints.append(sorted_data.loc[idx, var])

        # Remove duplicates and sort
        cutpoints = sorted(list(set(cutpoints)))

        # Round to integers for cleaner JSON
        cutpoints = [int(round(cp)) for cp in cutpoints]

        config[var] = cutpoints
        print(f"✅ {var}: Generated {len(cutpoints)} cutpoints: {cutpoints}")

    # Process categorical variables - extract all unique levels
    for var in categorical_vars:
        if var not in data.columns:
            print(f"⚠️  Warning: Categorical variable '{var}' not found in data, skipping")
            continue

        # Get all unique levels sorted
        levels = sorted(data[var].unique().astype(str).tolist())
        config[var] = levels
        print(f"✅ {var}: Extracted {len(levels)} levels")

    return config


def apply_banding_config(data: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply banding configuration from JSON to create *_band columns for all variables.

    Parameters:
        data: DataFrame to apply banding to
        config: Configuration dict from JSON with format:
                - Numeric: {"driver_age": [25, 35, 50, 65]}
                - Categorical (no grouping): {"province": ["RI", "NJ", "AS"]}
                - Categorical (with grouping): {"vehicle_make": [["Mercedes"], ["Toyota", "Ford"]]}

    Returns:
        Tuple of (modified DataFrame with *_band columns, bands_dict with band definitions)

    Example:
        >>> config = load_banding_config('outputs/bands_freq.json')
        >>> df, bands = apply_banding_config(df, config)
    """
    data_copy = data.copy()
    bands_dict = {}

    for var, value in config.items():
        if var not in data_copy.columns:
            print(f"⚠️  Warning: Variable '{var}' not found in data, skipping")
            continue

        band_col_name = f"{var}_band"

        # Check if it's numeric banding (list of numbers) or categorical
        if isinstance(value, list) and len(value) > 0:
            # Check if it's a list of numbers (numeric cutpoints)
            if isinstance(value[0], (int, float)):
                # Numeric variable with cutpoints
                cutpoints = value

                # Create bands dictionary
                bands = {}
                bands[f"0-{cutpoints[0]}"] = (0, cutpoints[0])

                for i in range(len(cutpoints) - 1):
                    bands[f"{cutpoints[i]}-{cutpoints[i+1]}"] = (cutpoints[i], cutpoints[i+1])

                bands[f"{cutpoints[-1]}-inf"] = (cutpoints[-1], float('inf'))

                # Apply bands using existing apply_bands function
                data_copy[band_col_name] = apply_bands(data_copy[var], bands)
                bands_dict[var] = bands

                print(f"✅ {var}: Applied {len(bands)} numeric bands")

            # Check if it's a nested list (categorical grouping)
            elif isinstance(value[0], list):
                # Categorical variable with grouping
                grouping = value

                # Create mapping from original levels to group names (using first element)
                level_to_group = {}
                group_levels = {}

                for group in grouping:
                    group_name = group[0]  # Use first element as group name
                    group_levels[group_name] = group
                    for level in group:
                        level_to_group[level] = group_name

                # Apply grouping
                data_copy[band_col_name] = data_copy[var].astype(str).map(
                    lambda x: level_to_group.get(x, x)
                )

                # Convert to categorical with ordered groups
                data_copy[band_col_name] = pd.Categorical(
                    data_copy[band_col_name],
                    categories=list(group_levels.keys()),
                    ordered=False
                )

                bands_dict[var] = group_levels

                print(f"✅ {var}: Applied grouping into {len(group_levels)} groups")

            else:
                # Categorical variable without grouping (flat list of levels)
                levels = value

                # Create pass-through mapping (each level to itself)
                level_mapping = {level: [level] for level in levels}

                # Validate that all data levels exist in config
                data_levels = set(data_copy[var].astype(str).unique())
                config_levels = set(levels)

                missing_in_config = data_levels - config_levels
                if missing_in_config:
                    print(f"⚠️  Warning: {var} has levels in data not in config: {missing_in_config}")

                extra_in_config = config_levels - data_levels
                if extra_in_config:
                    print(f"ℹ️  Info: {var} has levels in config not in data: {extra_in_config}")

                # Create band column (copy of original, but as categorical)
                data_copy[band_col_name] = pd.Categorical(
                    data_copy[var].astype(str),
                    categories=levels,
                    ordered=False
                )

                bands_dict[var] = level_mapping

                print(f"✅ {var}: No grouping, using {len(levels)} original levels")

        else:
            print(f"⚠️  Warning: Invalid config value for '{var}': {value}")

    return data_copy, bands_dict


def save_banding_config(config: Dict, filepath: str) -> None:
    """
    Save banding configuration to JSON file with pretty printing.

    Parameters:
        config: Configuration dictionary
        filepath: Path to save JSON file (e.g., 'outputs/bands_freq.json')

    Example:
        >>> config = generate_banding_config(df, ['driver_age'], ['province'])
        >>> save_banding_config(config, 'outputs/bands_freq.json')
    """
    import json
    import os

    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Saved banding configuration to: {filepath}")


def load_banding_config(filepath: str) -> Dict:
    """
    Load banding configuration from JSON file.

    Parameters:
        filepath: Path to JSON file (e.g., 'outputs/bands_freq.json')

    Returns:
        Configuration dictionary

    Example:
        >>> config = load_banding_config('outputs/bands_freq.json')
        >>> df, bands = apply_banding_config(df, config)
    """
    import json

    with open(filepath, 'r') as f:
        config = json.load(f)

    print(f"✅ Loaded banding configuration from: {filepath}")
    print(f"📊 Variables in config: {list(config.keys())}")

    return config


def create_consolidated_config(freq_bands: Dict, sev_bands: Dict) -> Dict:
    """
    Create consolidated configuration showing union of frequency and severity bands.

    For numeric variables:
    - Takes union of all cutpoints from freq and sev
    - Creates new consolidated bands

    For categorical variables:
    - Takes union of groupings from freq and sev
    - If both have groupings, uses frequency grouping (can be manually adjusted)

    Parameters:
        freq_bands: Band definitions from frequency model (output of apply_banding_config)
        sev_bands: Band definitions from severity model (output of apply_banding_config)

    Returns:
        Consolidated configuration dictionary ready for JSON serialization

    Example:
        >>> freq_config = load_banding_config('outputs/bands_freq.json')
        >>> sev_config = load_banding_config('outputs/bands_sev.json')
        >>> _, freq_bands = apply_banding_config(df, freq_config)
        >>> _, sev_bands = apply_banding_config(df, sev_config)
        >>> consolidated = create_consolidated_config(freq_bands, sev_bands)
        >>> save_banding_config(consolidated, 'outputs/bands_consolidated.json')
    """
    consolidated = {}

    # Get union of all variables
    all_vars = set(list(freq_bands.keys()) + list(sev_bands.keys()))

    for var in all_vars:
        freq_def = freq_bands.get(var)
        sev_def = sev_bands.get(var)

        # Determine if variable is numeric in either freq or sev definitions
        is_freq_numeric = bool(freq_def) and isinstance(list(freq_def.values())[0], tuple)
        is_sev_numeric = bool(sev_def) and isinstance(list(sev_def.values())[0], tuple)

        if is_freq_numeric or is_sev_numeric:
            # Numeric variable - union of cutpoints
            freq_cutoffs = []
            sev_cutoffs = []

            if is_freq_numeric:
                for band_label, (min_val, max_val) in freq_def.items():
                    if max_val != float('inf') and max_val > 0:
                        freq_cutoffs.append(int(max_val))

            if is_sev_numeric:
                for band_label, (min_val, max_val) in sev_def.items():
                    if max_val != float('inf') and max_val > 0:
                        sev_cutoffs.append(int(max_val))

            # Union of all cutoffs
            all_cutoffs = sorted(list(set(freq_cutoffs + sev_cutoffs)))
            consolidated[var] = all_cutoffs

            print(f"📊 {var}: Consolidated {len(all_cutoffs)} cutpoints from freq+sev")

        else:
            # Categorical variable - compute refined union (outer split) if both sides present
            # Otherwise, use whichever side is available
            if freq_def and sev_def:
                # Build level -> group name maps for both sides
                def build_level_to_group(defn: Dict[str, List[str]]) -> Dict[str, str]:
                    mapping: Dict[str, str] = {}
                    for group_name, levels in defn.items():
                        # levels is list of levels (len 1 for no-grouping case)
                        for lvl in levels:
                            mapping[str(lvl)] = str(group_name)
                    return mapping

                lvl_to_fg = build_level_to_group(freq_def)
                lvl_to_sg = build_level_to_group(sev_def)

                # Universe of levels appearing in either side
                all_levels = set(list(lvl_to_fg.keys()) + list(lvl_to_sg.keys()))

                # Combined refined groups keyed by (freq_group, sev_group)
                combined: Dict[Tuple[str, str], List[str]] = {}
                for lvl in all_levels:
                    fg = lvl_to_fg.get(lvl, lvl)
                    sg = lvl_to_sg.get(lvl, lvl)
                    key = (fg, sg)
                    combined.setdefault(key, []).append(lvl)

                # If all groups are singletons, emit flat list; else nested list of groups
                if all(len(members) == 1 for members in combined.values()):
                    consolidated[var] = sorted(list(all_levels))
                else:
                    consolidated[var] = [sorted(members) for members in combined.values()]

                print(f"📊 {var}: Refined union of frequency and severity groupings → {len(consolidated[var])} groups")
            elif freq_def:
                # Use frequency definition
                if all(len(v) == 1 for v in freq_def.values()):
                    consolidated[var] = list(freq_def.keys())
                else:
                    consolidated[var] = [list(v) for v in freq_def.values()]
                print(f"📊 {var}: Using frequency categorical definition")
            elif sev_def:
                # Use severity definition
                if all(len(v) == 1 for v in sev_def.values()):
                    consolidated[var] = list(sev_def.keys())
                else:
                    consolidated[var] = [list(v) for v in sev_def.values()]
                print(f"📊 {var}: Using severity categorical definition")

    return consolidated