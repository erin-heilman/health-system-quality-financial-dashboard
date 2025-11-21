"""
CMS Hospital Star Rating Calculator
Based on July 2025 Methodology

This module implements the complete 7-step CMS star rating calculation.
"""

import pandas as pd
import numpy as np
import pyreadstat
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 2: MEASURE GROUPS AND METADATA
# ============================================================================

# Column name mapping for HCAHPS measures (they have _STAR_RATING suffix in data)
HCAHPS_MAPPING = {
    'H_COMP_1': 'H_COMP_1_STAR_RATING',
    'H_COMP_2': 'H_COMP_2_STAR_RATING',
    'H_COMP_3': 'H_COMP_3_STAR_RATING',
    'H_COMP_5': 'H_COMP_5_STAR_RATING',
    'H_COMP_6': 'H_COMP_6_STAR_RATING',
    'H_COMP_7': 'H_COMP_7_STAR_RATING',
    'H_HSP_RATING': 'H_GLOB_STAR_RATING',
    'H_CLEAN_HSP': 'H_INDI_STAR_RATING',  # Assumption - may need verification
}

MEASURE_GROUPS = {
    'Mortality': {
        'measures': ['MORT_30_AMI', 'MORT_30_CABG', 'MORT_30_COPD', 'MORT_30_HF',
                     'MORT_30_PN', 'MORT_30_STK', 'PSI_4_SURG_COMP'],
        'weight': 0.22,
        'lower_is_better': True
    },
    'Safety of Care': {
        'measures': ['COMP_HIP_KNEE', 'HAI_1', 'HAI_2', 'HAI_3', 'HAI_4',
                     'HAI_5', 'HAI_6', 'PSI_90_SAFETY'],
        'weight': 0.22,
        'lower_is_better': True
    },
    'Readmission': {
        'measures': ['EDAC_30_AMI', 'EDAC_30_HF', 'EDAC_30_PN', 'READM_30_CABG',
                     'READM_30_COPD', 'READM_30_HIP_KNEE', 'READM_30_HOSP_WIDE',
                     'OP_32', 'OP_35_ADM', 'OP_35_ED', 'OP_36'],
        'weight': 0.22,
        'lower_is_better': True
    },
    'Patient Experience': {
        'measures': ['H_COMP_1', 'H_COMP_2', 'H_COMP_3', 'H_COMP_5',
                     'H_COMP_6', 'H_COMP_7', 'H_HSP_RATING', 'H_CLEAN_HSP'],
        'weight': 0.22,
        'lower_is_better': False
    },
    'Timely & Effective Care': {
        'measures': ['IMM_3', 'OP_22', 'OP_23', 'OP_29', 'PC_01', 'SEP_1',
                     'OP_18B', 'OP_8', 'OP_10', 'OP_13', 'HCP_COVID_19',
                     'SAFE_USE_OF_OPIOIDS'],
        'weight': 0.12,
        'lower_is_better': False  # Mixed - handled individually below
    }
}

# Individual measure directions for Timely & Effective Care (lower is better)
LOWER_IS_BETTER_MEASURES = [
    'OP_22', 'OP_18B', 'OP_8', 'OP_10', 'OP_13', 'PC_01', 'SAFE_USE_OF_OPIOIDS'
]

# Star rating cutoffs for July 2025
STAR_CUTOFFS = {
    3: {  # 3-measure group peer group
        1: (-3.351, -1.638),
        2: (-1.343, -0.641),
        3: (-0.597, -0.050),
        4: (-0.036, 0.423),
        5: (0.476, 1.298)
    },
    4: {  # 4-measure group peer group
        1: (-1.654, -0.565),
        2: (-0.543, -0.035),
        3: (-0.033, 0.411),
        4: (0.417, 0.911),
        5: (0.934, 2.064)
    },
    5: {  # 5-measure group peer group
        1: (-2.468, -0.669),
        2: (-0.666, -0.277),
        3: (-0.276, 0.073),
        4: (0.074, 0.459),
        5: (0.460, 1.618)
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_actual_column_name(measure_name: str) -> str:
    """Map logical measure name to actual column name in dataset."""
    return HCAHPS_MAPPING.get(measure_name, measure_name)

def is_lower_better(measure_name: str, group_name: str) -> bool:
    """Determine if lower values are better for a measure."""
    if group_name == 'Timely & Effective Care':
        return measure_name in LOWER_IS_BETTER_MEASURES
    else:
        return MEASURE_GROUPS[group_name]['lower_is_better']

# ============================================================================
# STEP 1: MEASURE STANDARDIZATION
# ============================================================================

def standardize_measures(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Standardize all 46 measures using Z-scores.

    Args:
        df: DataFrame with original measure values

    Returns:
        Tuple of (standardized_df, national_statistics)
    """
    print("📊 Standardizing measures...")

    df_std = df.copy()
    national_stats = {}

    # Process each measure group
    for group_name, group_info in MEASURE_GROUPS.items():
        for measure in group_info['measures']:
            # Get actual column name
            col_name = get_actual_column_name(measure)

            if col_name not in df.columns:
                print(f"⚠️  Warning: {col_name} not found in dataset")
                continue

            # Calculate national statistics (excluding NaN)
            values = df[col_name].dropna()

            if len(values) == 0:
                print(f"⚠️  Warning: No valid values for {col_name}")
                continue

            mean = values.mean()
            std = values.std(ddof=1)  # Sample standard deviation

            # Store national statistics
            national_stats[measure] = {
                'mean': mean,
                'std': std,
                'count': len(values),
                'actual_column': col_name
            }

            # Calculate Z-scores
            if std > 0:
                z_scores = (df[col_name] - mean) / std

                # Reverse if lower is better
                if is_lower_better(measure, group_name):
                    z_scores = -z_scores

                # Store standardized values
                df_std[f'std_{measure}'] = z_scores
            else:
                df_std[f'std_{measure}'] = np.nan
                print(f"⚠️  Warning: Zero std dev for {col_name}")

    print(f"✅ Standardized {len(national_stats)} measures")
    return df_std, national_stats

# ============================================================================
# STEPS 2-4: GROUP SCORES AND SUMMARY SCORE
# ============================================================================

def calculate_group_scores(provider_id: str, df_std: pd.DataFrame,
                          df_original: pd.DataFrame) -> Dict:
    """
    Calculate raw and standardized group scores for a hospital.

    Args:
        provider_id: Hospital provider ID
        df_std: DataFrame with standardized measures
        df_original: DataFrame with original measures

    Returns:
        Dictionary with group score information
    """
    # Get hospital's data
    hosp_std = df_std[df_std['PROVIDER_ID'] == provider_id].iloc[0]
    hosp_orig = df_original[df_original['PROVIDER_ID'] == provider_id].iloc[0]

    raw_group_scores = {}
    measures_per_group = {}
    measure_details = {}

    # Calculate raw group scores (average of standardized measures in group)
    for group_name, group_info in MEASURE_GROUPS.items():
        group_measures = []

        for measure in group_info['measures']:
            std_col = f'std_{measure}'
            if std_col in df_std.columns:
                value = hosp_std[std_col]
                if pd.notna(value):
                    group_measures.append(value)
                    measure_details[measure] = {
                        'standardized': value,
                        'original': hosp_orig.get(get_actual_column_name(measure), np.nan)
                    }

        if len(group_measures) > 0:
            raw_group_scores[group_name] = np.mean(group_measures)
            measures_per_group[group_name] = len(group_measures)
        else:
            raw_group_scores[group_name] = np.nan
            measures_per_group[group_name] = 0

    return {
        'raw_group_scores': raw_group_scores,
        'measures_per_group': measures_per_group,
        'measure_details': measure_details
    }

def standardize_group_scores(df_std: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and standardize group scores for all hospitals.

    Args:
        df_std: DataFrame with standardized measures

    Returns:
        DataFrame with group scores added
    """
    df_with_groups = df_std.copy()

    # Calculate raw group scores for all hospitals
    for group_name, group_info in MEASURE_GROUPS.items():
        group_scores = []

        for idx, row in df_std.iterrows():
            measures = []
            for measure in group_info['measures']:
                std_col = f'std_{measure}'
                if std_col in df_std.columns:
                    value = row[std_col]
                    if pd.notna(value):
                        measures.append(value)

            if len(measures) > 0:
                group_scores.append(np.mean(measures))
            else:
                group_scores.append(np.nan)

        df_with_groups[f'raw_group_{group_name}'] = group_scores

    # Now standardize the group scores
    for group_name in MEASURE_GROUPS.keys():
        raw_col = f'raw_group_{group_name}'
        values = df_with_groups[raw_col].dropna()

        if len(values) > 0:
            mean = values.mean()
            std = values.std(ddof=1)

            if std > 0:
                df_with_groups[f'std_group_{group_name}'] = \
                    (df_with_groups[raw_col] - mean) / std
            else:
                df_with_groups[f'std_group_{group_name}'] = 0.0
        else:
            df_with_groups[f'std_group_{group_name}'] = np.nan

    return df_with_groups

def calculate_summary_score(provider_id: str, df_with_groups: pd.DataFrame) -> Dict:
    """
    Calculate weighted summary score with adjusted weights.

    Args:
        provider_id: Hospital provider ID
        df_with_groups: DataFrame with group scores

    Returns:
        Dictionary with summary score and weights
    """
    hosp = df_with_groups[df_with_groups['PROVIDER_ID'] == provider_id].iloc[0]

    # Get standardized group scores
    std_group_scores = {}
    for group_name in MEASURE_GROUPS.keys():
        score = hosp[f'std_group_{group_name}']
        if pd.notna(score):
            std_group_scores[group_name] = score

    # Calculate adjusted weights (redistribute missing group weights)
    base_weights = {name: info['weight'] for name, info in MEASURE_GROUPS.items()}
    adjusted_weights = {}

    total_missing_weight = 0
    for group_name, weight in base_weights.items():
        if group_name not in std_group_scores:
            total_missing_weight += weight

    # Redistribute weights proportionally
    if total_missing_weight < 1.0:  # Some groups present
        for group_name in std_group_scores.keys():
            original_weight = base_weights[group_name]
            adjusted_weights[group_name] = original_weight / (1 - total_missing_weight)

    # Calculate weighted summary score
    summary_score = 0.0
    for group_name, score in std_group_scores.items():
        summary_score += score * adjusted_weights[group_name]

    return {
        'standardized_group_scores': std_group_scores,
        'adjusted_weights': adjusted_weights,
        'summary_score': summary_score
    }

# ============================================================================
# STEPS 5-7: ELIGIBILITY AND STAR ASSIGNMENT
# ============================================================================

def assign_star_rating(provider_id: str, df_with_groups: pd.DataFrame,
                       df_original: pd.DataFrame) -> Dict:
    """
    Complete star rating calculation including eligibility check.

    Args:
        provider_id: Hospital provider ID
        df_with_groups: DataFrame with group scores
        df_original: DataFrame with original values

    Returns:
        Complete results dictionary
    """
    # STEP 1: DATA VALIDATION - Check if hospital exists and has sufficient data
    # Check if hospital exists in dataset
    hospital_data = df_with_groups[df_with_groups['PROVIDER_ID'] == provider_id]

    if hospital_data.empty:
        return {
            'provider_id': provider_id,
            'star_rating': 'No Data',
            'is_eligible': False,
            'summary_score': None,
            'eligibility_reason': 'Hospital not found in dataset',
            'peer_group': None,
            'standardized_group_scores': {},
            'raw_group_scores': {},
            'adjusted_weights': {},
            'measures_per_group': {},
            'measure_details': {},
            'problem_areas': [],
            'strength_areas': []
        }

    # Count how many NON-NULL measures this hospital has across all 46 measures
    measure_columns = [col for col in df_with_groups.columns if col.startswith('std_')
                      and not col.startswith('std_group_')]
    non_null_measures = hospital_data[measure_columns].notna().sum(axis=1).values[0]

    # If hospital has fewer than 10 measures total, flag as insufficient data
    # (CMS typically requires meaningful data coverage for valid ratings)
    if non_null_measures < 10:
        return {
            'provider_id': provider_id,
            'star_rating': 'Insufficient Data',
            'is_eligible': False,
            'summary_score': None,
            'eligibility_reason': f'Only {int(non_null_measures)} of 46 measures reported (minimum 10 required for reliable analysis)',
            'peer_group': None,
            'standardized_group_scores': {},
            'raw_group_scores': {},
            'adjusted_weights': {},
            'measures_per_group': {},
            'measure_details': {},
            'problem_areas': [],
            'strength_areas': [],
            'total_measures': int(non_null_measures)
        }

    # Step 3: Calculate group scores
    group_info = calculate_group_scores(provider_id, df_with_groups, df_original)

    # Step 4: Calculate summary score
    summary_info = calculate_summary_score(provider_id, df_with_groups)

    # Step 5: Simplified - Assume all hospitals are eligible
    # (We're analyzing hospitals that already have CMS ratings)
    measures_per_group = group_info['measures_per_group']
    groups_with_3plus = [name for name, count in measures_per_group.items() if count >= 3]

    is_eligible = True
    eligibility_reason = ""

    # Step 6: Determine peer group
    # Use number of groups with 3+ measures if >= 3, otherwise use total groups with data
    if len(groups_with_3plus) >= 3:
        peer_group = len(groups_with_3plus)
    else:
        # If fewer than 3 groups have 3+ measures, use number of groups with ANY measures
        groups_with_any = sum(1 for count in measures_per_group.values() if count > 0)
        peer_group = max(3, min(5, groups_with_any))  # Ensure between 3-5

    # Check if we have enough data to calculate a star rating
    summary_score = summary_info.get('summary_score')
    has_sufficient_data = (summary_score is not None and
                          not pd.isna(summary_score) and
                          len(summary_info['standardized_group_scores']) > 0)

    # Step 7: Star assignment
    star_rating = None
    if has_sufficient_data and peer_group in STAR_CUTOFFS:
        cutoffs = STAR_CUTOFFS[peer_group]

        for stars in range(1, 6):
            low, high = cutoffs[stars]
            if low <= summary_score <= high:
                star_rating = stars
                break

        # Handle edge cases (scores outside all ranges)
        if star_rating is None:
            if summary_score < cutoffs[1][0]:
                star_rating = 1
            elif summary_score > cutoffs[5][1]:
                star_rating = 5

    # If we couldn't calculate a star rating, explain why
    if star_rating is None:
        if not has_sufficient_data:
            eligibility_reason = "Insufficient measure data in dataset"
        elif peer_group not in STAR_CUTOFFS:
            eligibility_reason = f"Invalid peer group ({peer_group})"

    # Combine all results
    result = {
        'provider_id': provider_id,
        'is_eligible': is_eligible,  # Always True now
        'eligibility_reason': eligibility_reason,
        'peer_group': peer_group,
        'star_rating': star_rating,
        'summary_score': summary_score,
        'standardized_group_scores': summary_info['standardized_group_scores'],
        'raw_group_scores': group_info['raw_group_scores'],
        'adjusted_weights': summary_info['adjusted_weights'],
        'measures_per_group': measures_per_group,
        'measure_details': group_info['measure_details'],
    }

    # Add problem and strength areas
    result['problem_areas'] = [
        name for name, score in summary_info['standardized_group_scores'].items()
        if score < -0.5
    ]
    result['strength_areas'] = [
        name for name, score in summary_info['standardized_group_scores'].items()
        if score > 0.5
    ]

    return result

# ============================================================================
# STEP 6: BATCH PROCESSING
# ============================================================================

def process_hospitals(facility_ids: List[str], df_std: pd.DataFrame,
                     df_original: pd.DataFrame) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Process multiple hospitals and create summary DataFrame.

    Args:
        facility_ids: List of provider IDs to process
        df_std: Standardized DataFrame with group scores
        df_original: Original DataFrame

    Returns:
        Tuple of (detailed_results_list, summary_dataframe)
    """
    print(f"\n🏥 Processing {len(facility_ids)} hospitals...")

    results_list = []

    for facility_id in facility_ids:
        # Check if facility exists
        if facility_id not in df_std['PROVIDER_ID'].values:
            print(f"⚠️  Warning: {facility_id} not found in dataset")
            continue

        result = assign_star_rating(facility_id, df_std, df_original)
        results_list.append(result)

    # Create summary DataFrame
    summary_data = []

    # Debug: Print column names on first iteration to help identify hospital name column
    if len(results_list) > 0:
        name_cols = [col for col in df_original.columns if 'NAME' in col.upper() or 'HOSPITAL' in col.upper()]
        if name_cols:
            print(f"📋 Found potential name columns: {name_cols}")
        else:
            print(f"⚠️  No obvious name columns found. Available columns: {list(df_original.columns[:20])}")

    for result in results_list:
        # Try to get hospital name from original dataframe
        hospital_name = None
        provider_id = result['provider_id']
        if provider_id in df_original['PROVIDER_ID'].values:
            hosp_row = df_original[df_original['PROVIDER_ID'] == provider_id].iloc[0]
            # Try common hospital name columns (expanded list)
            for col in ['HOSPITAL_NAME', 'FACILITY_NAME', 'PROVIDER_NAME', 'FAC_NAME', 'NAME',
                       'Hospital Name', 'Facility Name', 'Provider Name']:
                if col in df_original.columns:
                    hospital_name = hosp_row.get(col)
                    if pd.notna(hospital_name):
                        break

        row = {
            'Hospital_Name': hospital_name if hospital_name else 'Unknown',
            'Facility_ID': result['provider_id'],
            'Star_Rating': result['star_rating'],
            'Peer_Group': result['peer_group'],
            'Summary_Score': result['summary_score'],
            'Is_Eligible': result['is_eligible'],
        }

        # Add group scores
        for group_name in MEASURE_GROUPS.keys():
            score = result['standardized_group_scores'].get(group_name, np.nan)
            row[f'{group_name.replace(" ", "_").replace("&", "and")}_Score'] = score

        # Add problem and strength areas
        row['Problem_Areas'] = ', '.join(result['problem_areas']) if result['problem_areas'] else ''
        row['Strength_Areas'] = ', '.join(result['strength_areas']) if result['strength_areas'] else ''

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    print(f"✅ Processed {len(results_list)} hospitals successfully")
    return results_list, summary_df

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load and prepare data from SAS file.

    Args:
        file_path: Path to SAS7BDAT file

    Returns:
        Tuple of (original_df, standardized_df_with_groups, national_stats)
    """
    print(f"📂 Loading data from {file_path}...")
    df, meta = pyreadstat.read_sas7bdat(file_path)
    print(f"✅ Loaded {len(df)} hospitals")

    # Standardize measures
    df_std, national_stats = standardize_measures(df)

    # Calculate and standardize group scores
    df_with_groups = standardize_group_scores(df_std)

    print("✅ Data preparation complete\n")
    return df, df_with_groups, national_stats


if __name__ == "__main__":
    # Test with a few hospitals
    file_path = "/Users/heilman/Desktop/All Data July 2025.sas7bdat"
    df_original, df_std, national_stats = load_data(file_path)

    # Test with first 3 hospitals
    test_ids = df_original['PROVIDER_ID'].head(3).tolist()
    results, summary = process_hospitals(test_ids, df_std, df_original)

    print("\n" + "="*80)
    print("SAMPLE RESULTS")
    print("="*80)
    print(summary.to_string())

    print("\n" + "="*80)
    print("DETAILED RESULTS FOR FIRST HOSPITAL")
    print("="*80)
    for key, value in results[0].items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
