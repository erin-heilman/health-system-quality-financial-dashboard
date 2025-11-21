"""
Financial Impact Calculator: CMS Quality Program Financial Analysis

This module calculates financial impacts from CMS quality programs:
- HVBP (Hospital Value-Based Purchasing)
- HRRP (Hospital Readmissions Reduction Program)
- HACRP (Hospital-Acquired Condition Reduction Program)

Integrates with Star Rating data to provide comprehensive ROI analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# ==============================================================================
# PROGRAM FINANCIAL PARAMETERS
# ==============================================================================

PROGRAM_PARAMS = {
    'HVBP': {
        'max_withhold': 0.02,  # 2% of Medicare payments withheld
        'max_penalty': 0.02,   # Can lose up to 2%
        'max_bonus': 0.02,     # Can gain up to 2%
        'total_at_risk': 0.02  # 2% penalty risk (downside only)
    },
    'HRRP': {
        'max_penalty': 0.03,   # Up to 3% penalty
        'max_bonus': 0.00,     # No bonus, only penalty
        'total_at_risk': 0.03  # 3% at risk
    },
    'HACRP': {
        'max_penalty': 0.01,   # 1% penalty for worst quartile
        'max_bonus': 0.00,     # No bonus, only penalty
        'total_at_risk': 0.01  # 1% at risk
    }
}

# ==============================================================================
# PARSE DEFINITIVE CSV
# ==============================================================================

def parse_definitive_csv(csv_path: str) -> pd.DataFrame:
    """
    Parse Definitive Healthcare CSV with hospital financial data.

    Expected columns:
    - Hospital Name
    - CCN (6-digit CMS Certification Number)
    - Net Medicare Revenue (FY2024)
    - HVBP % (e.g., -0.52%)
    - HVBP $ (e.g., -$234,567)
    - HRRP % (e.g., -1.23%)
    - HRRP $ (e.g., -$567,890)
    - HACRP % (e.g., -1.00%)
    - HACRP $ (e.g., -$123,456)

    Returns:
        DataFrame with cleaned financial data
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Standardize column names (handle variations)
    column_mapping = {
        'Hospital Name': 'hospital_name',
        'CCN': 'facility_id',
        'Net Medicare Revenue (FY2024)': 'net_medicare_revenue',
        'HVBP %': 'hvbp_pct',
        'HVBP $': 'hvbp_dollars',
        'HRRP %': 'hrrp_pct',
        'HRRP $': 'hrrp_dollars',
        'HACRP %': 'hacrp_pct',
        'HACRP $': 'hacrp_dollars'
    }

    df = df.rename(columns=column_mapping)

    # Clean facility_id (ensure 6-digit format)
    df['facility_id'] = df['facility_id'].astype(str).str.zfill(6)

    # Helper function to clean accounting format numbers
    def clean_accounting_format(series):
        """Convert accounting format to standard numbers.
        Handles: $1,234, (1,234), ($1,234), -$1,234, etc.
        """
        # Convert to string first
        cleaned = series.astype(str)

        # Handle parentheses (negative numbers in accounting format)
        # (1234) or ($1,234) becomes -1234
        cleaned = cleaned.str.replace(r'^\((.+)\)$', r'-\1', regex=True)

        # Remove $, commas, and spaces
        cleaned = cleaned.str.replace(r'[\$,\s]', '', regex=True)

        # Convert to float
        return cleaned.astype(float)

    # Clean revenue
    df['net_medicare_revenue'] = clean_accounting_format(df['net_medicare_revenue'])

    # Clean percentages (remove %, convert to decimal, handle parentheses)
    for col in ['hvbp_pct', 'hrrp_pct', 'hacrp_pct']:
        # Handle parentheses for negative percentages
        df[col] = df[col].astype(str).str.replace(r'^\((.+)\)$', r'-\1', regex=True)
        # Remove % and convert
        df[col] = df[col].str.replace(r'%', '', regex=True).astype(float) / 100

    # Clean dollar amounts
    for col in ['hvbp_dollars', 'hrrp_dollars', 'hacrp_dollars']:
        df[col] = clean_accounting_format(df[col])

    return df


# ==============================================================================
# FINANCIAL IMPACT CALCULATIONS
# ==============================================================================

def calculate_total_quality_impact(financial_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total quality program financial impact for each hospital.

    Args:
        financial_data: DataFrame from parse_definitive_csv()

    Returns:
        DataFrame with additional calculated columns
    """
    df = financial_data.copy()

    # Total current impact (penalties/bonuses)
    df['total_current_impact_dollars'] = (
        df['hvbp_dollars'] +
        df['hrrp_dollars'] +
        df['hacrp_dollars']
    )

    df['total_current_impact_pct'] = (
        df['hvbp_pct'] +
        df['hrrp_pct'] +
        df['hacrp_pct']
    )

    # At-risk revenue (maximum possible swing)
    df['hvbp_at_risk'] = df['net_medicare_revenue'] * PROGRAM_PARAMS['HVBP']['total_at_risk']
    df['hrrp_at_risk'] = df['net_medicare_revenue'] * PROGRAM_PARAMS['HRRP']['total_at_risk']
    df['hacrp_at_risk'] = df['net_medicare_revenue'] * PROGRAM_PARAMS['HACRP']['total_at_risk']

    df['total_at_risk'] = (
        df['hvbp_at_risk'] +
        df['hrrp_at_risk'] +
        df['hacrp_at_risk']
    )

    # Opportunity revenue = Penalty recovery + HVBP bonus potential
    # Part 1: Penalty recovery (eliminate current penalties, get back to $0)
    df['hvbp_penalty_recovery'] = df['hvbp_dollars'].apply(lambda x: abs(x) if x < 0 else 0)
    df['hrrp_penalty_recovery'] = df['hrrp_dollars'].apply(lambda x: abs(x) if x < 0 else 0)
    df['hacrp_penalty_recovery'] = df['hacrp_dollars'].apply(lambda x: abs(x) if x < 0 else 0)

    # Part 2: HVBP bonus potential (2% of Medicare revenue if achieve top performance)
    df['hvbp_bonus_potential'] = df['net_medicare_revenue'] * PROGRAM_PARAMS['HVBP']['max_bonus']

    # Total opportunity per program
    df['hvbp_opportunity'] = df['hvbp_penalty_recovery'] + df['hvbp_bonus_potential']
    df['hrrp_opportunity'] = df['hrrp_penalty_recovery']
    df['hacrp_opportunity'] = df['hacrp_penalty_recovery']

    df['total_opportunity'] = (
        df['hvbp_opportunity'] +
        df['hrrp_opportunity'] +
        df['hacrp_opportunity']
    )

    # Penalty/bonus status
    df['hvbp_status'] = df['hvbp_dollars'].apply(
        lambda x: 'BONUS' if x > 0 else 'PENALTY' if x < 0 else 'NEUTRAL'
    )
    df['hrrp_status'] = df['hrrp_dollars'].apply(
        lambda x: 'NO PENALTY' if x == 0 else 'PENALTY'
    )
    df['hacrp_status'] = df['hacrp_dollars'].apply(
        lambda x: 'NO PENALTY' if x == 0 else 'PENALTY'
    )

    return df


def get_hospital_financial_summary(facility_id: str, financial_data: pd.DataFrame) -> Dict:
    """
    Get financial summary for a specific hospital.

    Args:
        facility_id: 6-digit CMS facility ID
        financial_data: DataFrame with calculated financial impacts

    Returns:
        Dictionary with financial summary metrics
    """
    hospital = financial_data[financial_data['facility_id'] == facility_id]

    if hospital.empty:
        return {
            'found': False,
            'facility_id': facility_id,
            'message': 'Hospital not found in financial data'
        }

    row = hospital.iloc[0]

    return {
        'found': True,
        'facility_id': facility_id,
        'hospital_name': row['hospital_name'],
        'net_medicare_revenue': row['net_medicare_revenue'],

        # Current impacts
        'hvbp_current': {
            'dollars': row['hvbp_dollars'],
            'percent': row['hvbp_pct'],
            'status': row['hvbp_status']
        },
        'hrrp_current': {
            'dollars': row['hrrp_dollars'],
            'percent': row['hrrp_pct'],
            'status': row['hrrp_status']
        },
        'hacrp_current': {
            'dollars': row['hacrp_dollars'],
            'percent': row['hacrp_pct'],
            'status': row['hacrp_status']
        },
        'total_current_impact': {
            'dollars': row['total_current_impact_dollars'],
            'percent': row['total_current_impact_pct']
        },

        # At-risk revenue
        'at_risk': {
            'hvbp': row['hvbp_at_risk'],
            'hrrp': row['hrrp_at_risk'],
            'hacrp': row['hacrp_at_risk'],
            'total': row['total_at_risk']
        },

        # Opportunity revenue
        'opportunity': {
            'hvbp': row['hvbp_opportunity'],
            'hrrp': row['hrrp_opportunity'],
            'hacrp': row['hacrp_opportunity'],
            'total': row['total_opportunity']
        }
    }


def get_system_financial_summary(financial_data: pd.DataFrame) -> Dict:
    """
    Get aggregated financial summary for entire health system.

    Args:
        financial_data: DataFrame with calculated financial impacts

    Returns:
        Dictionary with system-wide financial metrics
    """
    return {
        'total_hospitals': len(financial_data),
        'total_medicare_revenue': financial_data['net_medicare_revenue'].sum(),

        # Current impacts
        'current_impact': {
            'hvbp_total': financial_data['hvbp_dollars'].sum(),
            'hrrp_total': financial_data['hrrp_dollars'].sum(),
            'hacrp_total': financial_data['hacrp_dollars'].sum(),
            'total': financial_data['total_current_impact_dollars'].sum(),
            'avg_percent': financial_data['total_current_impact_pct'].mean()
        },

        # Program status counts
        'program_status': {
            'hvbp_penalties': len(financial_data[financial_data['hvbp_status'] == 'PENALTY']),
            'hvbp_bonuses': len(financial_data[financial_data['hvbp_status'] == 'BONUS']),
            'hvbp_neutral': len(financial_data[financial_data['hvbp_status'] == 'NEUTRAL']),
            'hrrp_penalties': len(financial_data[financial_data['hrrp_status'] == 'PENALTY']),
            'hacrp_penalties': len(financial_data[financial_data['hacrp_status'] == 'PENALTY'])
        },

        # At-risk revenue
        'at_risk': {
            'hvbp': financial_data['hvbp_at_risk'].sum(),
            'hrrp': financial_data['hrrp_at_risk'].sum(),
            'hacrp': financial_data['hacrp_at_risk'].sum(),
            'total': financial_data['total_at_risk'].sum()
        },

        # Opportunity revenue
        'opportunity': {
            'hvbp': financial_data['hvbp_opportunity'].sum(),
            'hrrp': financial_data['hrrp_opportunity'].sum(),
            'hacrp': financial_data['hacrp_opportunity'].sum(),
            'total': financial_data['total_opportunity'].sum()
        },

        # Top performers and bottom performers
        'top_performer': financial_data.nlargest(1, 'total_current_impact_dollars').iloc[0].to_dict(),
        'bottom_performer': financial_data.nsmallest(1, 'total_current_impact_dollars').iloc[0].to_dict()
    }


# ==============================================================================
# CROSS-PROGRAM IMPACT INTEGRATION
# ==============================================================================

def calculate_cascading_financial_impact(
    measure_performance: Dict,
    financial_summary: Dict,
    measure_id: str,
    programs: List[str]
) -> Dict:
    """
    Calculate the financial impact of improving a specific measure that affects multiple programs.

    Args:
        measure_performance: Performance data for the measure (from star rating calculator)
        financial_summary: Financial summary from get_hospital_financial_summary()
        measure_id: Measure ID (e.g., 'HAI_5' for MRSA)
        programs: List of programs affected (e.g., ['STAR_RATINGS', 'HVBP', 'HACRP'])

    Returns:
        Dictionary with cascading impact analysis
    """
    impact = {
        'measure_id': measure_id,
        'programs_affected': programs,
        'num_programs': len(programs),
        'total_financial_impact': 0,
        'program_breakdown': {}
    }

    # Calculate financial impact for each program
    for program in programs:
        if program == 'HVBP':
            # HVBP impact: proportion of total HVBP opportunity
            program_impact = financial_summary['opportunity']['hvbp'] / len(programs)
            impact['program_breakdown']['HVBP'] = program_impact
            impact['total_financial_impact'] += program_impact

        elif program == 'HRRP':
            # HRRP impact: proportion of total HRRP opportunity
            program_impact = financial_summary['opportunity']['hrrp'] / len(programs)
            impact['program_breakdown']['HRRP'] = program_impact
            impact['total_financial_impact'] += program_impact

        elif program == 'HACRP':
            # HACRP impact: proportion of total HACRP opportunity
            program_impact = financial_summary['opportunity']['hacrp'] / len(programs)
            impact['program_breakdown']['HACRP'] = program_impact
            impact['total_financial_impact'] += program_impact

    # Add star rating impact (indirect - market share, patient choice)
    if 'STAR_RATINGS' in programs:
        impact['star_rating_impact'] = 'Affects overall star rating (market positioning)'

    return impact


# ==============================================================================
# ROI PRIORITIZATION
# ==============================================================================

def calculate_measure_roi(
    measure_id: str,
    measure_performance: Dict,
    financial_summary: Dict,
    programs: List[str],
    improvement_cost_estimate: float = 50000  # Default: $50k per measure improvement
) -> Dict:
    """
    Calculate ROI for improving a specific measure.

    Args:
        measure_id: Measure ID
        measure_performance: Performance data from star rating calculator
        financial_summary: Financial summary
        programs: Programs affected by this measure
        improvement_cost_estimate: Estimated cost to improve measure

    Returns:
        Dictionary with ROI analysis
    """
    # Calculate potential financial gain
    financial_impact = calculate_cascading_financial_impact(
        measure_performance, financial_summary, measure_id, programs
    )

    potential_gain = financial_impact['total_financial_impact']

    # Calculate ROI
    roi_ratio = potential_gain / improvement_cost_estimate if improvement_cost_estimate > 0 else 0
    roi_percent = (roi_ratio - 1) * 100

    # Payback period (in months)
    payback_months = (improvement_cost_estimate / potential_gain) * 12 if potential_gain > 0 else float('inf')

    return {
        'measure_id': measure_id,
        'programs_affected': programs,
        'num_programs': len(programs),
        'potential_gain': potential_gain,
        'improvement_cost': improvement_cost_estimate,
        'roi_ratio': roi_ratio,
        'roi_percent': roi_percent,
        'payback_months': payback_months,
        'priority_score': roi_ratio * len(programs),  # Weight by number of programs
        'recommendation': 'HIGH PRIORITY' if roi_ratio > 5 else 'MEDIUM PRIORITY' if roi_ratio > 2 else 'LOW PRIORITY'
    }


def prioritize_measures_by_roi(
    problem_measures: List[Dict],
    financial_summary: Dict,
    measure_crosswalk: Dict
) -> pd.DataFrame:
    """
    Prioritize all problem measures by ROI.

    Args:
        problem_measures: List of problem measures from star rating analysis
        financial_summary: Financial summary
        measure_crosswalk: Measure crosswalk data

    Returns:
        DataFrame with ROI-prioritized measures
    """
    roi_results = []

    for measure in problem_measures:
        measure_id = measure['measure_id']

        # Get programs affected from crosswalk
        programs = measure_crosswalk.get(measure_id, {}).get('programs', ['STAR_RATINGS'])

        # Calculate ROI
        roi = calculate_measure_roi(
            measure_id,
            measure,
            financial_summary,
            programs
        )

        # Combine with measure performance data
        roi_results.append({
            'measure_id': measure_id,
            'measure_name': measure.get('name', measure_id),
            'current_performance': measure.get('score', 0),
            'programs_affected': ', '.join(programs),
            'num_programs': len(programs),
            'potential_financial_gain': roi['potential_gain'],
            'roi_ratio': roi['roi_ratio'],
            'roi_percent': roi['roi_percent'],
            'payback_months': roi['payback_months'],
            'priority_score': roi['priority_score'],
            'recommendation': roi['recommendation']
        })

    # Convert to DataFrame and sort by priority score
    df = pd.DataFrame(roi_results)
    df = df.sort_values('priority_score', ascending=False)

    return df


# ==============================================================================
# EXPORT FUNCTIONS
# ==============================================================================

def format_currency(value: float) -> str:
    """Format value as currency with proper sign."""
    if value >= 0:
        return f"${value:,.0f}"
    else:
        return f"-${abs(value):,.0f}"


def format_percent(value: float) -> str:
    """Format value as percentage with proper sign."""
    if value >= 0:
        return f"+{value*100:.2f}%"
    else:
        return f"{value*100:.2f}%"


def create_financial_summary_table(financial_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create formatted table for display/export.

    Returns:
        DataFrame formatted for display
    """
    summary = financial_data.copy()

    return pd.DataFrame({
        'Hospital': summary['hospital_name'],
        'Facility ID': summary['facility_id'],
        'Medicare Revenue': summary['net_medicare_revenue'].apply(format_currency),
        'HVBP Impact': summary['hvbp_dollars'].apply(format_currency),
        'HRRP Impact': summary['hrrp_dollars'].apply(format_currency),
        'HACRP Impact': summary['hacrp_dollars'].apply(format_currency),
        'Total Impact': summary['total_current_impact_dollars'].apply(format_currency),
        'Total At Risk': summary['total_at_risk'].apply(format_currency),
        'Opportunity': summary['total_opportunity'].apply(format_currency)
    })
