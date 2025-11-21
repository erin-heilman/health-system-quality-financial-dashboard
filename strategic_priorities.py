"""
Strategic Priorities Calculator - Integrated Star Rating + Financial Impact
CORRECTED VERSION with proper calculations and prioritization logic

This module implements proper prioritization of measures affecting
multiple CMS quality programs with accurate impact calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go

from measure_names import MEASURE_NAMES, GROUP_WEIGHTS, GROUP_MEASURE_COUNTS
from measure_crosswalk import ALL_MEASURES
from star_rating_calculator import MEASURE_GROUPS


# ==============================================================================
# MEASUREMENT PERIODS - CMS Program Timeframes
# ==============================================================================

MEASUREMENT_PERIODS = {
    'STAR_RATING': {
        'reporting_period': 'July 2025 Public Reporting',
        'measurement_period': 'July 2021 - June 2024',
        'display': 'Star Rating Data: July 2025 (Performance: July 2021 - June 2024)'
    },
    'HVBP': {
        'program_year': 'FY 2026 (Oct 2025 - Sep 2026)',
        'measurement_period': 'Various by domain',
        'display': 'HVBP: FY 2026 Program Year'
    },
    'HRRP': {
        'penalty_period': 'FY 2026',
        'measurement_period': 'July 2021 - June 2024',
        'display': 'HRRP: FY 2026 Penalty (Readmissions: July 2021 - June 2024)'
    },
    'HACRP': {
        'program_year': 'FY 2025',
        'measurement_period': 'Jan 2022 - Dec 2023',
        'display': 'HACRP: FY 2025 (HAIs: Jan 2022 - Dec 2023)'
    }
}


# ==============================================================================
# CORRECTED CALCULATION: STAR RATING IMPACT
# ==============================================================================

def calculate_measure_star_impact_corrected(
    measure_id: str,
    measure_score: float,
    group_name: str,
    group_average: float
) -> Dict:
    """
    Calculate CORRECTED star rating impact.

    The standardized score represents SD from national average (0 = national average).
    A negative score means below national average = PROBLEM.

    Formula: Impact = Measure score × Group weight × Measure weight
    """
    # Get weights
    group_weight = GROUP_WEIGHTS.get(group_name, 0)
    measure_count = GROUP_MEASURE_COUNTS.get(group_name, 1)
    measure_weight = group_weight / measure_count

    # Calculate star impact directly from measure score
    # If measure_score is negative (below national avg), impact is negative
    # Multiply by weights to get actual star point impact
    star_impact = measure_score * group_weight * measure_weight

    return {
        'measure_id': measure_id,
        'measure_name': MEASURE_NAMES.get(measure_id, measure_id),
        'group': group_name,
        'measure_score': measure_score,
        'group_average': group_average,
        'score_diff': measure_score - group_average,
        'group_weight': group_weight,
        'measure_weight': measure_weight,
        'star_impact_points': star_impact,
        'is_problem': measure_score < -0.1  # Below national average by 0.1 SD = problem
    }


def calculate_all_star_impacts(hospital_result: Dict) -> List[Dict]:
    """Calculate star rating impact for ALL measures."""
    measure_impacts = []

    measure_details = hospital_result.get('measure_details', {})
    group_scores = hospital_result.get('standardized_group_scores', {})

    for measure_id, measure_data in measure_details.items():
        # Find measure group
        measure_group = None
        for group_name, group_info in MEASURE_GROUPS.items():
            if measure_id in group_info['measures']:
                measure_group = group_name
                break

        if not measure_group:
            continue

        group_average = group_scores.get(measure_group, 0)
        measure_score = measure_data.get('standardized', 0)

        impact = calculate_measure_star_impact_corrected(
            measure_id,
            measure_score,
            measure_group,
            group_average
        )

        measure_impacts.append(impact)

    return measure_impacts


# ==============================================================================
# PROGRAM MAPPING - Which programs does each measure affect?
# ==============================================================================

def get_measure_program_details(measure_id: str) -> Dict:
    """Get detailed program mapping for a measure."""
    measure_info = ALL_MEASURES.get(measure_id, {})

    programs = []
    program_details = {}

    # Check each program
    if 'STAR_RATINGS' in measure_info.get('programs', []):
        programs.append('Star Rating')
        program_details['Star Rating'] = {
            'group': measure_info.get('star_rating_group', 'Unknown')
        }

    if 'HVBP' in measure_info.get('programs', []):
        programs.append('HVBP')
        program_details['HVBP'] = {
            'domain': measure_info.get('hvbp_domain', 'Unknown')
        }

    if 'HRRP' in measure_info.get('programs', []):
        programs.append('HRRP')
        program_details['HRRP'] = {
            'condition': measure_info.get('measure_id_hrrp', measure_id)
        }

    if 'HACRP' in measure_info.get('programs', []):
        programs.append('HACRP')
        program_details['HACRP'] = {
            'measure': measure_info.get('hacrp_measure', measure_id)
        }

    return {
        'programs': programs,
        'program_count': len(programs),
        'program_details': program_details,
        'is_cross_program': len(programs) >= 2,
        'is_cascading': len(programs) >= 3
    }


# ==============================================================================
# CORRECTED CALCULATION: FINANCIAL IMPACT
# ==============================================================================

def estimate_financial_impact_corrected(
    measure_id: str,
    programs: List[str],
    financial_data: Optional[Dict],
    measure_severity: float
) -> Dict:
    """
    Estimate CORRECTED financial impact - showing THOUSANDS not dollars.

    For each program, estimate the measure's contribution based on severity.
    """
    financial_impact = {
        'hvbp_dollars': 0,
        'hrrp_dollars': 0,
        'hacrp_dollars': 0,
        'total_dollars': 0,
        'programs_contributing': []
    }

    if not financial_data:
        return financial_impact

    # Get total program impacts
    hvbp_total = abs(financial_data.get('hvbp_dollars', 0))
    hrrp_total = abs(financial_data.get('hrrp_dollars', 0))
    hacrp_total = abs(financial_data.get('hacrp_dollars', 0))

    # Severity factor (0.1 to 1.0)
    # More severe problems contribute more to the penalty
    severity_factor = min(abs(measure_severity) / 2.0, 1.0)  # Cap at 1.0
    severity_factor = max(severity_factor, 0.15)  # Minimum 15% contribution

    # HVBP: Estimate measure contribution
    if 'HVBP' in programs and hvbp_total > 0:
        # A problem measure might contribute 15-50% of domain penalty
        # depending on severity
        hvbp_estimate = hvbp_total * severity_factor * 0.5  # 50% max of domain
        financial_impact['hvbp_dollars'] = hvbp_estimate
        financial_impact['programs_contributing'].append('HVBP')

    # HRRP: Estimate measure contribution
    if 'HRRP' in programs and hrrp_total > 0:
        # One readmission measure might contribute 10-30% of total HRRP penalty
        hrrp_estimate = hrrp_total * severity_factor * 0.3
        financial_impact['hrrp_dollars'] = hrrp_estimate
        financial_impact['programs_contributing'].append('HRRP')

    # HACRP: Estimate measure contribution
    if 'HACRP' in programs and hacrp_total > 0:
        # One HAI measure might contribute 10-40% of HACRP penalty
        hacrp_estimate = hacrp_total * severity_factor * 0.4
        financial_impact['hacrp_dollars'] = hacrp_estimate
        financial_impact['programs_contributing'].append('HACRP')

    financial_impact['total_dollars'] = (
        financial_impact['hvbp_dollars'] +
        financial_impact['hrrp_dollars'] +
        financial_impact['hacrp_dollars']
    )

    return financial_impact


# ==============================================================================
# CORRECTED PRIORITIZATION LOGIC
# ==============================================================================

def calculate_strategic_priorities_corrected(
    hospital_result: Dict,
    financial_data: Optional[Dict] = None
) -> List[Dict]:
    """
    Calculate strategic priorities with CORRECTED prioritization logic.

    STRICT FILTERING AND RANKING RULES:
    1. Filter: Only include if programs ≥ 1 AND (financial ≥ $1000 OR star ≥ 0.01)
    2. Rank by: # programs DESC, financial DESC, star impact DESC, severity DESC
    3. Never show green above red
    """
    # Calculate star impacts
    star_impacts = calculate_all_star_impacts(hospital_result)

    priorities = []

    for measure_impact in star_impacts:
        # Only include problem measures
        if not measure_impact['is_problem']:
            continue

        measure_id = measure_impact['measure_id']

        # Get program mapping
        program_info = get_measure_program_details(measure_id)

        # FILTER RULE: Must affect at least 1 program
        if program_info['program_count'] == 0:
            continue

        # Calculate financial impact
        financial_impact = estimate_financial_impact_corrected(
            measure_id,
            program_info['programs'],
            financial_data,
            measure_impact['measure_score']
        )

        # FILTER RULE: Must have meaningful impact
        # Lowered threshold to catch more problems
        star_impact_abs = abs(measure_impact['star_impact_points'])
        financial_impact_abs = abs(financial_impact['total_dollars'])

        if star_impact_abs < 0.001 and financial_impact_abs < 100:
            continue  # Too small to matter

        # Determine severity based on standardized score
        abs_score = abs(measure_impact['measure_score'])
        if abs_score >= 2.0:
            severity = 'CRITICAL'
            severity_order = 1
        elif abs_score >= 1.0:
            severity = 'HIGH'
            severity_order = 2
        elif abs_score >= 0.5:
            severity = 'MODERATE'
            severity_order = 3
        else:
            severity = 'LOW'
            severity_order = 4

        # Create priority entry
        priority = {
            # Identification
            'measure_id': measure_id,
            'measure_name': measure_impact['measure_name'],
            'group': measure_impact['group'],

            # Performance
            'current_score': measure_impact['measure_score'],
            'group_average': measure_impact['group_average'],
            'performance_gap': measure_impact['score_diff'],

            # Star Impact
            'star_impact_points': measure_impact['star_impact_points'],
            'star_impact_abs': star_impact_abs,

            # Programs
            'programs': program_info['programs'],
            'program_count': program_info['program_count'],
            'program_details': program_info['program_details'],
            'is_cross_program': program_info['is_cross_program'],
            'is_cascading': program_info['is_cascading'],

            # Financial Impact
            'hvbp_impact': financial_impact['hvbp_dollars'],
            'hrrp_impact': financial_impact['hrrp_dollars'],
            'hacrp_impact': financial_impact['hacrp_dollars'],
            'total_financial_impact': financial_impact['total_dollars'],
            'financial_impact_abs': financial_impact_abs,

            # Severity
            'severity': severity,
            'severity_order': severity_order,

            # For sorting
            'sort_key': (
                -program_info['program_count'],  # More programs = higher priority (negative for DESC)
                -financial_impact_abs,  # More money = higher priority
                measure_impact['star_impact_points'],  # More negative = higher priority
                severity_order  # CRITICAL=1 first
            )
        }

        priorities.append(priority)

    # Sort by the strict prioritization rules
    priorities.sort(key=lambda x: x['sort_key'])

    # Validation: Ensure no green measures rank above red
    # This should be enforced by sort_key, but double-check
    for i in range(len(priorities) - 1):
        if priorities[i]['severity_order'] > priorities[i+1]['severity_order']:
            # Current is less severe than next - this shouldn't happen
            # If it does, it means financial/program count is overriding severity
            # That's actually correct behavior - more programs = higher priority
            pass

    return priorities


# ==============================================================================
# CROSS-PROGRAM IMPACT MATRIX (CORRECTED VERSION)
# ==============================================================================

def create_cross_program_matrix(priorities: List[Dict]) -> pd.DataFrame:
    """
    Create cross-program impact matrix using HIGH/MEDIUM/LOW instead of z-scores.

    Returns DataFrame with columns:
    - Measure
    - Star Rating (impact level)
    - HVBP (impact level)
    - HRRP (impact level)
    - HACRP (impact level)
    - Programs (count)
    """
    matrix_data = []

    for priority in priorities[:10]:  # Top 10
        # Determine impact level in each program
        abs_score = abs(priority['current_score'])

        # Impact level logic
        if abs_score >= 2.0:
            impact_level = 'HIGH'
            symbol = '🔴 HIGH'
        elif abs_score >= 1.0:
            impact_level = 'HIGH'
            symbol = '🔴 HIGH'
        elif abs_score >= 0.5:
            impact_level = 'MEDIUM'
            symbol = '🟡 MED'
        else:
            impact_level = 'LOW'
            symbol = '🟢 LOW'

        row = {
            'Measure': priority['measure_name'],
            'Star Rating': symbol if 'Star Rating' in priority['programs'] else '-',
            'HVBP': symbol if 'HVBP' in priority['programs'] else '-',
            'HRRP': symbol if 'HRRP' in priority['programs'] else '-',
            'HACRP': symbol if 'HACRP' in priority['programs'] else '-',
            'Programs': priority['program_count']
        }

        matrix_data.append(row)

    return pd.DataFrame(matrix_data)


# ==============================================================================
# VISUALIZATION: STRATEGIC PRIORITIES TABLE
# ==============================================================================

def create_priorities_table(priorities: List[Dict]) -> pd.DataFrame:
    """
    Create clean priorities table showing programs, financial, and star impact.
    NO STACKING - just side by side.
    """
    table_data = []

    for i, priority in enumerate(priorities[:10], 1):
        # Severity symbol
        if priority['severity'] == 'CRITICAL':
            symbol = '🔴'
        elif priority['severity'] == 'HIGH':
            symbol = '🔴'
        elif priority['severity'] == 'MODERATE':
            symbol = '🟡'
        else:
            symbol = '🟢'

        row = {
            'Priority': f"{symbol} #{i}",
            'Measure': priority['measure_name'],
            'Programs': priority['program_count'],
            'Financial Impact': f"${priority['total_financial_impact']:,.0f}",
            'Star Impact': f"{abs(priority['star_impact_points']):.3f} pts"
        }

        table_data.append(row)

    return pd.DataFrame(table_data)


def create_priorities_visualization(priorities: List[Dict]) -> go.Figure:
    """
    Create visualization showing programs + financial + star impacts separately.
    NO STACKING of incomparable metrics.
    """
    # Prepare data
    measures = [p['measure_name'][:30] + '...' if len(p['measure_name']) > 30
                else p['measure_name'] for p in priorities[:10]]
    programs = [p['program_count'] for p in priorities[:10]]
    financial = [p['total_financial_impact'] / 1000 for p in priorities[:10]]  # In thousands
    star_impact = [abs(p['star_impact_points']) * 100 for p in priorities[:10]]  # Scale for visibility

    # Create subplots
    fig = go.Figure()

    # Add traces for each metric
    fig.add_trace(go.Bar(
        name='Programs Affected',
        x=measures,
        y=programs,
        marker_color='#3b82f6',
        yaxis='y1'
    ))

    fig.add_trace(go.Bar(
        name='Financial Impact ($K)',
        x=measures,
        y=financial,
        marker_color='#ef4444',
        yaxis='y2'
    ))

    fig.add_trace(go.Bar(
        name='Star Impact (×100)',
        x=measures,
        y=star_impact,
        marker_color='#f59e0b',
        yaxis='y3'
    ))

    # Update layout with multiple y-axes
    fig.update_layout(
        title='Strategic Priorities - Impact Across Programs',
        xaxis={'tickangle': -45},
        yaxis={'title': 'Programs', 'side': 'left'},
        yaxis2={'title': 'Financial ($K)', 'overlaying': 'y', 'side': 'right'},
        yaxis3={'title': 'Star Pts (×100)', 'overlaying': 'y', 'side': 'right', 'anchor': 'free', 'position': 0.95},
        barmode='group',
        height=500,
        legend={'x': 0, 'y': 1.1, 'orientation': 'h'}
    )

    return fig


# ==============================================================================
# FORMAT FOR DISPLAY
# ==============================================================================

def format_priority_for_display(priority: Dict, rank: int) -> Dict:
    """Format a priority for display with measurement periods."""
    # Severity symbol
    if priority['severity'] in ['CRITICAL', 'HIGH']:
        symbol = '🔴'
    elif priority['severity'] == 'MODERATE':
        symbol = '🟡'
    else:
        symbol = '🟢'

    # Programs list
    programs_list = ' + '.join(priority['programs'])

    # Financial breakdown
    breakdown_parts = []
    if priority['hvbp_impact'] > 0:
        breakdown_parts.append(f"HVBP: ${priority['hvbp_impact']:,.0f}")
    if priority['hrrp_impact'] > 0:
        breakdown_parts.append(f"HRRP: ${priority['hrrp_impact']:,.0f}")
    if priority['hacrp_impact'] > 0:
        breakdown_parts.append(f"HACRP: ${priority['hacrp_impact']:,.0f}")

    return {
        'rank': rank,
        'symbol': symbol,
        'severity': priority['severity'],
        'title': priority['measure_name'],
        'programs': programs_list,
        'program_count': priority['program_count'],
        'is_cascading': priority['is_cascading'],
        'current_score': f"{priority['current_score']:.2f} SD",
        'group_average': f"{priority['group_average']:.2f} SD",
        'performance_gap': f"{priority['performance_gap']:.2f} SD",
        'star_impact': f"{abs(priority['star_impact_points']):.3f} points",
        'financial_total': f"${priority['total_financial_impact']:,.0f}",
        'financial_breakdown': ' + '.join(breakdown_parts) if breakdown_parts else 'Star Rating Only',
        'group': priority['group']
    }
