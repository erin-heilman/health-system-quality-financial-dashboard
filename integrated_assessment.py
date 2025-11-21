"""
Integrated Quality & Financial Risk Assessment

This module combines:
- CMS Star Ratings performance
- Financial impacts (HVBP, HRRP, HACRP)
- Cross-program measure mapping
- ROI-based prioritization

Answers C-suite questions:
1. What is our total financial exposure across all quality programs?
2. Which quality problems are costing us the most money?
3. Which quality problems affect both our star rating AND our financials?
4. Where should we invest our quality improvement resources for maximum ROI?
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from measure_crosswalk import (
    ALL_MEASURES,
    get_cross_program_measures,
    get_cascading_impact_measures,
    get_measure_programs,
    PROGRAM_WEIGHTS
)
from financial_calculator import (
    get_hospital_financial_summary,
    get_system_financial_summary,
    calculate_cascading_financial_impact,
    calculate_measure_roi,
    format_currency,
    format_percent
)

# ==============================================================================
# INTEGRATED HOSPITAL ASSESSMENT
# ==============================================================================

def create_integrated_hospital_assessment(
    facility_id: str,
    star_rating_result: Dict,
    financial_data: pd.DataFrame
) -> Dict:
    """
    Create comprehensive assessment combining star ratings and financial impacts.

    Args:
        facility_id: 6-digit CMS facility ID
        star_rating_result: Result from star rating calculator
        financial_data: DataFrame with financial impacts

    Returns:
        Integrated assessment dictionary
    """
    # Get financial summary
    financial_summary = get_hospital_financial_summary(facility_id, financial_data)

    # Extract problem measures from measure_details (standardized score < -0.5)
    problem_measures_integrated = []
    measure_details = star_rating_result.get('measure_details', {})

    from measure_crosswalk import ALL_MEASURES

    for measure_id, measure_data in measure_details.items():
        standardized_score = measure_data.get('standardized', 0)

        # Only include measures that are problems (score < -0.5)
        if standardized_score < -0.5:
            # Get measure info from crosswalk
            measure_info = ALL_MEASURES.get(measure_id, {})
            measure_name = measure_info.get('name', measure_id)

            # Get programs affected
            programs = get_measure_programs(measure_id)

            # Build measure dict
            measure = {
                'measure_id': measure_id,
                'name': measure_name,
                'score': standardized_score,
                'original_value': measure_data.get('original', None),
                'severity': abs(standardized_score)  # How bad it is
            }

            # Calculate cascading financial impact
            if financial_summary['found'] and len(programs) > 1:
                financial_impact = calculate_cascading_financial_impact(
                    measure,
                    financial_summary,
                    measure_id,
                    programs
                )

                # Calculate ROI
                roi = calculate_measure_roi(
                    measure_id,
                    measure,
                    financial_summary,
                    programs
                )
            else:
                financial_impact = {'total_financial_impact': 0, 'programs_affected': programs}
                roi = {'roi_ratio': 0, 'potential_gain': 0, 'recommendation': 'N/A'}

            problem_measures_integrated.append({
                **measure,
                'programs_affected': programs,
                'num_programs': len(programs),
                'financial_impact': financial_impact.get('total_financial_impact', 0),
                'roi_ratio': roi.get('roi_ratio', 0),
                'roi_recommendation': roi.get('recommendation', 'N/A'),
                'is_cascading': len(programs) >= 3,
                'is_cross_program': len(programs) >= 2
            })

    # Sort by combined impact (ROI ratio * number of programs)
    problem_measures_integrated.sort(
        key=lambda x: x['roi_ratio'] * x['num_programs'],
        reverse=True
    )

    # Get hospital name from financial data
    hospital_name = 'Unknown'
    if not financial_data.empty:
        hospital_row = financial_data[financial_data['facility_id'] == facility_id]
        if not hospital_row.empty and 'hospital_name' in hospital_row.columns:
            hospital_name = hospital_row['hospital_name'].values[0]

    # Count measures from measure_details
    measures_reported = len(star_rating_result.get('measure_details', {}))

    # Create integrated assessment
    assessment = {
        'facility_id': facility_id,
        'hospital_name': hospital_name,

        # Star Rating Performance
        'star_rating': {
            'current_rating': star_rating_result.get('star_rating', 'N/A'),
            'summary_score': star_rating_result.get('summary_score', 0),
            'group_scores': star_rating_result.get('standardized_group_scores', {}),
            'measures_reported': measures_reported,
            'gap_to_next_star': star_rating_result.get('gap_to_next_star', None)
        },

        # Financial Performance
        'financial': financial_summary if financial_summary['found'] else None,

        # Integrated Analysis
        'integrated_analysis': {
            'total_problem_measures': len(problem_measures_integrated),
            'cross_program_measures': len([m for m in problem_measures_integrated if m['is_cross_program']]),
            'cascading_impact_measures': len([m for m in problem_measures_integrated if m['is_cascading']]),
            'total_financial_exposure': financial_summary.get('total_current_impact', {}).get('dollars', 0) if financial_summary['found'] else 0,
            'total_opportunity': financial_summary.get('opportunity', {}).get('total', 0) if financial_summary['found'] else 0
        },

        # Top Strategic Priorities (top 5 by ROI)
        'strategic_priorities': problem_measures_integrated[:5],

        # All problem measures with integrated data
        'all_problem_measures': problem_measures_integrated
    }

    return assessment


# ==============================================================================
# SYSTEM-WIDE INTEGRATED ASSESSMENT
# ==============================================================================

def create_system_integrated_assessment(
    star_rating_results: List[Dict],
    financial_data: pd.DataFrame
) -> Dict:
    """
    Create system-wide integrated assessment for multiple hospitals.

    Args:
        star_rating_results: List of star rating results from calculator
        financial_data: DataFrame with financial impacts for all hospitals

    Returns:
        System-wide integrated assessment
    """
    # Get system financial summary
    system_financial = get_system_financial_summary(financial_data)

    # Aggregate problem measures across all hospitals
    all_problem_measures = {}

    from measure_crosswalk import ALL_MEASURES

    for hospital_result in star_rating_results:
        facility_id = hospital_result.get('provider_id', hospital_result.get('facility_id', 'Unknown'))

        # Get hospital name from financial data
        hospital_name = 'Unknown'
        if not financial_data.empty:
            hospital_row = financial_data[financial_data['facility_id'] == facility_id]
            if not hospital_row.empty and 'hospital_name' in hospital_row.columns:
                hospital_name = hospital_row['hospital_name'].values[0]

        # Extract problem measures from measure_details
        measure_details = hospital_result.get('measure_details', {})

        for measure_id, measure_data in measure_details.items():
            standardized_score = measure_data.get('standardized', 0)

            # Only include measures that are problems (score < -0.5)
            if standardized_score < -0.5:
                measure_info = ALL_MEASURES.get(measure_id, {})
                measure_name = measure_info.get('name', measure_id)

                if measure_id not in all_problem_measures:
                    all_problem_measures[measure_id] = {
                        'measure_id': measure_id,
                        'measure_name': measure_name,
                        'hospitals_affected': [],
                        'total_hospitals_affected': 0,
                        'avg_performance': 0,
                        'avg_severity': 0,
                        'programs_affected': get_measure_programs(measure_id),
                        'financial_impact_total': 0
                    }

                all_problem_measures[measure_id]['hospitals_affected'].append({
                    'facility_id': facility_id,
                    'hospital_name': hospital_name,
                    'score': standardized_score,
                    'severity': abs(standardized_score)
                })

    # Calculate aggregates for each measure
    system_problem_measures = []

    for measure_id, measure_data in all_problem_measures.items():
        num_hospitals = len(measure_data['hospitals_affected'])
        total_hospitals = len(star_rating_results)

        # Calculate averages
        avg_score = np.mean([h['score'] for h in measure_data['hospitals_affected']])
        avg_severity = np.mean([h['severity'] for h in measure_data['hospitals_affected']])

        # Calculate system-wide financial impact
        programs = measure_data['programs_affected']
        num_programs = len(programs)

        # Estimate financial impact (proportion of total opportunity)
        total_opportunity = system_financial['opportunity']['total']
        estimated_impact = (total_opportunity / len(all_problem_measures)) * num_programs

        # Calculate priority score
        pct_affected = num_hospitals / total_hospitals
        priority_score = pct_affected * abs(avg_severity) * num_programs * 100

        system_problem_measures.append({
            'measure_id': measure_id,
            'measure_name': measure_data['measure_name'],
            'hospitals_affected': num_hospitals,
            'pct_system_affected': pct_affected * 100,
            'avg_performance': avg_score,
            'avg_severity': avg_severity,
            'programs_affected': ', '.join(programs),
            'num_programs': num_programs,
            'estimated_financial_impact': estimated_impact,
            'priority_score': priority_score,
            'is_cascading': num_programs >= 3,
            'is_cross_program': num_programs >= 2,
            'severity_level': 'CRITICAL' if abs(avg_severity) > 2 else 'HIGH' if abs(avg_severity) > 1 else 'MODERATE'
        })

    # Sort by priority score
    system_problem_measures.sort(key=lambda x: x['priority_score'], reverse=True)

    # Calculate system star rating metrics (BUG FIX #1: Use correct key name)
    # Try both 'star_rating' and 'predicted_star_rating' for compatibility
    valid_ratings = [r for r in star_rating_results
                     if r.get('star_rating', r.get('predicted_star_rating')) not in ['N/A', 'Not Rated', None]]

    avg_star_rating = np.mean([
        float(r.get('star_rating', r.get('predicted_star_rating', 0)))
        for r in valid_ratings
    ]) if valid_ratings else 0

    star_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for result in valid_ratings:
        rating = int(result.get('star_rating', result.get('predicted_star_rating', 0)))
        if 1 <= rating <= 5:
            star_distribution[rating] += 1

    # Identify quick wins (hospitals close to next star)
    quick_wins = []
    for result in valid_ratings:
        if result.get('gap_to_next_star') is not None and result['gap_to_next_star'] < 0.3:
            # Get financial data for this hospital
            facility_id = result['provider_id']
            financial_summary = get_hospital_financial_summary(facility_id, financial_data)

            quick_wins.append({
                'facility_id': facility_id,
                'hospital_name': result.get('hospital_name', 'Unknown'),
                'current_rating': result.get('star_rating', result.get('predicted_star_rating', 0)),
                'gap_to_next_star': result['gap_to_next_star'],
                'total_opportunity': financial_summary.get('opportunity', {}).get('total', 0) if financial_summary['found'] else 0,
                'feasibility': 'HIGH' if result['gap_to_next_star'] < 0.1 else 'MEDIUM'
            })

    quick_wins.sort(key=lambda x: x['gap_to_next_star'])

    # Create system assessment
    system_assessment = {
        'system_overview': {
            'total_hospitals': len(star_rating_results),
            'hospitals_with_ratings': len(valid_ratings),
            'avg_star_rating': avg_star_rating,
            'star_distribution': star_distribution,
            'hospitals_below_3_stars': star_distribution[1] + star_distribution[2],
            'pct_below_3_stars': ((star_distribution[1] + star_distribution[2]) / len(valid_ratings) * 100) if valid_ratings else 0
        },

        'financial_overview': {
            'total_medicare_revenue': system_financial['total_medicare_revenue'],
            'total_current_impact': system_financial['current_impact']['total'],
            'total_at_risk': system_financial['at_risk']['total'],
            'total_opportunity': system_financial['opportunity']['total'],
            'hvbp_penalties': system_financial['program_status']['hvbp_penalties'],
            'hrrp_penalties': system_financial['program_status']['hrrp_penalties'],
            'hacrp_penalties': system_financial['program_status']['hacrp_penalties']
        },

        'integrated_metrics': {
            'total_unique_problem_measures': len(system_problem_measures),
            'cross_program_measures': len([m for m in system_problem_measures if m['is_cross_program']]),
            'cascading_impact_measures': len([m for m in system_problem_measures if m['is_cascading']]),
            'avg_measures_per_hospital': np.mean([r.get('total_measures_reported', 0) for r in star_rating_results])
        },

        'strategic_priorities': system_problem_measures[:10],  # Top 10
        'all_problem_measures': system_problem_measures,
        'quick_wins': quick_wins[:10],  # Top 10 quick wins

        # Best practice sharing opportunities (measures where some hospitals excel)
        'best_practice_opportunities': identify_best_practice_opportunities(star_rating_results),

        # Risk trajectory
        'risk_trajectory': calculate_risk_trajectory(star_rating_results, financial_data)
    }

    return system_assessment


# ==============================================================================
# SUPPORTING ANALYSIS FUNCTIONS
# ==============================================================================

def identify_best_practice_opportunities(star_rating_results: List[Dict]) -> List[Dict]:
    """
    Identify measures where some hospitals are performing well and others are struggling.
    These represent opportunities for internal best practice sharing.

    Args:
        star_rating_results: List of star rating results

    Returns:
        List of best practice opportunities
    """
    measure_performance = {}

    for result in star_rating_results:
        if 'measure_scores' not in result:
            continue

        for measure_id, score in result['measure_scores'].items():
            if measure_id not in measure_performance:
                measure_performance[measure_id] = []

            measure_performance[measure_id].append({
                'facility_id': result['provider_id'],
                'hospital_name': result.get('hospital_name', 'Unknown'),
                'score': score
            })

    opportunities = []

    for measure_id, performances in measure_performance.items():
        if len(performances) < 2:
            continue

        scores = [p['score'] for p in performances]
        score_range = max(scores) - min(scores)

        # Opportunity exists if there's significant variation (>1 std dev range)
        if score_range > 1.0:
            best_performer = max(performances, key=lambda x: x['score'])
            worst_performer = min(performances, key=lambda x: x['score'])

            opportunities.append({
                'measure_id': measure_id,
                'measure_name': ALL_MEASURES.get(measure_id, {}).get('name', measure_id),
                'performance_gap': score_range,
                'best_performer': best_performer['hospital_name'],
                'best_score': best_performer['score'],
                'worst_performer': worst_performer['hospital_name'],
                'worst_score': worst_performer['score'],
                'opportunity_type': 'Best Practice Sharing'
            })

    opportunities.sort(key=lambda x: x['performance_gap'], reverse=True)

    return opportunities[:5]  # Top 5


def calculate_risk_trajectory(
    star_rating_results: List[Dict],
    financial_data: pd.DataFrame
) -> Dict:
    """
    Calculate risk trajectory based on current performance and trends.

    Args:
        star_rating_results: List of star rating results
        financial_data: DataFrame with financial impacts

    Returns:
        Risk trajectory assessment
    """
    # Count hospitals at risk (2 stars or below) (BUG FIX #1: Use correct key)
    at_risk_hospitals = [
        r for r in star_rating_results
        if r.get('star_rating', r.get('predicted_star_rating')) not in ['N/A', 'Not Rated', None]
        and int(r.get('star_rating', r.get('predicted_star_rating', 0))) <= 2
    ]

    # Calculate financial exposure for at-risk hospitals
    total_exposure = 0
    for result in at_risk_hospitals:
        facility_id = result['provider_id']
        financial_summary = get_hospital_financial_summary(facility_id, financial_data)
        if financial_summary['found']:
            total_exposure += abs(financial_summary['total_current_impact']['dollars'])

    # Determine trajectory
    pct_at_risk = (len(at_risk_hospitals) / len(star_rating_results) * 100) if star_rating_results else 0

    if pct_at_risk > 50:
        trajectory = 'HIGH RISK'
        color = '🔴'
    elif pct_at_risk > 25:
        trajectory = 'MODERATE RISK'
        color = '🟠'
    else:
        trajectory = 'LOW RISK'
        color = '🟢'

    return {
        'hospitals_at_risk': len(at_risk_hospitals),
        'pct_at_risk': pct_at_risk,
        'financial_exposure': total_exposure,
        'trajectory': trajectory,
        'trajectory_color': color,
        'recommendation': 'Immediate system-wide intervention required' if pct_at_risk > 50
                         else 'Targeted support for struggling hospitals' if pct_at_risk > 25
                         else 'Continue monitoring and maintain performance'
    }


# ==============================================================================
# CROSS-PROGRAM IMPACT ANALYSIS
# ==============================================================================

def create_cross_program_impact_report(
    integrated_assessment: Dict
) -> pd.DataFrame:
    """
    Create detailed cross-program impact report showing only measures
    that affect multiple programs.

    Args:
        integrated_assessment: Result from create_integrated_hospital_assessment()
                              or create_system_integrated_assessment()

    Returns:
        DataFrame with cross-program measures
    """
    if 'all_problem_measures' in integrated_assessment:
        problem_measures = integrated_assessment['all_problem_measures']
    else:
        return pd.DataFrame()

    # Filter to cross-program measures only
    cross_program = [m for m in problem_measures if m.get('is_cross_program', False)]

    if not cross_program:
        return pd.DataFrame()

    # Create formatted report
    report_data = []

    for measure in cross_program:
        # Calculate priority score: programs × hospitals × severity
        num_programs = measure.get('num_programs', len(str(measure.get('programs_affected', '')).split(',')))
        hospitals_affected = measure.get('hospitals_affected', 0)
        avg_performance = measure.get('avg_performance', measure.get('score', 0))
        priority_score = num_programs * hospitals_affected * abs(avg_performance)

        report_data.append({
            'Measure': measure.get('measure_name', measure['measure_id']),
            'Programs Affected': measure.get('programs_affected', 'N/A'),
            'Hospitals Affected': hospitals_affected,
            'Avg Performance': f"{avg_performance:.2f} SD",
            'Priority Score': round(priority_score, 1)
        })

    # Sort by Priority Score descending
    df = pd.DataFrame(report_data)
    if not df.empty:
        df = df.sort_values('Priority Score', ascending=False)
    return df


# ==============================================================================
# EXECUTIVE SUMMARY GENERATOR
# ==============================================================================

def create_executive_summary_data(
    system_assessment: Dict
) -> Dict:
    """
    Create executive summary data for dashboard/PDF generation.

    Args:
        system_assessment: Result from create_system_integrated_assessment()

    Returns:
        Dictionary with executive summary data formatted for presentation
    """
    return {
        # Header metrics
        'key_metrics': {
            'total_hospitals': system_assessment['system_overview']['total_hospitals'],
            'avg_star_rating': f"{system_assessment['system_overview']['avg_star_rating']:.2f}",
            'hospitals_at_risk': system_assessment['system_overview']['hospitals_below_3_stars'],
            'total_financial_impact': format_currency(system_assessment['financial_overview']['total_current_impact']),
            'total_opportunity': format_currency(system_assessment['financial_overview']['total_opportunity'])
        },

        # Financial Impact Summary
        'financial_summary': {
            'medicare_revenue': format_currency(system_assessment['financial_overview']['total_medicare_revenue']),
            'current_impact': format_currency(system_assessment['financial_overview']['total_current_impact']),
            'at_risk_revenue': format_currency(system_assessment['financial_overview']['total_at_risk']),
            'opportunity_revenue': format_currency(system_assessment['financial_overview']['total_opportunity']),
            'hvbp_penalties': system_assessment['financial_overview']['hvbp_penalties'],
            'hrrp_penalties': system_assessment['financial_overview']['hrrp_penalties'],
            'hacrp_penalties': system_assessment['financial_overview']['hacrp_penalties']
        },

        # Market Position
        'market_position': {
            'star_distribution': system_assessment['system_overview']['star_distribution'],
            'pct_below_3_stars': f"{system_assessment['system_overview']['pct_below_3_stars']:.1f}%",
            'avg_rating': system_assessment['system_overview']['avg_star_rating']
        },

        # Top 5 Strategic Priorities
        'strategic_priorities': [
            {
                'rank': idx + 1,
                'measure': m['measure_name'],
                'programs': m['programs_affected'],
                'hospitals_affected': f"{m.get('hospitals_affected', 0)} ({m.get('pct_system_affected', 0):.0f}%)",
                'financial_impact': format_currency(m.get('estimated_financial_impact', 0)),
                'severity': m.get('severity_level', 'N/A')
            }
            for idx, m in enumerate(system_assessment['strategic_priorities'][:5])
        ],

        # Risk Trajectory
        'risk_trajectory': {
            'status': system_assessment['risk_trajectory']['trajectory'],
            'color': system_assessment['risk_trajectory']['trajectory_color'],
            'hospitals_at_risk': system_assessment['risk_trajectory']['hospitals_at_risk'],
            'financial_exposure': format_currency(system_assessment['risk_trajectory']['financial_exposure']),
            'recommendation': system_assessment['risk_trajectory']['recommendation']
        },

        # Quick Wins
        'quick_wins': [
            {
                'hospital': qw['hospital_name'],
                'current_rating': qw['current_rating'],
                'gap': f"{qw['gap_to_next_star']:.3f}",
                'opportunity': format_currency(qw['total_opportunity']),
                'feasibility': qw['feasibility']
            }
            for qw in system_assessment['quick_wins'][:3]
        ],

        # Best Practice Opportunities
        'best_practices': [
            {
                'measure': bp['measure_name'],
                'leader': bp['best_performer'],
                'learner': bp['worst_performer'],
                'gap': f"{bp['performance_gap']:.2f} SD"
            }
            for bp in system_assessment['best_practice_opportunities'][:3]
        ]
    }
