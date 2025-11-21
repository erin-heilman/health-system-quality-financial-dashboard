"""
Visualization functions for CMS Star Rating Calculator
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List


def create_group_scores_chart(result: Dict) -> go.Figure:
    """
    Create horizontal bar chart of group scores.

    Args:
        result: Hospital result dictionary

    Returns:
        Plotly figure
    """
    if not result['is_eligible']:
        # Create empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="Hospital not eligible for star rating",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Group Performance Scores",
            height=400
        )
        return fig

    # Get standardized group scores
    group_scores = result['standardized_group_scores']

    # Prepare data
    groups = list(group_scores.keys())
    scores = [group_scores[g] for g in groups]
    measures_count = [result['measures_per_group'][g] for g in groups]

    # Color code: red (<-0.5), yellow (-0.5 to 0.5), green (>0.5)
    colors = []
    for score in scores:
        if score < -0.5:
            colors.append('#ef4444')  # Red
        elif score > 0.5:
            colors.append('#22c55e')  # Green
        else:
            colors.append('#eab308')  # Yellow

    # Create horizontal bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=groups,
        x=scores,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{s:.2f} ({m} measures)" for s, m in zip(scores, measures_count)],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>'
    ))

    # Add national average line at 0
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=2,
                  annotation_text="National Average", annotation_position="top")

    # Styling
    fig.update_layout(
        title=f"Group Performance Scores - {result['provider_id']}",
        xaxis_title="Standardized Score (0 = National Average)",
        yaxis_title="Measure Group",
        height=400,
        showlegend=False,
        template="plotly_white",
        hovermode='closest'
    )

    return fig


def create_star_distribution_chart(results_list: List[Dict]) -> go.Figure:
    """
    Create star rating distribution chart.

    Args:
        results_list: List of hospital results

    Returns:
        Plotly figure
    """
    # Handle empty list
    if not results_list or len(results_list) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No hospitals to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="Star Rating Distribution (n=0)",
            height=400,
            template="plotly_white"
        )
        return fig

    # Count star ratings
    star_counts = {}
    for result in results_list:
        stars = result['star_rating']
        if stars is not None:
            star_counts[stars] = star_counts.get(stars, 0) + 1
        else:
            star_counts['Not Eligible'] = star_counts.get('Not Eligible', 0) + 1

    # Prepare data
    labels = []
    values = []
    colors = []

    star_colors = {
        1: '#ef4444',  # Red
        2: '#f97316',  # Orange
        3: '#eab308',  # Yellow
        4: '#22c55e',  # Green
        5: '#3b82f6',  # Blue
        'Not Eligible': '#9ca3af'  # Gray
    }

    # Build list of stars to display (sorted integers, then "Not Eligible" if present)
    stars_to_display = sorted([k for k in star_counts.keys() if isinstance(k, int)])
    if 'Not Eligible' in star_counts:
        stars_to_display.append('Not Eligible')

    for star in stars_to_display:
        if star in star_counts:
            if isinstance(star, int):
                labels.append(f"{star} Star{'s' if star != 1 else ''}")
            else:
                labels.append(star)
            values.append(star_counts[star])
            colors.append(star_colors[star])

    # Create bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker=dict(color=colors),
        text=values,
        textposition='inside',
        textfont=dict(size=18, color='white'),
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    ))

    total = sum(values)
    fig.update_layout(
        title=f"Star Rating Distribution (n={total})",
        xaxis_title="Star Rating",
        yaxis_title="Number of Hospitals",
        height=400,
        showlegend=False,
        template="plotly_white"
    )

    return fig


def create_comparison_table(result: Dict) -> pd.DataFrame:
    """
    Create detailed measure comparison table.

    Args:
        result: Hospital result dictionary

    Returns:
        DataFrame with measure details
    """
    measure_details = result['measure_details']

    table_data = []
    for measure, details in measure_details.items():
        row = {
            'Measure': measure,
            'Hospital Value': f"{details['original']:.3f}" if pd.notna(details['original']) else 'N/A',
            'Standardized Score': f"{details['standardized']:.3f}" if pd.notna(details['standardized']) else 'N/A',
            'Performance': 'Above Average' if details['standardized'] > 0 else 'Below Average' if details['standardized'] < 0 else 'Average'
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)
    df = df.sort_values('Standardized Score', ascending=False)

    return df


def create_summary_metrics_html(result: Dict) -> str:
    """
    Create HTML for summary metrics cards.

    Args:
        result: Hospital result dictionary

    Returns:
        HTML string
    """
    if not result['is_eligible']:
        return f"""
        <div style='padding: 20px; background-color: #fee2e2; border-radius: 8px; border-left: 4px solid #ef4444;'>
            <h3 style='color: #991b1b; margin-top: 0;'>Not Eligible for Star Rating</h3>
            <p style='color: #7f1d1d;'>{result['eligibility_reason']}</p>
        </div>
        """

    stars = '⭐' * result['star_rating']
    score_color = '#22c55e' if result['summary_score'] > 0 else '#ef4444' if result['summary_score'] < -0.5 else '#eab308'

    html = f"""
    <div style='display: flex; gap: 20px; flex-wrap: wrap;'>
        <div style='flex: 1; min-width: 200px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px; color: white; text-align: center;'>
            <div style='font-size: 48px;'>{stars}</div>
            <div style='font-size: 32px; font-weight: bold; margin: 10px 0;'>{result['star_rating']} Stars</div>
            <div style='font-size: 14px; opacity: 0.9;'>CMS Overall Rating</div>
        </div>

        <div style='flex: 1; min-width: 200px; padding: 20px; background-color: {score_color}20;
                    border-radius: 10px; border-left: 4px solid {score_color};'>
            <div style='font-size: 14px; color: #666; margin-bottom: 5px;'>Summary Score</div>
            <div style='font-size: 32px; font-weight: bold; color: {score_color};'>{result['summary_score']:.3f}</div>
            <div style='font-size: 12px; color: #666; margin-top: 5px;'>
                {'Above' if result['summary_score'] > 0 else 'Below'} National Average
            </div>
        </div>

        <div style='flex: 1; min-width: 200px; padding: 20px; background-color: #f0f9ff;
                    border-radius: 10px; border-left: 4px solid #3b82f6;'>
            <div style='font-size: 14px; color: #666; margin-bottom: 5px;'>Peer Group</div>
            <div style='font-size: 32px; font-weight: bold; color: #3b82f6;'>{result['peer_group']}</div>
            <div style='font-size: 12px; color: #666; margin-top: 5px;'>
                Measure Groups with ≥3 Measures
            </div>
        </div>

        <div style='flex: 1; min-width: 200px; padding: 20px; background-color: #f0fdf4;
                    border-radius: 10px; border-left: 4px solid #22c55e;'>
            <div style='font-size: 14px; color: #666; margin-bottom: 5px;'>Total Measures</div>
            <div style='font-size: 32px; font-weight: bold; color: #22c55e;'>{len(result['measure_details'])}</div>
            <div style='font-size: 12px; color: #666; margin-top: 5px;'>
                Out of 46 Possible
            </div>
        </div>
    </div>
    """

    return html


def create_peer_comparison_chart(results_list: List[Dict], hospital_names: Dict[str, str] = None) -> go.Figure:
    """
    Create scatter plot comparing hospitals in same peer group.

    Args:
        results_list: List of hospital results
        hospital_names: Optional dictionary mapping provider_id to hospital name

    Returns:
        Plotly figure
    """
    # Prepare data
    data = []
    for result in results_list:
        if result['is_eligible']:
            provider_id = result['provider_id']
            hospital_name = hospital_names.get(provider_id, provider_id) if hospital_names else provider_id

            data.append({
                'Provider_ID': provider_id,
                'Hospital_Name': hospital_name,
                'Summary_Score': result['summary_score'],
                'Star_Rating': result['star_rating'],
                'Peer_Group': result['peer_group']
            })

    df = pd.DataFrame(data)

    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No eligible hospitals to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig

    # Create scatter plot with darker yellow for accessibility
    # Custom color scale: Red -> Dark Yellow -> Green
    custom_colors = ['#ef4444', '#d97706', '#22c55e']  # red, dark yellow/amber, green

    fig = px.scatter(
        df,
        x='Hospital_Name',
        y='Summary_Score',
        color='Star_Rating',
        size=[15]*len(df),
        hover_data=['Peer_Group', 'Provider_ID'],
        color_continuous_scale=custom_colors,
        title="Summary Scores by Hospital"
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray",
                  annotation_text="National Average")

    fig.update_layout(
        xaxis_title="Hospital",
        yaxis_title="Summary Score",
        height=600,  # Increased height for better visibility
        template="plotly_white",
        xaxis={'tickangle': -45}  # Angle the hospital names for readability
    )

    return fig


def create_measure_heatmap(results_list: List[Dict], hospital_names: Dict[str, str] = None) -> go.Figure:
    """
    Create heatmap showing all hospitals' performance across measure groups.

    Args:
        results_list: List of hospital results
        hospital_names: Optional dictionary mapping provider_id to hospital name

    Returns:
        Plotly figure
    """
    from star_rating_calculator import MEASURE_GROUPS

    # Prepare data matrix
    hospital_labels = []
    group_names = list(MEASURE_GROUPS.keys())
    matrix = []

    for result in results_list:
        if result['is_eligible']:
            provider_id = result['provider_id']
            hospital_name = hospital_names.get(provider_id, provider_id) if hospital_names else provider_id
            hospital_labels.append(hospital_name)

            row = []
            for group in group_names:
                score = result['standardized_group_scores'].get(group, np.nan)
                row.append(score)
            matrix.append(row)

    if len(matrix) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No eligible hospitals to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=group_names,
        y=hospital_labels,
        colorscale='RdYlGn',
        zmid=0,
        text=[[f"{val:.2f}" if not np.isnan(val) else "N/A" for val in row] for row in matrix],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Score")
    ))

    fig.update_layout(
        title="Performance Heatmap Across All Measure Groups",
        xaxis_title="Measure Group",
        yaxis_title="Hospital",
        height=max(400, len(hospital_labels) * 30),
        template="plotly_white"
    )

    return fig
