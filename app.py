"""
Streamlit Web Application for CMS Hospital Star Rating Calculator

This app provides an interactive interface for calculating and visualizing
CMS hospital star ratings based on the July 2025 methodology.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import io
from datetime import datetime
import plotly.graph_objects as go

# Import our modules
from star_rating_calculator import (
    load_data, process_hospitals, MEASURE_GROUPS, STAR_CUTOFFS
)
from visualizations import (
    create_group_scores_chart,
    create_star_distribution_chart,
    create_comparison_table,
    create_summary_metrics_html,
    create_peer_comparison_chart,
    create_measure_heatmap
)
from excel_export import export_to_excel
from pdf_export import generate_executive_summary_pdf, generate_system_executive_summary_pdf

# Integrated assessment imports
from financial_calculator import (
    parse_definitive_csv,
    calculate_total_quality_impact,
    get_system_financial_summary,
    create_financial_summary_table
)
from integrated_assessment import (
    create_system_integrated_assessment,
    create_executive_summary_data,
    create_cross_program_impact_report
)
from executive_dashboard_pdf import generate_executive_dashboard_pdf

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="CMS Hospital Star Rating Calculator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #366092;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #366092;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_hospital_data():
    """Load and cache hospital data from Supabase or return None for file upload."""
    try:
        # Try to load from Supabase if configured
        if 'SUPABASE_URL' in st.secrets and 'SUPABASE_KEY' in st.secrets:
            from supabase import create_client
            import tempfile

            supabase = create_client(st.secrets['SUPABASE_URL'], st.secrets['SUPABASE_KEY'])

            # Download file from Supabase storage
            bucket_name = st.secrets.get('SUPABASE_BUCKET', 'hospital-data')
            file_name = st.secrets.get('SUPABASE_FILE', 'All Data July 2025.sas7bdat')

            # Download to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.sas7bdat') as tmp_file:
                data = supabase.storage.from_(bucket_name).download(file_name)
                tmp_file.write(data)
                tmp_path = tmp_file.name

            # Load data from temp file
            df_original, df_std, national_stats = load_data(tmp_path)

            # Clean up temp file
            import os
            os.unlink(tmp_path)

            return df_original, df_std, national_stats, None
    except Exception as e:
        # Supabase not configured or failed - return None to prompt for upload
        return None, None, None, None

    # No Supabase configured
    return None, None, None, None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_star_rating(rating):
    """Format star rating with emoji stars or special status."""
    # Handle special status values
    if rating == 'No Data':
        return '❌ No Data'
    elif rating == 'Insufficient Data':
        return '⚠️ Insufficient Data'
    elif rating is None or pd.isna(rating):
        return '❓ Not Rated'

    # Convert to integer (handles floats like 2.0)
    try:
        rating_int = int(rating)
        # Ensure it's between 1-5
        rating_int = max(1, min(5, rating_int))
        return '⭐' * rating_int
    except (ValueError, TypeError):
        return '⚠️ Error'

def get_star_color(rating):
    """Get color for star rating."""
    colors = {
        5: '#3b82f6',  # Blue
        4: '#22c55e',  # Green
        3: '#eab308',  # Yellow
        2: '#f97316',  # Orange
        1: '#ef4444',  # Red
        None: '#9ca3af'  # Gray
    }
    return colors.get(rating, '#9ca3af')

def create_executive_summary(results_list):
    """
    Enhanced executive summary with weighted impact analysis and actionable intelligence.
    Designed for C-suite executives with specific, prioritized recommendations.
    """
    from measure_names import MEASURE_NAMES, GROUP_WEIGHTS, GROUP_MEASURE_COUNTS

    # Filter out invalid results
    valid_results = [r for r in results_list if isinstance(r.get('star_rating'), int)]

    if len(valid_results) == 0:
        return None

    # Calculate per-measure weights
    MEASURE_WEIGHTS = {}
    for group, group_weight in GROUP_WEIGHTS.items():
        measure_count = GROUP_MEASURE_COUNTS[group]
        per_measure_weight = group_weight / measure_count
        MEASURE_WEIGHTS[group] = per_measure_weight

    # 1. SYSTEM OVERVIEW
    current_avg_stars = np.mean([r['star_rating'] for r in valid_results])
    system_overview = {
        'Total_Hospitals': len(results_list),
        'Hospitals_with_Valid_Ratings': len(valid_results),
        'Average_Star_Rating': round(current_avg_stars, 2),
        '5_Star_Hospitals': sum(1 for r in valid_results if r.get('star_rating') == 5),
        '4_Star_Hospitals': sum(1 for r in valid_results if r.get('star_rating') == 4),
        '3_Star_Hospitals': sum(1 for r in valid_results if r.get('star_rating') == 3),
        '2_Star_Hospitals': sum(1 for r in valid_results if r.get('star_rating') == 2),
        '1_Star_Hospitals': sum(1 for r in valid_results if r.get('star_rating') == 1),
    }

    # 2. MEASURE-LEVEL IMPACT ANALYSIS (WEIGHTED)
    measure_analysis = {}

    for result in valid_results:
        if 'measure_details' not in result:
            continue

        for measure, details in result['measure_details'].items():
            z_score = details.get('standardized', 0)

            if measure not in measure_analysis:
                # Find measure group
                measure_group = None
                for group, measures in MEASURE_GROUPS.items():
                    if measure in measures['measures']:
                        measure_group = group
                        break

                if measure_group is None:
                    continue

                measure_analysis[measure] = {
                    'measure_group': measure_group,
                    'per_measure_weight': MEASURE_WEIGHTS[measure_group],
                    'hospitals_reporting': 0,
                    'hospitals_significantly_below': 0,
                    'z_scores': []
                }

            measure_analysis[measure]['hospitals_reporting'] += 1
            measure_analysis[measure]['z_scores'].append(z_score)

            if z_score < -0.5:
                measure_analysis[measure]['hospitals_significantly_below'] += 1

    # Calculate impact scores
    impact_data = []
    for measure, data in measure_analysis.items():
        if data['hospitals_reporting'] >= 3:  # Only include if at least 3 hospitals report it
            avg_z = np.mean(data['z_scores'])
            pct_affected = data['hospitals_significantly_below'] / data['hospitals_reporting']
            severity = abs(avg_z)
            measure_weight = data['per_measure_weight']

            # Impact score = (% affected) × (severity) × (weight) × 100
            impact_score = pct_affected * severity * measure_weight * 100

            impact_data.append({
                'Measure': measure,
                'Measure_Name': MEASURE_NAMES.get(measure, measure),
                'Group': data['measure_group'],
                'Per_Measure_Weight': f"{data['per_measure_weight']*100:.1f}%",
                'Hospitals_Affected': data['hospitals_significantly_below'],
                'Pct_System': f"{(data['hospitals_significantly_below']/len(valid_results)*100):.0f}%",
                'Avg_Z_Score': round(avg_z, 2),
                'Impact_Score': round(impact_score, 2),
                'Severity': 'CRITICAL' if avg_z < -1.0 else 'HIGH' if avg_z < -0.5 else 'MODERATE'
            })

    top_problem_measures_df = pd.DataFrame(impact_data).sort_values('Impact_Score', ascending=False).head(15)

    # 3. GROUP PERFORMANCE
    group_performance = []
    for group in ['Mortality', 'Safety of Care', 'Readmission', 'Patient Experience', 'Timely & Effective Care']:
        scores = [r['standardized_group_scores'].get(group) for r in valid_results
                  if 'standardized_group_scores' in r and r['standardized_group_scores'].get(group) is not None]

        if scores:
            avg_score = np.mean(scores)
            hospitals_below = sum(1 for s in scores if s < -0.5)
            group_performance.append({
                'Measure_Group': group,
                'System_Average': round(avg_score, 3),
                'Best_Hospital': round(max(scores), 3),
                'Worst_Hospital': round(min(scores), 3),
                'Hospitals_Below_National': hospitals_below,
                'Pct_Below_National': f"{(hospitals_below/len(scores)*100):.1f}%",
                'Priority': 'HIGH' if avg_score < -0.5 else 'MEDIUM' if avg_score < 0 else 'LOW'
            })

    group_performance_df = pd.DataFrame(group_performance).sort_values('System_Average') if group_performance else pd.DataFrame()

    # 4. IMPROVEMENT OPPORTUNITIES (Quick Wins)
    improvement_opps = []

    for result in valid_results:
        if result['star_rating'] >= 5:
            continue

        current_rating = result['star_rating']
        summary_score = result['summary_score']
        peer_group = result['peer_group']

        # Get cutoff for next star
        if peer_group in STAR_CUTOFFS and current_rating < 5:
            next_rating = current_rating + 1
            next_cutoff_min = STAR_CUTOFFS[peer_group][next_rating][0]
            gap_to_next_star = next_cutoff_min - summary_score

            # Only include realistic improvements (<0.3 points)
            if gap_to_next_star <= 0.3:
                group_scores = result.get('standardized_group_scores', {})
                worst_group = min(group_scores.items(), key=lambda x: x[1]) if group_scores else ('Unknown', 0)

                improvement_opps.append({
                    'Facility_ID': result['provider_id'],
                    'Current_Rating': current_rating,
                    'Target_Rating': next_rating,
                    'Gap_to_Next_Star': round(gap_to_next_star, 3),
                    'Primary_Focus_Area': worst_group[0],
                    'Focus_Area_Score': round(worst_group[1], 2),
                    'Feasibility': 'HIGH' if gap_to_next_star < 0.15 else 'MEDIUM',
                    'Recommendation': f"Focus on {worst_group[0]} (currently {worst_group[1]:.2f}). Improving top 2-3 measures could achieve {next_rating} stars."
                })

    improvement_opps_df = pd.DataFrame(improvement_opps).sort_values('Gap_to_Next_Star') if improvement_opps else pd.DataFrame()

    # 5. WEIGHTED RECOMMENDATIONS
    recommendations = []

    # Top 3 problem measures
    if not top_problem_measures_df.empty:
        for _, row in top_problem_measures_df.head(3).iterrows():
            recommendations.append({
                'Priority': row['Severity'],
                'Focus_Area': f"{row['Measure_Name']} ({row['Group']})",
                'Issue': f"{row['Hospitals_Affected']} hospitals affected ({row['Pct_System']})",
                'Impact_Score': row['Impact_Score'],
                'Recommendation': f"System-wide intervention for {row['Measure_Name']}. Weighted impact: {row['Impact_Score']:.1f} points."
            })

    recommendations_df = pd.DataFrame(recommendations) if recommendations else pd.DataFrame()

    return {
        'System_Overview': pd.DataFrame([system_overview]),
        'Top_Problem_Measures': top_problem_measures_df,
        'Group_Performance': group_performance_df,
        'Improvement_Opportunities': improvement_opps_df,
        'Recommendations': recommendations_df
    }

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    global go  # Ensure we use the globally imported plotly.graph_objects

    # Initialize session state to preserve results across reruns (prevents clearing on download)
    if 'results_calculated' not in st.session_state:
        st.session_state.results_calculated = False
        st.session_state.results_list = None
        st.session_state.summary_df = None
        st.session_state.exec_summary = None
        st.session_state.facility_ids = []
        st.session_state.financial_data = None
        st.session_state.integrated_assessment = None

    # Header (EDIT #2: Fixed Branding)
    st.markdown('<div class="main-header">Integrated Quality & Financial Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive quality and financial performance analysis across CMS star ratings and value-based programs</div>', unsafe_allow_html=True)
    st.caption("Star Rating Data: July 2025 (Performance: July 2021 - June 2024)")

    # Load data
    with st.spinner("Loading hospital data..."):
        df_original, df_std, national_stats, error = load_hospital_data()

    if df_original is None:
        st.info("📁 **No demo data configured.** Please upload your hospital quality data file to get started.")
        st.markdown("---")
        st.markdown("### Upload Hospital Data")
        st.markdown("Upload your CMS Star Rating data file (Excel, CSV, or SAS format):")
        uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'xls', 'csv', 'sas7bdat'], key="hospital_upload")

        if uploaded_file is not None:
            with st.spinner("Processing uploaded file..."):
                try:
                    # Save uploaded file temporarily
                    import tempfile
                    import os
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # Load data
                    df_original, df_std, national_stats = load_data(tmp_path)

                    # Clean up
                    os.unlink(tmp_path)

                    st.success("✅ Data loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
                    return
        else:
            st.info("👆 Upload a file to begin analysis")
            return

        if df_original is None:
            st.error("❌ Failed to load hospital data from uploaded file.")
            return

    # Sidebar
    with st.sidebar:
        st.header("📋 Input Options")

        # Input method selection
        input_method = st.radio(
            "Select input method:",
            ["Upload Financial CSV (Single Input)", "Manual Entry", "Upload Facility IDs Only", "Sample Hospitals"]
        )

        # Initialize facility_ids from session state if available (for combined input method)
        facility_ids = st.session_state.get('facility_ids', [])

        if input_method == "Upload Financial CSV (Single Input)":
            st.info("Upload your financial CSV below to generate a comprehensive quality and financial risk assessment")

            financial_file_combined = st.file_uploader(
                "Upload Financial CSV with Hospital Data",
                type=['csv'],
                key="combined_upload",
                help="""Your CSV should include:
• Hospital Name and Facility ID (CCN)
• Net Medicare Revenue
• HVBP adjustment (FY 2026)
• HRRP penalty (FY 2026)
• HACRP penalty (FY 2025)

The tool will automatically:
✓ Calculate CMS Star Ratings (July 2025)
✓ Analyze HVBP, HRRP, and HACRP performance
✓ Identify cross-program problem areas
✓ Generate strategic priorities for improvement"""
            )

            if financial_file_combined:
                try:
                    # Parse financial data
                    from financial_calculator import parse_definitive_csv, calculate_total_quality_impact
                    financial_data = parse_definitive_csv(financial_file_combined)
                    financial_data = calculate_total_quality_impact(financial_data)
                    st.session_state.financial_data = financial_data

                    # Extract facility IDs automatically and store in session state
                    facility_ids = [str(fid) for fid in financial_data['facility_id'].tolist()]
                    st.session_state.facility_ids = facility_ids

                    st.success(f"✅ Loaded financial data for {len(facility_ids)} hospitals")
                    st.info(f"🏥 Facilities detected: {', '.join(facility_ids)}")

                except Exception as e:
                    st.error(f"Error loading financial data: {e}")
                    st.session_state.facility_ids = []

        elif input_method == "Manual Entry":
            facility_input = st.text_area(
                "Enter Facility IDs (one per line):",
                height=200,
                placeholder="010001\n010005\n010006",
                help="Enter CMS Certification Numbers (CCN), one per line"
            )
            if facility_input:
                facility_ids = [fid.strip() for fid in facility_input.split('\n') if fid.strip()]

        elif input_method == "Upload Facility IDs Only":
            uploaded_file = st.file_uploader(
                "Upload a CSV or TXT file with Facility IDs",
                type=['csv', 'txt'],
                key="ids_only_upload",
                help="File should contain facility IDs, one per line or in a column"
            )
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df_upload = pd.read_csv(uploaded_file)
                        # Try to find the ID column
                        id_cols = [col for col in df_upload.columns if 'id' in col.lower() or 'provider' in col.lower()]
                        if id_cols:
                            facility_ids = df_upload[id_cols[0]].astype(str).tolist()
                        else:
                            facility_ids = df_upload.iloc[:, 0].astype(str).tolist()
                    else:
                        content = uploaded_file.read().decode('utf-8')
                        facility_ids = [line.strip() for line in content.split('\n') if line.strip()]
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        else:  # Sample Hospitals
            num_samples = st.slider("Number of sample hospitals:", 1, 20, 5)
            if st.button("Generate Random Sample"):
                facility_ids = df_original['PROVIDER_ID'].sample(n=num_samples).tolist()
                st.success(f"✅ Selected {num_samples} random hospitals")

        st.markdown("---")

        # Financial Data Upload Section (only show for non-combined input methods)
        if input_method != "Upload Financial CSV (Single Input)":
            st.header("Financial Data (Optional)")

            financial_file = st.file_uploader(
                "Upload Definitive Healthcare CSV",
                type=['csv'],
                help="CSV with columns: Hospital Name, CCN, Net Medicare Revenue, HVBP %, HVBP $, HRRP %, HRRP $, HACRP %, HACRP $"
            )

            if financial_file:
                try:
                    financial_data = parse_definitive_csv(financial_file)
                    financial_data = calculate_total_quality_impact(financial_data)
                    st.session_state.financial_data = financial_data
                    st.success(f"✅ Loaded financial data for {len(financial_data)} hospitals")

                    # Show quick summary
                    total_revenue = financial_data['net_medicare_revenue'].sum()
                    total_impact = financial_data['total_current_impact_dollars'].sum()

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Medicare Revenue", f"${total_revenue/1e6:.1f}M")
                    with col2:
                        st.metric("Current Impact", f"${total_impact/1e6:.1f}M")

                except Exception as e:
                    st.error(f"Error loading financial data: {e}")
                    st.session_state.financial_data = None

        st.markdown("---")

        # Calculate button (EDIT #2: Fixed branding)
        calculate_button = st.button("Generate Integrated Assessment", type="primary", use_container_width=True)

        # Info section (EDIT #2 & #3: Fixed branding, removed emojis)
        with st.expander("About This Tool"):
            st.markdown("""
            **Integrated Quality & Financial Risk Assessment**

            This tool provides comprehensive analysis across CMS quality programs:
            - **Star Ratings**: 46 quality measures across 5 groups (July 2025 data)
            - **HVBP**: Hospital Value-Based Purchasing (FY 2026)
            - **HRRP**: Readmissions Reduction Program (FY 2026)
            - **HACRP**: Hospital-Acquired Conditions Program (FY 2025)

            **Data Sources:** CMS Hospital Compare (July 2025) + Financial Program Data

            **Measurement Periods:** July 2021 - June 2024 (most measures)
            """)

        with st.expander("📊 Dataset Info"):
            st.metric("Total Hospitals", f"{len(df_original):,}")
            st.metric("Total Measures", "46")
            st.metric("Measure Groups", "5")

    # Main content area
    if calculate_button:
        # Process hospitals when button is clicked
        if not facility_ids:
            st.error("❌ Please enter at least one Facility ID")
        else:
            st.markdown("---")

            with st.spinner(f"🔄 Calculating star ratings for {len(facility_ids)} hospitals..."):
                try:
                    results_list, summary_df = process_hospitals(facility_ids, df_std, df_original)

                    # Store results in session state to preserve across reruns
                    st.session_state.results_calculated = True
                    st.session_state.results_list = results_list
                    st.session_state.summary_df = summary_df
                    st.session_state.facility_ids = facility_ids

                except Exception as e:
                    st.error(f"❌ Error processing hospitals: {e}")
                    st.session_state.results_calculated = False

    # Check if we have results to display (from session state)
    if not st.session_state.results_calculated:
        # Welcome screen
        st.info("👈 Enter facility IDs in the sidebar and click 'Calculate Star Ratings' to begin")

        # Display methodology overview
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Measure Groups")
            for group_name, group_info in MEASURE_GROUPS.items():
                with st.expander(f"{group_name} ({group_info['weight']:.0%} weight)"):
                    st.write(f"**Measures:** {len(group_info['measures'])}")
                    st.write("**Measures included:**")
                    for measure in group_info['measures']:
                        st.write(f"- {measure}")

        with col2:
            st.subheader("Star Rating Criteria")
            st.markdown("""
            **CMS Methodology:**
            - Based on performance across 5 measure groups
            - Hospitals with more complete data yield more reliable ratings

            **Peer Grouping:**
            - Hospitals grouped by number of measure groups (3, 4, or 5)
            - Compared only to hospitals in same peer group

            **Star Assignment:**
            - 1 Star: Significantly below average
            - 2 Stars: Below average
            - 3 Stars: Average
            - 4 Stars: Above average
            - 5 Stars: Significantly above average
            """)

        return

    # ========================================================================
    # RESULTS DISPLAY (using session state)
    # ========================================================================

    # Get results from session state
    results_list = st.session_state.results_list
    summary_df = st.session_state.summary_df
    facility_ids = st.session_state.facility_ids

    if len(results_list) == 0:
        st.error("❌ No valid hospitals found. Please check your facility IDs.")
        # Add a button to reset and try again
        if st.button("🔄 Clear and Try Again"):
            st.session_state.results_calculated = False
            st.rerun()
        return

    # Add clear button at top right
    col_header, col_clear = st.columns([6, 1])
    with col_header:
        st.success(f"✅ Successfully processed {len(results_list)} hospitals!")
    with col_clear:
        if st.button("🔄 New Analysis", help="Clear results and start over"):
            st.session_state.results_calculated = False
            st.session_state.results_list = None
            st.session_state.summary_df = None
            st.session_state.facility_ids = []
            st.rerun()

    # Summary metrics - Calculate executive summary for accurate metrics
    if len(results_list) > 1:
        exec_summary = create_executive_summary(results_list)

    st.header("System-Wide Star Rating Insights")

    # Single row with all 4 metrics
    col1, col2, col3, col4 = st.columns(4)

    # Calculate rated count for use in all metrics
    rated_count = summary_df['Star_Rating'].notna().sum()

    with col1:
        st.metric("Hospitals Analyzed", len(facility_ids))

    with col2:
        if rated_count > 0:
            avg_stars = summary_df[summary_df['Star_Rating'].notna()]['Star_Rating'].mean()
            st.metric("Average Star Rating", f"{avg_stars:.2f}")
        else:
            st.metric("Average Star Rating", "N/A")

    with col3:
        # Count hospitals below 3 stars (1 or 2 star ratings)
        if rated_count > 0:
            hospitals_below_3 = ((summary_df['Star_Rating'] == 1) | (summary_df['Star_Rating'] == 2)).sum()
            pct_of_system = hospitals_below_3/len(results_list)*100

            st.metric(
                "Hospitals Below 3 Stars",
                f"{hospitals_below_3:.0f} hospitals"
            )
            st.markdown(
                f"<span style='color:#d32f2f; background-color:#ffeded; padding:2px 8px; "
                f"border-radius:12px; font-size:14px; display:inline-block;'>{pct_of_system:.0f}% of system</span>",
                unsafe_allow_html=True
            )
        else:
            st.metric("Hospitals Below 3 Stars", "0")

    with col4:
        # Quick win opportunities - use exec summary for accurate calculation
        if len(results_list) > 1 and exec_summary is not None:
            quick_wins = len(exec_summary.get('Improvement_Opportunities', pd.DataFrame()))
            st.metric(
                "Quick Win Opportunities",
                f"{quick_wins} hospitals",
                help="Hospitals close to next star rating"
            )
        else:
            st.metric("Quick Win Opportunities", "0")

    st.markdown("---")

    # Single hospital financial analysis (if financial data available)
    if st.session_state.get('financial_data') is not None and not st.session_state.financial_data.empty:
        financial_df = st.session_state.financial_data

        if len(facility_ids) == 1:
            st.markdown("### Financial Impact Breakdown")

            facility_id = str(facility_ids[0])
            fin_row = financial_df[financial_df['facility_id'].astype(str) == facility_id]

            if not fin_row.empty:
                fin = fin_row.iloc[0]

                # Waterfall Chart (ISSUE #4: Fixed to show final revenue, not net impact)
                revenue = fin['net_medicare_revenue']
                hvbp = fin.get('hvbp_dollars', 0)
                hrrp = fin.get('hrrp_dollars', 0)
                hacrp = fin.get('hacrp_dollars', 0)
                total_change = hvbp + hrrp + hacrp
                final_revenue = revenue + total_change

                import plotly.graph_objects
                fig = plotly.graph_objects.Figure(plotly.graph_objects.Waterfall(
                    name="Financial Impact",
                    orientation="v",
                    measure=["absolute", "relative", "relative", "relative", "total"],
                    x=["Medicare<br>Revenue", "HVBP<br>Impact", "HRRP<br>Impact", "HACRP<br>Impact", "Final Revenue<br>After Adjustments"],
                    textposition="outside",
                    text=[
                        f"${revenue/1e6:.1f}M",
                        f"${hvbp/1e6:.1f}M",
                        f"${hrrp/1e6:.1f}M",
                        f"${hacrp/1e6:.1f}M",
                        f"${final_revenue/1e6:.1f}M<br>(${total_change/1e6:+.1f}M)"
                    ],
                    y=[revenue, hvbp, hrrp, hacrp, final_revenue],
                    connector={"line": {"color": "rgb(63, 63, 63)", "width": 2}},
                    decreasing={"marker": {"color": "#ff4444", "line": {"width": 0}}},
                    increasing={"marker": {"color": "#44ff44", "line": {"width": 0}}},
                    totals={"marker": {"color": "#4444ff", "line": {"width": 0}}},
                    textfont={"size": 14}
                ))

                # Calculate y-axis range to zoom in on the changes
                min_val = min(revenue, final_revenue, revenue + hvbp, revenue + hvbp + hrrp)
                max_val = max(revenue, final_revenue, revenue + hvbp, revenue + hvbp + hrrp)
                y_min = min_val * 0.95
                y_max = max_val * 1.05

                fig.update_layout(
                    title={
                        "text": "How Quality Performance Impacts Revenue",
                        "font": {"size": 18}
                    },
                    showlegend=False,
                    height=600,
                    margin=dict(t=100, b=100, l=80, r=80),
                    xaxis={
                        "tickfont": {"size": 12},
                        "title": {"font": {"size": 14}}
                    },
                    yaxis={
                        "tickfont": {"size": 12},
                        "title": {"text": "Revenue ($)", "font": {"size": 14}},
                        "tickformat": "$,.0f",
                        "range": [y_min, y_max]
                    }
                )

                st.plotly_chart(fig, use_container_width=True)

                # Program Details
                st.markdown("### Program-by-Program Analysis")

                prog_col1, prog_col2, prog_col3 = st.columns(3)

                with prog_col1:
                    st.markdown("#### 🏥 HVBP")
                    st.markdown("*Hospital Value-Based Purchasing*")
                    status = fin.get('hvbp_status', 'UNKNOWN')
                    status_color = "🔴" if status == "PENALTY" else "🟢" if status == "BONUS" else "⚪"
                    st.markdown(f"**Status:** {status_color} {status}")
                    st.markdown(f"**Impact:** ${hvbp:,.0f}")
                    st.markdown(f"**Percentage:** {fin.get('hvbp_pct', 0):.2f}%")
                    opp = fin.get('hvbp_opportunity', 0)
                    if opp > 0:
                        st.markdown(f"**Opportunity:** ${opp:,.0f}")

                with prog_col2:
                    st.markdown("#### 🔄 HRRP")
                    st.markdown("*Readmissions Reduction*")
                    status = fin.get('hrrp_status', 'UNKNOWN')
                    status_color = "🔴" if status == "PENALTY" else "🟢" if status == "BONUS" else "⚪"
                    st.markdown(f"**Status:** {status_color} {status}")
                    st.markdown(f"**Impact:** ${hrrp:,.0f}")
                    st.markdown(f"**Percentage:** {fin.get('hrrp_pct', 0):.2f}%")
                    opp = fin.get('hrrp_opportunity', 0)
                    if opp > 0:
                        st.markdown(f"**Opportunity:** ${opp:,.0f}")

                with prog_col3:
                    st.markdown("#### 🦠 HACRP")
                    st.markdown("*Hospital-Acquired Conditions*")
                    status = fin.get('hacrp_status', 'UNKNOWN')
                    status_color = "🔴" if status == "PENALTY" else "🟢" if status == "BONUS" else "⚪"
                    st.markdown(f"**Status:** {status_color} {status}")
                    st.markdown(f"**Impact:** ${hacrp:,.0f}")
                    st.markdown(f"**Percentage:** {fin.get('hacrp_pct', 0):.2f}%")
                    opp = fin.get('hacrp_opportunity', 0)
                    if opp > 0:
                        st.markdown(f"**Opportunity:** ${opp:,.0f}")

                # Strategic Priorities - CORRECTED VERSION (EDIT #1-7)
                st.markdown("---")
                st.markdown("### Strategic Priorities: Cross-Program Impact")
                st.caption("Star Rating: July 2025 | HVBP: FY 2026 | HRRP: FY 2026 | HACRP: FY 2025")
                st.markdown("""
                **Ranked by Programs Affected, Then Financial Impact, Then Star Rating Impact**

                These priorities identify measures that hurt BOTH your star rating AND financial programs,
                giving you the highest ROI when improved.
                """)

                # Import the CORRECTED strategic priorities module
                from strategic_priorities import (
                    calculate_strategic_priorities_corrected,
                    create_cross_program_matrix,
                    create_priorities_table,
                    create_priorities_visualization,
                    MEASUREMENT_PERIODS,
                    format_priority_for_display
                )

                # Get the star rating results for this hospital
                matching_result = None
                for result in results_list:
                    if str(result['provider_id']) == facility_id:
                        matching_result = result
                        break

                if matching_result and matching_result.get('measure_details'):
                    # Get financial data for this hospital
                    fin_dict = None
                    if not fin_row.empty:
                        fin_dict = fin.to_dict()

                    # Calculate strategic priorities using CORRECTED logic (EDIT #4)
                    try:
                        priorities = calculate_strategic_priorities_corrected(
                            matching_result,
                            financial_data=fin_dict
                        )

                        if priorities:
                            # Summary table (EDIT #6)
                            st.markdown(f"#### Top {min(10, len(priorities))} Strategic Priorities")
                            st.caption("*Measures ranked by programs affected, then financial impact, then star rating impact*")

                            priorities_table = create_priorities_table(priorities)
                            st.dataframe(priorities_table, use_container_width=True, hide_index=True)

                            # Cross-Program Matrix (EDIT #5)
                            st.markdown("---")
                            st.markdown("#### Cross-Program Impact Matrix")
                            st.caption("Shows which measures affect which programs")

                            matrix_df = create_cross_program_matrix(priorities)
                            st.dataframe(matrix_df, use_container_width=True, hide_index=True)

                            st.markdown("""
                            **LEGEND:**
                            - 🔴 HIGH = Significant negative impact in this program
                            - 🟡 MED = Moderate negative impact in this program
                            - 🟢 LOW = Minor negative impact in this program
                            - \- = Not included in this program
                            """)

                            # Key Insight Box
                            cascading_measures = [p for p in priorities if p['is_cascading']]
                            if cascading_measures:
                                total_financial = sum(p['total_financial_impact'] for p in cascading_measures)
                                total_star = sum(abs(p['star_impact_points']) for p in cascading_measures)

                                st.info(f"""
**KEY INSIGHT**

Your hospital has {len(cascading_measures)} measures affecting 3 programs simultaneously:
{chr(10).join(['• ' + p['measure_name'] + ': ' + ' + '.join(p['programs']) + f" = ${p['total_financial_impact']:,.0f} at risk" for p in cascading_measures[:3]])}

Fixing these {len(cascading_measures)} measures would improve performance across 3 programs.
Combined potential value: ${total_financial:,.0f} + {total_star:.3f} star rating points
                                """)

                            # Detailed Priority Boxes (EDIT #1, #3: Added dates, removed emojis)
                            st.markdown("---")
                            st.markdown("#### Detailed Priority Analysis")

                            for i, priority in enumerate(priorities[:10], 1):
                                formatted = format_priority_for_display(priority, i)

                                # Cascading badge (EDIT #3: removed emoji)
                                cascading_text = ' - CASCADING IMPACT (3+ programs)' if formatted['is_cascading'] else ''

                                with st.expander(
                                    f"{formatted['symbol']} PRIORITY #{formatted['rank']}: {formatted['title']}{cascading_text}",
                                    expanded=(i <= 3)
                                ):
                                    # Programs and severity
                                    st.markdown(f"**Programs Affected:** {formatted['programs']}")
                                    st.markdown(f"**Severity:** {formatted['symbol']} {formatted['severity']}")
                                    st.markdown(f"**Measure Group:** {formatted['group']}")

                                    if formatted['is_cascading']:
                                        st.warning("**CASCADING IMPACT**: This measure affects 3+ programs. Fixing it creates benefits across multiple quality programs!")

                                    st.markdown("---")

                                    # Two-column layout
                                    col1, col2 = st.columns([3, 2])

                                    with col1:
                                        st.markdown("**CURRENT PERFORMANCE**")
                                        st.markdown(f"- Your Score: {formatted['current_score']}")
                                        st.markdown(f"- Group Average: {formatted['group_average']}")
                                        st.markdown(f"- Performance Gap: {formatted['performance_gap']} (below average)")

                                        st.markdown("**STAR RATING IMPACT**")
                                        st.markdown(f"- Impact: {formatted['star_impact']} lost")
                                        st.markdown(f"- Measurement: {MEASUREMENT_PERIODS['STAR_RATING']['display']}")

                                        st.markdown("**FINANCIAL IMPACT**")
                                        st.markdown(f"- Total: {formatted['financial_total']}")
                                        st.markdown(f"- Breakdown: {formatted['financial_breakdown']}")

                                    with col2:
                                        st.metric(
                                            "Programs Affected",
                                            formatted['program_count'],
                                            help="More programs = higher ROI"
                                        )

                                        st.markdown(f"**Why Priority #{formatted['rank']}?**")
                                        st.markdown(f"""
1. Affects {formatted['program_count']} programs
2. {formatted['severity']} severity
3. High ROI when improved
                                        """)

                                    st.markdown("---")

                                    # Recommended actions (EDIT #3: removed emoji)
                                    st.markdown("**RECOMMENDED ACTIONS**")
                                    st.markdown(f"""
1. Review protocols for {formatted['title']}
2. Benchmark against high-performing hospitals
3. Implement evidence-based interventions for {formatted['group']}
4. Monitor monthly to track improvement
5. Calculate ROI after improvement
                                    """)

                                    # Potential gain
                                    st.success(f"""
**Potential Gain if Improved to National Average:**
- Star Rating: +{formatted['star_impact']}
- Financial: {formatted['financial_total']} recovered/gained
- Programs: {formatted['program_count']} quality programs benefit
                                    """)

                            # Visualization (EDIT #6: Single clean visualization)
                            st.markdown("---")
                            st.markdown("#### Strategic Priorities Visualization")
                            st.caption("Programs, financial impact, and star impact shown separately (not stacked)")

                            fig_priorities = create_priorities_visualization(priorities)
                            st.plotly_chart(fig_priorities, use_container_width=True)

                        else:
                            st.success("Excellent performance! No measures found with negative impact on star ratings.")
                            st.info("Your hospital is performing at or above the national average across all reported measures.")

                    except Exception as e:
                        st.error(f"Error calculating strategic priorities: {e}")
                        import traceback
                        st.code(traceback.format_exc())

                else:
                    st.info("Complete star rating calculation to see strategic priorities")

    # ==================================================================
    # EXECUTIVE SUMMARY SECTIONS (for multi-hospital analysis)
    # ==================================================================

    if len(results_list) > 1:
        # Create hospital names mapping from financial data (truncated for display)
        hospital_names = {}
        if st.session_state.get('financial_data') is not None:
            financial_df = st.session_state.financial_data
            if 'hospital_name' in financial_df.columns or 'Hospital Name' in financial_df.columns:
                name_col = 'hospital_name' if 'hospital_name' in financial_df.columns else 'Hospital Name'
                # Truncate names to 25 characters
                hospital_names = {
                    str(fid): (name[:25] + '...' if len(name) > 25 else name)
                    for fid, name in zip(financial_df['facility_id'], financial_df[name_col])
                }

        # Visualizations
        st.subheader("Visualizations")

        tab1, tab2, tab3 = st.tabs(["Star Distribution", "Peer Comparison", "Performance Heatmap"])

        with tab1:
            # Star distribution works for all hospitals
            fig_dist = create_star_distribution_chart(results_list)
            st.plotly_chart(fig_dist, use_container_width=True)

        with tab2:
            # Peer comparison only for hospitals with ratings
            if rated_count > 0:
                fig_peer = create_peer_comparison_chart(results_list, hospital_names)
                st.plotly_chart(fig_peer, use_container_width=True)
            else:
                st.info("Peer comparison requires at least one hospital with a valid star rating")

        with tab3:
            # Performance heatmap only for hospitals with ratings
            if rated_count > 0:
                fig_heatmap = create_measure_heatmap(results_list, hospital_names)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Performance heatmap requires at least one hospital with a valid star rating")

        st.markdown("---")

        # SECTION 2: PERFORMANCE BY MEASURE GROUP
        st.header("Performance by Measure Group")

        group_perf = exec_summary['Group_Performance']

        if not group_perf.empty:
            # Create a horizontal bar chart
            fig_groups = go.Figure()

            # Sort by System_Average
            group_perf_sorted = group_perf.sort_values('System_Average')

            colors = ['red' if x < -0.5 else 'orange' if x < 0 else 'green'
                      for x in group_perf_sorted['System_Average']]

            fig_groups.add_trace(go.Bar(
                y=group_perf_sorted['Measure_Group'],
                x=group_perf_sorted['System_Average'],
                orientation='h',
                marker_color=colors,
                text=[f"{x:.2f}" for x in group_perf_sorted['System_Average']],
                textposition='outside',
            ))

            fig_groups.update_layout(
                title="System-Wide Group Performance (vs National Average)",
                xaxis_title="Average Z-Score",
                yaxis_title="",
                height=400,
                showlegend=False,
                xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
            )

            st.plotly_chart(fig_groups, use_container_width=True, key="system_group_performance")

            # Show detailed table
            st.subheader("Detailed Group Performance")

            display_groups = group_perf[['Measure_Group', 'System_Average', 'Pct_Below_National', 'Priority']].copy()
            display_groups.columns = ['Measure Group', 'System Average', '% Hospitals Below National', 'Priority']

            st.dataframe(display_groups, use_container_width=True, hide_index=True)

        st.markdown("---")

        # SECTION 3: TOP PROBLEM MEASURES
        st.header("Top Problem Measures (Weighted by Impact)")

        st.markdown("""
        **Ranked by Impact Score** - combines % of hospitals affected, severity of underperformance, and measure weight in star ratings.

        **Severity Levels:**
        - 🔴 **CRITICAL** = Average performance >1.0 standard deviations below national average
        - 🟡 **HIGH** = Average performance 0.5-1.0 standard deviations below national average
        - ⚪ **MODERATE** = Average performance <0.5 standard deviations below national average
        """)

        # Display top 10 problem measures
        top_measures = exec_summary['Top_Problem_Measures'].head(10)

        if not top_measures.empty:
            # Format the dataframe for display
            display_measures = top_measures[['Measure_Name', 'Group', 'Pct_System', 'Impact_Score', 'Severity']].copy()
            display_measures.columns = ['Measure', 'Category', '% of System Affected', 'Impact Score', 'Severity']

            # Style the dataframe
            def highlight_severity(row):
                if row['Severity'] == 'CRITICAL':
                    return ['background-color: #ffcccc'] * len(row)
                elif row['Severity'] == 'HIGH':
                    return ['background-color: #ffe6cc'] * len(row)
                else:
                    return ['background-color: #ffffcc'] * len(row)

            styled_df = display_measures.style.apply(highlight_severity, axis=1)

            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            # Highlight the TOP 3
            st.subheader("Top 3 Priority Measures:")

            for idx, (_, row) in enumerate(top_measures.head(3).iterrows()):
                with st.expander(f"#{idx+1}: {row['Measure_Name']} - Impact Score: {row['Impact_Score']:.1f}", expanded=(idx==0)):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"""
                        **Category:** {row['Group']}
                        **Hospitals Affected:** {row['Hospitals_Affected']} ({row['Pct_System']})
                        **Average Performance:** {row['Avg_Z_Score']:.2f} standard deviations below national average
                        **Severity:** {row['Severity']}
                        """)

                    with col2:
                        # Show impact score as a gauge/metric
                        st.metric("Impact Score", f"{row['Impact_Score']:.1f}")
                        st.caption("Higher = Greater impact on star ratings")
        else:
            st.info("No significant problem measures identified across the system.")

        st.markdown("---")

        # SECTION 3: QUICK WINS
        st.header("Quick Win Opportunities")

        improvement_opps = exec_summary.get('Improvement_Opportunities', pd.DataFrame())

        if not improvement_opps.empty:
            st.markdown("""
            These hospitals are **closest to achieving the next star rating**.
            Small improvements in their focus areas could result in immediate star rating gains.
            """)

            # Show top 10 opportunities
            top_opps = improvement_opps.head(10)

            # Format for display
            display_opps = top_opps[['Facility_ID', 'Current_Rating', 'Target_Rating', 'Gap_to_Next_Star', 'Primary_Focus_Area', 'Feasibility']].copy()

            # Map facility IDs to hospital names from financial data
            if st.session_state.get('financial_data') is not None:
                financial_df = st.session_state.financial_data
                if 'hospital_name' in financial_df.columns or 'Hospital Name' in financial_df.columns:
                    name_col = 'hospital_name' if 'hospital_name' in financial_df.columns else 'Hospital Name'
                    # Create mapping with truncated names
                    name_map = {
                        str(fid): (name[:25] + '...' if len(name) > 25 else name)
                        for fid, name in zip(financial_df['facility_id'], financial_df[name_col])
                    }
                    display_opps['Hospital'] = display_opps['Facility_ID'].astype(str).map(name_map).fillna(display_opps['Facility_ID'])

                    # Also get the full name for the top opportunity callout
                    top_opp_name = name_map.get(str(top_opps.iloc[0]['Facility_ID']), str(top_opps.iloc[0]['Facility_ID']))
                else:
                    display_opps['Hospital'] = display_opps['Facility_ID']
                    top_opp_name = str(top_opps.iloc[0]['Facility_ID'])
            else:
                display_opps['Hospital'] = display_opps['Facility_ID']
                top_opp_name = str(top_opps.iloc[0]['Facility_ID'])

            # Remove Facility_ID column and reorder
            display_opps = display_opps[['Hospital', 'Current_Rating', 'Target_Rating', 'Gap_to_Next_Star', 'Primary_Focus_Area', 'Feasibility']]
            display_opps.columns = ['Hospital', 'Current ⭐', 'Target ⭐', 'Gap to Next Star', 'Focus Area', 'Feasibility']

            # Add star emojis
            display_opps['Current ⭐'] = display_opps['Current ⭐'].apply(lambda x: '⭐' * x)
            display_opps['Target ⭐'] = display_opps['Target ⭐'].apply(lambda x: '⭐' * x)

            # Color code by feasibility
            def color_feasibility(val):
                if val == 'HIGH':
                    return 'background-color: #ccffcc'
                elif val == 'MEDIUM':
                    return 'background-color: #ffffcc'
                else:
                    return ''

            styled_opps = display_opps.style.applymap(color_feasibility, subset=['Feasibility'])

            st.dataframe(styled_opps, use_container_width=True, hide_index=True)

            # Callout for the #1 opportunity
            top_opp = improvement_opps.iloc[0]

            st.success(f"""
            **BEST OPPORTUNITY:** {top_opp_name}
            Currently {top_opp['Current_Rating']} stars, only **{top_opp['Gap_to_Next_Star']:.3f} points** away from {top_opp['Target_Rating']} stars!
            **Focus on:** {top_opp['Primary_Focus_Area']} (Score: {top_opp['Focus_Area_Score']:.2f})
            """)
        else:
            st.info("No immediate quick win opportunities identified. Most hospitals need significant improvements to advance.")

        st.markdown("---")

        # SECTION 5: INTEGRATED FINANCIAL & CROSS-PROGRAM ANALYSIS (if financial data uploaded)
        if st.session_state.financial_data is not None:
            st.header("System-Wide Quality Performance Financial Insights")

            with st.spinner("Analyzing cross-program impacts and financial ROI..."):
                # Create integrated system assessment
                system_assessment = create_system_integrated_assessment(
                    results_list,
                    st.session_state.financial_data
                )
                st.session_state.integrated_assessment = system_assessment

                # Financial Overview
                st.subheader("Financial Impact Summary")

                fin_overview = system_assessment['financial_overview']

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Medicare Revenue",
                        f"${fin_overview['total_medicare_revenue']/1e6:.1f}M"
                    )

                with col2:
                    impact_pct = (fin_overview['total_current_impact']/fin_overview['total_medicare_revenue']*100)
                    # Determine colors for pill background and text: red for negative, green for positive
                    if fin_overview['total_current_impact'] < 0:
                        text_color = "#d32f2f"
                        bg_color = "#ffeded"
                    elif fin_overview['total_current_impact'] > 0:
                        text_color = "#09ab3b"
                        bg_color = "#e8f5e9"
                    else:
                        text_color = "#808495"
                        bg_color = "#f0f2f6"

                    st.metric(
                        "Current Impact",
                        f"${fin_overview['total_current_impact']/1e6:.1f}M"
                    )
                    st.markdown(
                        f"<span style='color:{text_color}; background-color:{bg_color}; padding:2px 8px; "
                        f"border-radius:12px; font-size:14px; display:inline-block;'>{impact_pct:+.1f}% of revenue</span>",
                        unsafe_allow_html=True
                    )

                with col3:
                    st.metric(
                        "At-Risk Revenue",
                        f"${fin_overview['total_at_risk']/1e6:.1f}M",
                        help="Maximum penalty exposure across all programs: 6% of Medicare revenue (HVBP 2% + HRRP 3% + HACRP 1%)."
                    )

                with col4:
                    st.metric(
                        "Opportunity",
                        f"${fin_overview['total_opportunity']/1e6:.1f}M",
                        help="Potential financial upside from eliminating current penalties and achieving HVBP bonus (2% of Medicare revenue)."
                    )

                # Program Penalties
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("HVBP Penalties", fin_overview['hvbp_penalties'])
                with col2:
                    st.metric("HRRP Penalties", fin_overview['hrrp_penalties'])
                with col3:
                    st.metric("HACRP Penalties", fin_overview['hacrp_penalties'])

                st.markdown("---")

                # System-Wide Financial Waterfall Chart
                st.subheader("System Financial Impact Waterfall")

                # Get system financial data
                revenue = fin_overview['total_medicare_revenue']
                hvbp = financial_data['hvbp_dollars'].sum()
                hrrp = financial_data['hrrp_dollars'].sum()
                hacrp = financial_data['hacrp_dollars'].sum()
                total_change = hvbp + hrrp + hacrp
                final_revenue = revenue + total_change

                # Create waterfall chart with improved readability
                fig = go.Figure(go.Waterfall(
                    name="Financial Impact",
                    orientation="v",
                    measure=["absolute", "relative", "relative", "relative", "total"],
                    x=["Total Medicare<br>Revenue", "HVBP<br>Impact", "HRRP<br>Impact", "HACRP<br>Impact", "Final Revenue<br>After Adjustments"],
                    textposition="outside",
                    text=[
                        f"${revenue/1e6:.1f}M",
                        f"${hvbp/1e6:.1f}M",
                        f"${hrrp/1e6:.1f}M",
                        f"${hacrp/1e6:.1f}M",
                        f"${final_revenue/1e6:.1f}M<br>(${total_change/1e6:+.1f}M)"
                    ],
                    y=[revenue, hvbp, hrrp, hacrp, final_revenue],
                    connector={"line": {"color": "rgb(63, 63, 63)", "width": 2}},
                    decreasing={"marker": {"color": "#ff4444", "line": {"width": 0}}},
                    increasing={"marker": {"color": "#44ff44", "line": {"width": 0}}},
                    totals={"marker": {"color": "#4444ff", "line": {"width": 0}}},
                    textfont={"size": 14}
                ))

                # Calculate y-axis range to zoom in on the changes
                min_val = min(revenue, final_revenue, revenue + hvbp, revenue + hvbp + hrrp)
                max_val = max(revenue, final_revenue, revenue + hvbp, revenue + hvbp + hrrp)
                y_min = min_val * 0.95  # Start at 95% of minimum value
                y_max = max_val * 1.05  # End at 105% of maximum value

                fig.update_layout(
                    title={
                        "text": "System-Wide Quality Performance Financial Impact",
                        "font": {"size": 18}
                    },
                    showlegend=False,
                    height=600,
                    margin=dict(t=100, b=100, l=80, r=80),
                    xaxis={
                        "tickfont": {"size": 12},
                        "title": {"font": {"size": 14}}
                    },
                    yaxis={
                        "tickfont": {"size": 12},
                        "title": {"text": "Revenue ($)", "font": {"size": 14}},
                        "tickformat": "$,.0f",
                        "range": [y_min, y_max]
                    }
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")

                # Cross-Program Impact Analysis
                st.subheader("Cross-Program Impact Analysis")

                st.markdown("""
                **Ranked by Priority Score** - combines number of programs affected, hospitals affected, and severity of underperformance.

                **Priority Score Calculation:**
                - Priority Score = Programs Affected × Hospitals Affected × |Avg Performance (SD)|
                - Higher scores = more urgent (affects more programs, more hospitals, worse performance)
                """)

                integrated = system_assessment['integrated_metrics']

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Problem Measures", integrated['total_unique_problem_measures'])
                with col2:
                    st.metric("Cross-Program Measures", integrated['cross_program_measures'],
                             help="Measures affecting 2+ programs")
                with col3:
                    st.metric("Cascading Impact Measures", integrated['cascading_impact_measures'],
                             help="Measures affecting 3+ programs (highest ROI)")

                # Top Strategic Priorities
                st.subheader("Top Strategic Priorities")

                cross_program_df = create_cross_program_impact_report(system_assessment)

                if not cross_program_df.empty:
                    # Show top 10 measures
                    top_10 = cross_program_df.head(10)
                    st.dataframe(top_10, use_container_width=True, hide_index=True)

                    # Callout for top priority
                    if len(top_10) > 0:
                        top = top_10.iloc[0]
                        st.error(f"""
                        **TOP PRIORITY:** {top['Measure']}

                        - **Programs Affected:** {top['Programs Affected']}
                        - **Hospitals Affected:** {top['Hospitals Affected']}
                        - **Avg Performance:** {top['Avg Performance']}
                        - **Priority Score:** {top['Priority Score']}
                        """)
                else:
                    st.info("No cross-program measures identified in your problem areas.")

        st.markdown("---")

    # Results table
    st.header("Hospital Level Details")
    st.subheader("Hospital Star Ratings")

    # Format the summary dataframe for display
    display_df = summary_df.copy()

    # Try to add hospital names from financial data if available
    if st.session_state.get('financial_data') is not None and 'Facility_ID' in display_df.columns:
        financial_df = st.session_state.financial_data
        if 'hospital_name' in financial_df.columns or 'Hospital Name' in financial_df.columns:
            # Create a mapping from facility_id to hospital name
            name_col = 'hospital_name' if 'hospital_name' in financial_df.columns else 'Hospital Name'
            name_map = dict(zip(
                financial_df['facility_id'].astype(str),
                financial_df[name_col]
            ))

            # Map facility IDs to hospital names
            display_df['Hospital_Name'] = display_df['Facility_ID'].astype(str).map(name_map)
            # Fill any NaNs with 'Unknown'
            display_df['Hospital_Name'] = display_df['Hospital_Name'].fillna('Unknown')

    # Remove eligibility column if it exists
    if 'Is_Eligible' in display_df.columns:
        display_df = display_df.drop(columns=['Is_Eligible'])

    if 'Star_Rating' in display_df.columns:
        display_df['Stars'] = display_df['Star_Rating'].apply(format_star_rating)
        # Remove the numeric Star_Rating column as it's duplicative with Stars
        display_df = display_df.drop(columns=['Star_Rating'])

    # Reorder columns - Hospital Name first, then stars, then other details
    cols = ['Hospital_Name', 'Stars', 'Summary_Score', 'Peer_Group', 'Problem_Areas', 'Strength_Areas']
    cols = [c for c in cols if c in display_df.columns]
    display_df = display_df[cols + [c for c in display_df.columns if c not in cols]]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(400, (len(display_df) + 1) * 35 + 3)
    )

    # Detailed hospital views
    st.markdown("---")
    st.subheader("Detailed Analysis")

    # Create hospital names mapping from financial data for expander labels
    hospital_names_map = {}
    if st.session_state.get('financial_data') is not None:
        financial_df = st.session_state.financial_data
        if 'hospital_name' in financial_df.columns or 'Hospital Name' in financial_df.columns:
            name_col = 'hospital_name' if 'hospital_name' in financial_df.columns else 'Hospital Name'
            hospital_names_map = {
                str(fid): name
                for fid, name in zip(financial_df['facility_id'], financial_df[name_col])
            }

    for idx, result in enumerate(results_list):
        star_display = format_star_rating(result['star_rating'])

        # Get hospital name from mapping, fallback to provider_id
        hospital_name = hospital_names_map.get(str(result['provider_id']), str(result['provider_id']))
        expander_label = f"{hospital_name} - {star_display}"

        with st.expander(expander_label, expanded=(len(results_list) == 1)):

            # Check if we have a star rating
            if result['star_rating'] is None:
                st.warning(f"**⚠️ Unable to Calculate Star Rating**\n\n**Reason:** {result['eligibility_reason']}")
                if result['summary_score'] is not None:
                    st.info(f"Summary score: {result['summary_score']:.3f}, but insufficient data for star assignment.")
                # Still show available data below
            else:
                # Summary metrics - using native Streamlit components
                st.subheader(f"{'⭐' * result['star_rating']} {result['star_rating']} Star Rating")

            # Performance score with color-coded indicator
            st.subheader("Performance Scores")

            score_col1, score_col2 = st.columns([2, 1])

            with score_col1:
                if result['summary_score'] is not None:
                    st.metric(
                        label="Summary Score",
                        value=f"{result['summary_score']:.3f}",
                        help="Overall performance across all measure groups (0 = national average)"
                    )
                else:
                    st.metric(label="Summary Score", value="N/A")

            with score_col2:
                if result['summary_score'] is not None:
                    if result['summary_score'] < -0.5:
                        st.markdown("**Status:**  \n🔴 Significantly Below Avg")
                    elif result['summary_score'] < 0:
                        st.markdown("**Status:**  \n🟠 Below National Avg")
                    elif result['summary_score'] > 0.5:
                        st.markdown("**Status:**  \n🟢 Significantly Above Avg")
                    else:
                        st.markdown("**Status:**  \n🟡 Above National Avg")

            st.markdown("---")

            # Other metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Peer Group",
                    value=f"{result['peer_group']}",
                    help="Number of measure groups with ≥3 measures"
                )

            with col2:
                total_measures = len(result['measure_details'])
                st.metric(
                    label="Total Measures",
                    value=f"{total_measures}",
                    help=f"{total_measures} out of 46 possible measures"
                )

            with col3:
                groups_with_3plus = len([g for g, c in result['measures_per_group'].items() if c >= 3])
                st.metric(
                    label="Reporting Groups",
                    value=f"{groups_with_3plus}",
                    help="Groups with ≥3 measures"
                )

            st.markdown("---")

            # Create tabs for different views
            tabs = ["Group Scores", "Measure Details", "Analysis"]

            # Add Financial Impact tab if financial data is available
            has_financial = (st.session_state.get('financial_data') is not None and
                           not st.session_state.financial_data.empty)
            if has_financial:
                tabs.append("Financial Impact")

            tab_objects = st.tabs(tabs)
            detail_tab1 = tab_objects[0]
            detail_tab2 = tab_objects[1]
            detail_tab3 = tab_objects[2]

            with detail_tab1:
                # Group scores chart
                fig_groups = create_group_scores_chart(result)
                st.plotly_chart(fig_groups, use_container_width=True, key=f"group_chart_{result['provider_id']}_{idx}")

                # Group scores table with color-coded performance
                st.subheader("Measure Group Performance")

                group_data = []
                for group_name in MEASURE_GROUPS.keys():
                    score = result['standardized_group_scores'].get(group_name)
                    measures = result['measures_per_group'].get(group_name, 0)
                    if score is not None:
                        # Color-coded performance indicator
                        if score < -0.5:
                            status = "🔴 Significantly Below"
                        elif score < 0:
                            status = "🟠 Below Average"
                        elif score > 0.5:
                            status = "🟢 Significantly Above"
                        else:
                            status = "🟡 Above Average"

                        group_data.append({
                            'Measure Group': group_name,
                            'Score': f"{score:.3f}",
                            'Measures Reported': measures,
                            'Performance': status
                        })

                group_scores_df = pd.DataFrame(group_data)
                st.dataframe(group_scores_df, use_container_width=True, hide_index=True)

            with detail_tab2:
                # Measure details table
                measure_df = create_comparison_table(result)
                st.dataframe(measure_df, use_container_width=True, hide_index=True, height=400)

            with detail_tab3:
                col1, col2 = st.columns(2)

                # Get all measures with valid standardized scores
                measure_scores = []
                for measure_name, details in result['measure_details'].items():
                    if pd.notna(details['standardized']):
                        measure_scores.append({
                            'measure': measure_name,
                            'score': details['standardized']
                        })

                # Sort by score
                measure_scores_sorted = sorted(measure_scores, key=lambda x: x['score'], reverse=True)

                with col1:
                    st.markdown("### Strengths")
                    # Show top 3 best performing measures
                    top_3 = measure_scores_sorted[:3]
                    if top_3:
                        for item in top_3:
                            st.success(f"**{item['measure']}**: {item['score']:.3f} ({abs(item['score']):.2f} SD above average)")
                    else:
                        st.info("No measures available")

                with col2:
                    st.markdown("### Areas for Improvement")
                    # Show top 3 worst performing measures
                    bottom_3 = measure_scores_sorted[-3:][::-1]  # Reverse to show worst first
                    if bottom_3:
                        for item in bottom_3:
                            st.warning(f"**{item['measure']}**: {item['score']:.3f} ({abs(item['score']):.2f} SD below average)")
                    else:
                        st.info("No measures available")

            # Financial Impact tab (only shown if financial data exists)
            if has_financial:
                detail_tab4 = tab_objects[3]

                with detail_tab4:
                    # Get financial data for this hospital
                    facility_id = str(result['provider_id'])
                    financial_row = st.session_state.financial_data[
                        st.session_state.financial_data['facility_id'].astype(str) == facility_id
                    ]

                    if financial_row.empty:
                        st.warning(f"No financial data found for facility {facility_id}")
                    else:
                        fin = financial_row.iloc[0]

                        # Summary metrics at the top
                        st.subheader("Financial Overview")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Medicare Revenue",
                                f"${fin['net_medicare_revenue']:,.0f}",
                                help="Total annual Medicare revenue for this facility"
                            )
                        with col2:
                            total_impact = fin.get('total_current_impact_dollars', 0)
                            impact_pct = (total_impact / fin['net_medicare_revenue'] * 100) if fin['net_medicare_revenue'] > 0 else 0
                            # Determine colors for pill background and text: red for negative, green for positive
                            if total_impact < 0:
                                text_color = "#d32f2f"
                                bg_color = "#ffeded"
                            elif total_impact > 0:
                                text_color = "#09ab3b"
                                bg_color = "#e8f5e9"
                            else:
                                text_color = "#808495"
                                bg_color = "#f0f2f6"

                            st.metric(
                                "Current Financial Impact",
                                f"${total_impact/1e6:.2f}M"
                            )
                            st.markdown(
                                f"<span style='color:{text_color}; background-color:{bg_color}; padding:2px 8px; "
                                f"border-radius:12px; font-size:14px; display:inline-block;'>{impact_pct:+.2f}% of revenue</span>",
                                unsafe_allow_html=True
                            )
                        with col3:
                            total_opp = fin.get('total_opportunity', 0)
                            st.metric(
                                "Improvement Opportunity",
                                f"${total_opp:,.0f}",
                                help="Potential revenue recovery if penalties are eliminated"
                            )

                        st.markdown("---")

                        # Financial Impact Waterfall Chart
                        st.subheader("Financial Impact Breakdown")

                        # Create waterfall chart
                        revenue = fin['net_medicare_revenue']
                        hvbp = fin.get('hvbp_dollars', 0)
                        hrrp = fin.get('hrrp_dollars', 0)
                        hacrp = fin.get('hacrp_dollars', 0)
                        total_change = hvbp + hrrp + hacrp
                        final_revenue = revenue + total_change

                        import plotly.graph_objects
                        fig = plotly.graph_objects.Figure(plotly.graph_objects.Waterfall(
                            name="Financial Impact",
                            orientation="v",
                            measure=["absolute", "relative", "relative", "relative", "total"],
                            x=["Medicare<br>Revenue", "HVBP<br>Impact", "HRRP<br>Impact", "HACRP<br>Impact", "Final Revenue<br>After Adjustments"],
                            textposition="outside",
                            text=[
                                f"${revenue/1e6:.1f}M",
                                f"${hvbp/1e6:.1f}M",
                                f"${hrrp/1e6:.1f}M",
                                f"${hacrp/1e6:.1f}M",
                                f"${final_revenue/1e6:.1f}M<br>(${total_change/1e6:+.1f}M)"
                            ],
                            y=[revenue, hvbp, hrrp, hacrp, final_revenue],
                            connector={"line": {"color": "rgb(63, 63, 63)", "width": 2}},
                            decreasing={"marker": {"color": "#ff4444", "line": {"width": 0}}},
                            increasing={"marker": {"color": "#44ff44", "line": {"width": 0}}},
                            totals={"marker": {"color": "#4444ff", "line": {"width": 0}}},
                            textfont={"size": 14}
                        ))

                        # Calculate y-axis range to zoom in on the changes
                        min_val = min(revenue, final_revenue, revenue + hvbp, revenue + hvbp + hrrp)
                        max_val = max(revenue, final_revenue, revenue + hvbp, revenue + hvbp + hrrp)
                        y_min = min_val * 0.95
                        y_max = max_val * 1.05

                        fig.update_layout(
                            title={
                                "text": "Financial Impact Breakdown",
                                "font": {"size": 18}
                            },
                            showlegend=False,
                            height=600,
                            margin=dict(t=100, b=100, l=80, r=80),
                            xaxis={
                                "tickfont": {"size": 12},
                                "title": {"font": {"size": 14}}
                            },
                            yaxis={
                                "tickfont": {"size": 12},
                                "title": {"text": "Revenue ($)", "font": {"size": 14}},
                                "tickformat": "$,.0f",
                                "range": [y_min, y_max]
                            }
                        )

                        st.plotly_chart(fig, use_container_width=True, key=f"waterfall_{facility_id}_{idx}")

                        st.markdown("---")

                        # Detailed Program Breakdown
                        st.subheader("Program-by-Program Impact")

                        st.markdown("""
                        **How Current Impact is Calculated:**
                        Each CMS quality program has specific penalties and bonuses based on performance.
                        Below is the breakdown for this facility.
                        """)

                        # HVBP
                        with st.expander("HVBP (Hospital Value-Based Purchasing)", expanded=True):
                            col1, col2 = st.columns(2)
                            with col1:
                                status = fin.get('hvbp_status', 'UNKNOWN')
                                st.write(f"- **Status:** {status}")
                                st.write(f"- **Percentage:** {fin.get('hvbp_pct', 0):.2f}%")
                                st.write(f"- **Dollar Impact:** ${fin.get('hvbp_dollars', 0):,.0f}")
                            with col2:
                                opp = fin.get('hvbp_opportunity', 0)
                                st.write(f"- **Opportunity:** ${opp:,.0f}")
                                st.write(f"- **Improvement Potential:** {(opp/revenue*100) if revenue > 0 else 0:.2f}% of revenue")

                        # HRRP
                        with st.expander("HRRP (Hospital Readmissions Reduction Program)"):
                            col1, col2 = st.columns(2)
                            with col1:
                                status = fin.get('hrrp_status', 'UNKNOWN')
                                st.write(f"- **Status:** {status}")
                                st.write(f"- **Percentage:** {fin.get('hrrp_pct', 0):.2f}%")
                                st.write(f"- **Dollar Impact:** ${fin.get('hrrp_dollars', 0):,.0f}")
                            with col2:
                                opp = fin.get('hrrp_opportunity', 0)
                                st.write(f"- **Opportunity:** ${opp:,.0f}")
                                st.write(f"- **Improvement Potential:** {(opp/revenue*100) if revenue > 0 else 0:.2f}% of revenue")

                        # HACRP
                        with st.expander("HACRP (Hospital-Acquired Condition Reduction Program)"):
                            col1, col2 = st.columns(2)
                            with col1:
                                status = fin.get('hacrp_status', 'UNKNOWN')
                                st.write(f"- **Status:** {status}")
                                st.write(f"- **Percentage:** {fin.get('hacrp_pct', 0):.2f}%")
                                st.write(f"- **Dollar Impact:** ${fin.get('hacrp_dollars', 0):,.0f}")
                            with col2:
                                opp = fin.get('hacrp_opportunity', 0)
                                st.write(f"- **Opportunity:** ${opp:,.0f}")
                                st.write(f"- **Improvement Potential:** {(opp/revenue*100) if revenue > 0 else 0:.2f}% of revenue")

    # Export section
    st.markdown("---")
    st.header("📥 Export Detailed Reports")

    st.markdown("""
    The reports below contain additional details and are formatted for sharing with leadership:
    - **Excel:** Complete data with all calculations, formulas, and detailed measure-level information
    - **PDF:** Executive summary formatted for C-suite presentation
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        # Excel export
        try:
            # Create executive summary if multiple hospitals
            exec_summary = create_executive_summary(results_list) if len(results_list) > 1 else None
            excel_buffer = export_to_excel(results_list, exec_summary=exec_summary)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                label="📊 Download Full Excel Report",
                data=excel_buffer,
                file_name=f"star_ratings_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error generating Excel: {e}")
            import traceback
            st.code(traceback.format_exc())

    with col2:
        # CSV export
        csv_buffer = io.StringIO()
        summary_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📄 Download CSV Summary",
            data=csv_buffer.getvalue(),
            file_name=f"star_ratings_summary_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col3:
        # PDF export - single hospital or multi-hospital system summary
        if len(results_list) == 1:
            # Single hospital PDF with integrated financial & cross-program analysis
            try:
                # Get financial data and integrated assessment if available
                financial_data = st.session_state.get('financial_data')
                integrated_assessment = st.session_state.get('integrated_assessment')

                pdf_buffer = generate_executive_summary_pdf(
                    results_list[0],
                    financial_data=financial_data,
                    integrated_assessment=integrated_assessment
                )
                st.download_button(
                    label="📄 Download Integrated Assessment (PDF)",
                    data=pdf_buffer,
                    file_name=f"integrated_assessment_{results_list[0]['provider_id']}_{timestamp}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
        else:
            # Multi-hospital system-wide PDF with integrated financial & cross-program analysis
            try:
                exec_summary = create_executive_summary(results_list)
                financial_data = st.session_state.get('financial_data')
                integrated_assessment = st.session_state.get('integrated_assessment')

                pdf_buffer = generate_system_executive_summary_pdf(
                    results_list,
                    exec_summary,
                    financial_data=financial_data,
                    integrated_assessment=integrated_assessment
                )
                st.download_button(
                    label="📊 Download Integrated System Assessment (PDF)",
                    data=pdf_buffer,
                    file_name=f"integrated_system_assessment_{len(results_list)}_hospitals_{timestamp}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating system PDF: {e}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p><b>CMS Hospital Star Rating Calculator</b> | Data: CMS July 2025 | Methodology: v4.1</p>
        <p>Built with Streamlit 🎈 | For questions or support, contact your administrator</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
