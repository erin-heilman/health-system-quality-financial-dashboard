"""
Integrated Quality & Financial Risk Assessment Tool

Single-page application for C-suite executives combining:
- User-provided financial data (CSV or manual entry)
- CMS public star rating data (auto-fetched)
- Cross-program impact analysis
- ROI-prioritized recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

# Import our modules
from star_rating_calculator import load_data, process_hospitals, MEASURE_GROUPS
from financial_calculator import (
    calculate_total_quality_impact,
    get_system_financial_summary,
    get_hospital_financial_summary
)
from integrated_assessment import (
    create_system_integrated_assessment,
    create_integrated_hospital_assessment,
    create_executive_summary_data
)
from visualizations import (
    create_group_scores_chart,
    create_star_distribution_chart,
    create_peer_comparison_chart
)
from executive_dashboard_pdf import generate_executive_dashboard_pdf
from excel_export import export_to_excel
import plotly.graph_objects as go
import plotly.express as px

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Integrated Quality & Financial Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# CUSTOM CSS
# ==============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4788;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f4788;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f4788;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    .info-box {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4788;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# FINANCIAL DATA PARSER (Updated for new column names)
# ==============================================================================

def parse_financial_csv_new_format(file_or_df):
    """
    Parse financial CSV with new column names.

    Expected columns:
    - Hospital Name
    - facility_id
    - Net Medicare Revenue
    - FY2025 Medicare Hospital-Acquired Condition Adjustment
    - Est. FY2025 Revenue Adjustment Due to Hospital-Acquired Condition Penalty
    - FY2026 Medicare Value-Based Purchasing Adjustment
    - Est. FY2026 Revenue Adjustment Due to Value-Based Purchasing
    - FY2026 Medicare Readmission Rate Penalty
    - Est. FY2026 Revenue Loss Due to Readmission Penalty
    """
    if isinstance(file_or_df, pd.DataFrame):
        df = file_or_df.copy()
    else:
        df = pd.read_csv(file_or_df)

    # Map new column names to standard names
    column_mapping = {
        'Hospital Name': 'hospital_name',
        'facility_id': 'facility_id',
        'Net Medicare Revenue': 'net_medicare_revenue',
        'FY2025 Medicare Hospital-Acquired Condition Adjustment': 'hacrp_pct',
        'Est. FY2025 Revenue Adjustment Due to Hospital-Acquired Condition Penalty': 'hacrp_dollars',
        'FY2026 Medicare Value-Based Purchasing Adjustment': 'hvbp_pct',
        'Est. FY2026 Revenue Adjustment Due to Value-Based Purchasing': 'hvbp_dollars',
        'FY2026 Medicare Readmission Rate Penalty': 'hrrp_pct',
        'Est. FY2026 Revenue Loss Due to Readmission Penalty': 'hrrp_dollars'
    }

    df = df.rename(columns=column_mapping)

    # Ensure facility_id is 6-digit string
    df['facility_id'] = df['facility_id'].astype(str).str.zfill(6)

    # Helper function to clean accounting format numbers
    def clean_accounting_format(series):
        """Convert accounting format to standard numbers."""
        cleaned = series.astype(str)
        # Handle parentheses (negative numbers)
        cleaned = cleaned.str.replace(r'^\((.+)\)$', r'-\1', regex=True)
        # Remove $, commas, and spaces
        cleaned = cleaned.str.replace(r'[\$,\s]', '', regex=True)
        # Convert to float
        return cleaned.astype(float)

    # Clean revenue
    df['net_medicare_revenue'] = clean_accounting_format(df['net_medicare_revenue'])

    # Clean percentages
    for col in ['hvbp_pct', 'hrrp_pct', 'hacrp_pct']:
        df[col] = df[col].astype(str).str.replace(r'^\((.+)\)$', r'-\1', regex=True)
        df[col] = df[col].str.replace(r'%', '', regex=True).astype(float) / 100

    # Clean dollar amounts
    for col in ['hvbp_dollars', 'hrrp_dollars', 'hacrp_dollars']:
        df[col] = clean_accounting_format(df[col])

    # Add calculated columns that financial_calculator expects
    # Calculate opportunities (maximum potential if eliminated penalties)
    df['hvbp_opportunity'] = df.apply(
        lambda row: abs(row['hvbp_dollars']) if row['hvbp_dollars'] < 0 else 0,
        axis=1
    )
    df['hrrp_opportunity'] = df.apply(
        lambda row: abs(row['hrrp_dollars']) if row['hrrp_dollars'] < 0 else 0,
        axis=1
    )
    df['hacrp_opportunity'] = df.apply(
        lambda row: abs(row['hacrp_dollars']) if row['hacrp_dollars'] < 0 else 0,
        axis=1
    )
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

# ==============================================================================
# MANUAL ENTRY TO DATAFRAME
# ==============================================================================

def manual_entry_to_dataframe(
    hospital_name, facility_id, revenue,
    hacrp_pct, hacrp_dollars,
    hvbp_pct, hvbp_dollars,
    hrrp_pct, hrrp_dollars
):
    """Convert manual entry to DataFrame matching CSV format."""
    return pd.DataFrame([{
        'hospital_name': hospital_name,
        'facility_id': str(facility_id).zfill(6),
        'net_medicare_revenue': revenue,
        'hacrp_pct': hacrp_pct / 100,  # Convert to decimal
        'hacrp_dollars': hacrp_dollars,
        'hvbp_pct': hvbp_pct / 100,
        'hvbp_dollars': hvbp_dollars,
        'hrrp_pct': hrrp_pct / 100,
        'hrrp_dollars': hrrp_dollars
    }])

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

if 'financial_data' not in st.session_state:
    st.session_state.financial_data = None
if 'star_results' not in st.session_state:
    st.session_state.star_results = None
if 'assessment_generated' not in st.session_state:
    st.session_state.assessment_generated = False

# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🏥 Integrated Quality & Financial Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive C-Suite Report Combining Star Ratings, HVBP, HRRP, and HACRP</div>', unsafe_allow_html=True)

    # Introduction
    st.markdown("""
    <div class="info-box">
    <b>What This Tool Does:</b><br>
    1. You provide financial data (CSV file or manual entry)<br>
    2. We automatically fetch CMS public data for your hospitals<br>
    3. We analyze cross-program impacts and prioritize by ROI<br>
    4. You get a comprehensive executive report in one click
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # INPUT SECTION
    # ==============================================================================

    st.markdown('<div class="section-header">Step 1: Provide Hospital Financial Data</div>', unsafe_allow_html=True)

    input_method = st.radio(
        "Choose input method:",
        ["📤 Upload CSV File (Multiple Hospitals)", "✍️ Manual Entry (Single Hospital)"],
        horizontal=True
    )

    financial_data = None

    if input_method == "📤 Upload CSV File (Multiple Hospitals)":
        st.markdown("### Upload CSV File")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=['csv'],
                help="CSV must contain required columns"
            )

        with col2:
            st.markdown("**Required Columns:**")
            st.markdown("""
            - Hospital Name
            - facility_id
            - Net Medicare Revenue
            - FY2025 Medicare Hospital-Acquired Condition Adjustment
            - Est. FY2025 Revenue Adjustment Due to Hospital-Acquired Condition Penalty
            - FY2026 Medicare Value-Based Purchasing Adjustment
            - Est. FY2026 Revenue Adjustment Due to Value-Based Purchasing
            - FY2026 Medicare Readmission Rate Penalty
            - Est. FY2026 Revenue Loss Due to Readmission Penalty
            """)

        if uploaded_file:
            try:
                financial_data = parse_financial_csv_new_format(uploaded_file)
                financial_data = calculate_total_quality_impact(financial_data)
                st.success(f"✅ Loaded financial data for {len(financial_data)} hospitals")

                # Show preview
                with st.expander("📋 Preview Loaded Data"):
                    st.dataframe(financial_data[['hospital_name', 'facility_id', 'net_medicare_revenue',
                                                 'hvbp_dollars', 'hrrp_dollars', 'hacrp_dollars']],
                                hide_index=True)

            except Exception as e:
                st.error(f"❌ Error loading CSV: {e}")
                st.info("Please check that your CSV has the correct column names.")

    else:  # Manual Entry
        st.markdown("### Manual Entry Form")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Hospital Information**")
            hospital_name = st.text_input("Hospital Name", placeholder="e.g., Community Medical Center")
            facility_id = st.text_input("Facility ID (CCN)", placeholder="e.g., 340127", max_chars=6)
            revenue = st.number_input("Net Medicare Revenue ($)", min_value=0.0, value=0.0, step=100000.0,
                                     help="Total Medicare revenue for the facility")

        with col2:
            st.markdown("**Quality Program Financial Impacts**")

            st.markdown("**HACRP (FY2025):**")
            hacrp_pct = st.number_input("Penalty %", value=0.0, step=0.1, format="%.2f", key="hacrp_pct")
            hacrp_dollars = st.number_input("Dollar Impact ($)", value=0.0, step=1000.0, key="hacrp_dollars",
                                           help="Negative for penalty, positive for bonus")

            st.markdown("**HVBP (FY2026):**")
            hvbp_pct = st.number_input("Adjustment %", value=0.0, step=0.1, format="%.2f", key="hvbp_pct")
            hvbp_dollars = st.number_input("Dollar Impact ($)", value=0.0, step=1000.0, key="hvbp_dollars")

            st.markdown("**HRRP (FY2026):**")
            hrrp_pct = st.number_input("Penalty %", value=0.0, step=0.1, format="%.2f", key="hrrp_pct")
            hrrp_dollars = st.number_input("Dollar Impact ($)", value=0.0, step=1000.0, key="hrrp_dollars")

        if st.button("💾 Load Manual Entry Data", type="primary", use_container_width=True):
            if not hospital_name or not facility_id or not revenue:
                st.error("❌ Please fill in all required fields (Hospital Name, Facility ID, Revenue)")
            else:
                try:
                    financial_data = manual_entry_to_dataframe(
                        hospital_name, facility_id, revenue,
                        hacrp_pct, hacrp_dollars,
                        hvbp_pct, hvbp_dollars,
                        hrrp_pct, hrrp_dollars
                    )
                    financial_data = calculate_total_quality_impact(financial_data)
                    st.success(f"✅ Loaded data for {hospital_name}")

                    # Show summary
                    with st.expander("📋 Data Summary"):
                        st.write(f"**Facility ID:** {facility_id}")
                        st.write(f"**Revenue:** ${revenue:,.0f}")
                        st.write(f"**Total Impact:** ${hacrp_dollars + hvbp_dollars + hrrp_dollars:,.0f}")

                except Exception as e:
                    st.error(f"❌ Error processing manual entry: {e}")

    # Store financial data in session state
    if financial_data is not None:
        st.session_state.financial_data = financial_data

    st.markdown("---")

    # ==============================================================================
    # GENERATE ASSESSMENT BUTTON
    # ==============================================================================

    if st.session_state.financial_data is not None:
        st.markdown('<div class="section-header">Step 2: Generate Comprehensive Assessment</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <b>What happens next:</b><br>
        1. We'll fetch CMS public data for each facility ID<br>
        2. Combine financial data with quality performance<br>
        3. Analyze cross-program impacts<br>
        4. Generate executive-ready PDF report
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Generate Assessment Report", type="primary", use_container_width=True):
            with st.spinner("🔄 Fetching CMS data and generating assessment..."):
                try:
                    # Load CMS data - use the same data source as the main app
                    # This is already loaded in the background
                    import os

                    # Try multiple possible locations for CMS data
                    possible_paths = [
                        '/Users/heilman/Desktop/All Data July 2025.sas7bdat',
                        'Hospital_General_Information.sas7bdat',
                        '/Users/heilman/Hospital_General_Information.sas7bdat',
                        '/Users/heilman/Library/CloudStorage/OneDrive-MedisolvInc/Documents - Professional Services/Resources/Research/Care Compare/2025/07/Hospital_General_Information.sas7bdat'
                    ]

                    cms_data_path = None
                    for path in possible_paths:
                        if os.path.exists(path):
                            cms_data_path = path
                            break

                    if cms_data_path is None:
                        st.error("❌ CMS data file not found. Please ensure Hospital_General_Information.sas7bdat is available.")
                        return

                    df_original, df_std, national_stats = load_data(cms_data_path)

                    # Get facility IDs from financial data
                    facility_ids = st.session_state.financial_data['facility_id'].tolist()

                    # Process star ratings
                    star_results, summary_df = process_hospitals(facility_ids, df_std, df_original)
                    st.session_state.star_results = star_results

                    # Create integrated assessment
                    if len(star_results) > 1:
                        # Multi-hospital assessment
                        system_assessment = create_system_integrated_assessment(
                            star_results,
                            st.session_state.financial_data
                        )
                        st.session_state.system_assessment = system_assessment
                    else:
                        # Single hospital assessment
                        hospital_assessment = create_integrated_hospital_assessment(
                            facility_ids[0],
                            star_results[0],
                            st.session_state.financial_data
                        )
                        st.session_state.hospital_assessment = hospital_assessment

                    st.session_state.assessment_generated = True
                    st.success("✅ Assessment generated successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error generating assessment: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # ==============================================================================
    # DISPLAY RESULTS
    # ==============================================================================

    if st.session_state.assessment_generated:
        st.markdown('<div class="section-header">Assessment Results</div>', unsafe_allow_html=True)

        if 'system_assessment' in st.session_state:
            # Multi-hospital results
            display_system_assessment(st.session_state.system_assessment)
        elif 'hospital_assessment' in st.session_state:
            # Single hospital results
            display_hospital_assessment(st.session_state.hospital_assessment)

        # PDF Download
        st.markdown("---")
        st.markdown('<div class="section-header">Download Executive Report</div>', unsafe_allow_html=True)

        if st.button("📊 Generate Executive Dashboard PDF", use_container_width=True, type="primary"):
            with st.spinner("Generating PDF..."):
                try:
                    if 'system_assessment' in st.session_state:
                        exec_summary = create_executive_summary_data(st.session_state.system_assessment)
                        pdf_bytes = generate_executive_dashboard_pdf(
                            exec_summary,
                            system_name="Health System",
                            output_path=None
                        )
                    else:
                        # Single hospital PDF
                        st.warning("Single hospital PDF generation coming soon!")
                        return

                    st.download_button(
                        label="📥 Download Executive Dashboard PDF",
                        data=pdf_bytes,
                        file_name=f"executive_dashboard_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                    st.success("✅ PDF generated successfully!")

                except Exception as e:
                    st.error(f"❌ Error generating PDF: {e}")

# ==============================================================================
# VISUALIZATION HELPER FUNCTIONS
# ==============================================================================

def create_financial_waterfall_chart(financial_data):
    """Create waterfall chart showing financial impact breakdown."""

    # Extract values with correct key names
    revenue = financial_data.get('net_medicare_revenue', 0)
    hvbp = financial_data.get('hvbp_current', {}).get('dollars', 0)
    hrrp = financial_data.get('hrrp_current', {}).get('dollars', 0)
    hacrp = financial_data.get('hacrp_current', {}).get('dollars', 0)
    total_impact = financial_data.get('total_current_impact', {}).get('dollars', 0)

    # Create waterfall data
    fig = go.Figure(go.Waterfall(
        name = "Financial Impact",
        orientation = "v",
        measure = ["absolute", "relative", "relative", "relative", "total"],
        x = ["Medicare<br>Revenue", "HVBP", "HRRP", "HACRP", "Net Impact"],
        textposition = "outside",
        text = [f"${revenue:,.0f}",
                f"${hvbp:+,.0f}",
                f"${hrrp:+,.0f}",
                f"${hacrp:+,.0f}",
                f"${revenue + total_impact:,.0f}"],
        y = [revenue, hvbp, hrrp, hacrp, 0],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        decreasing = {"marker":{"color":"#d62728"}},
        increasing = {"marker":{"color":"#2ca02c"}},
        totals = {"marker":{"color":"#1f77b4"}}
    ))

    fig.update_layout(
        title = "Financial Impact Breakdown",
        showlegend = False,
        height = 500,  # Increased from 400
        margin=dict(t=100, b=50)  # Add top margin so text isn't cut off
    )

    return fig

def create_measure_detail_table(star_result, financial_data):
    """Create detailed measure table with scores and financial impacts."""

    measure_details = star_result.get('measure_details', {})
    from measure_crosswalk import ALL_MEASURES

    table_data = []
    for measure_id, data in measure_details.items():
        measure_info = ALL_MEASURES.get(measure_id, {})

        table_data.append({
            'Measure': measure_info.get('name', measure_id),
            'Score (SD)': f"{data.get('standardized', 0):.2f}",
            'Original Value': f"{data.get('original', 0):.3f}",
            'Status': '🔴 Problem' if data.get('standardized', 0) < -0.5 else '🟢 OK',
            'Programs': ', '.join(measure_info.get('programs', []))
        })

    df = pd.DataFrame(table_data)
    df = df.sort_values('Score (SD)')  # Worst first

    return df

# ==============================================================================
# DISPLAY FUNCTIONS
# ==============================================================================

def display_system_assessment(system_assessment):
    """Display system-wide assessment results."""

    # Overview metrics
    st.subheader("📊 System Overview")

    col1, col2, col3, col4 = st.columns(4)

    overview = system_assessment['system_overview']
    financial = system_assessment['financial_overview']

    with col1:
        st.metric("Total Hospitals", overview['total_hospitals'])
    with col2:
        st.metric("Avg Star Rating", f"{overview['avg_star_rating']:.2f} ⭐")
    with col3:
        st.metric("Hospitals Below 3 Stars",
                 f"{overview['hospitals_below_3_stars']} ({overview['pct_below_3_stars']:.0f}%)")
    with col4:
        st.metric("Medicare Revenue", f"${financial['total_medicare_revenue']/1e6:.1f}M")

    st.markdown("---")

    # Financial Impact
    st.subheader("💰 Financial Impact Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Current Impact", f"${financial['total_current_impact']/1e6:.2f}M",
                 delta=f"{(financial['total_current_impact']/financial['total_medicare_revenue']*100):.2f}%",
                 delta_color="inverse")
    with col2:
        st.metric("At-Risk Revenue", f"${financial['total_at_risk']/1e6:.1f}M")
    with col3:
        st.metric("Opportunity", f"${financial['total_opportunity']/1e6:.1f}M")
    with col4:
        total_penalties = (financial['hvbp_penalties'] +
                          financial['hrrp_penalties'] +
                          financial['hacrp_penalties'])
        st.metric("Total Penalties", total_penalties)

    st.markdown("---")

    # Top Strategic Priorities
    st.subheader("🎯 Top 5 Strategic Priorities (ROI-Ranked)")

    priorities = system_assessment['strategic_priorities'][:5]

    if priorities:
        priority_data = []
        for idx, p in enumerate(priorities, 1):
            priority_data.append({
                'Rank': idx,
                'Measure': p['measure_name'],
                'Programs': p['programs_affected'],
                'Hospitals': f"{p['hospitals_affected']} ({p['pct_system_affected']:.0f}%)",
                'Performance': f"{p['avg_performance']:.2f} SD",
                'Financial Impact': f"${p['estimated_financial_impact']:,.0f}",
                'Severity': p['severity_level']
            })

        priority_df = pd.DataFrame(priority_data)

        # Color code by severity
        def color_severity(row):
            if row['Severity'] == 'CRITICAL':
                return ['background-color: #ffcccc'] * len(row)
            elif row['Severity'] == 'HIGH':
                return ['background-color: #ffddaa'] * len(row)
            else:
                return ['background-color: #ffffcc'] * len(row)

        styled = priority_df.style.apply(color_severity, axis=1)
        st.dataframe(styled, hide_index=True, use_container_width=True)

        # Highlight top priority
        top = priorities[0]
        st.error(f"""
        🚨 **TOP PRIORITY:** {top['measure_name']}

        - **Programs Affected:** {top['programs_affected']}
        - **Hospitals:** {top['hospitals_affected']} ({top['pct_system_affected']:.0f}% of system)
        - **Performance:** {top['avg_performance']:.2f} SD below national
        - **Financial Impact:** ${top['estimated_financial_impact']:,.0f}
        - **Severity:** {top['severity_level']}
        {'- **✓ Cascading Impact** (affects 3+ programs)' if top.get('is_cascading') else ''}
        """)

def display_hospital_assessment(hospital_assessment):
    """Display comprehensive single hospital assessment with all visualizations."""

    st.subheader(f"🏥 {hospital_assessment['hospital_name']}")

    rating = hospital_assessment['star_rating']
    financial = hospital_assessment.get('financial')

    # Header Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Star Rating", f"{rating['current_rating']} ⭐")
    with col2:
        st.metric("Summary Score", f"{rating['summary_score']:.3f}")
    with col3:
        st.metric("Measures Reported", rating['measures_reported'])
    with col4:
        if financial and financial['found']:
            st.metric("Current Impact", f"${financial['total_current_impact']['dollars']:,.0f}")

    # ========== GROUP SCORES VISUALIZATION ==========
    st.markdown("---")
    st.subheader("📊 Performance by Measure Group")

    try:
        # Create group scores chart from original visualizations
        star_result = st.session_state.star_results[0] if st.session_state.star_results else None
        if star_result and 'standardized_group_scores' in star_result:
            fig_groups = create_group_scores_chart(star_result)
            st.plotly_chart(fig_groups, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not display group scores chart: {e}")

    # ========== FINANCIAL BREAKDOWN WITH VISUALIZATION ==========
    if financial and financial['found']:
        st.markdown("---")
        st.subheader("💰 Financial Impact Breakdown")

        # Financial waterfall chart
        try:
            fig_waterfall = create_financial_waterfall_chart(financial)
            st.plotly_chart(fig_waterfall, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create waterfall chart: {e}")

        # Detailed breakdown table
        st.write("**How Current Impact is Calculated:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**HVBP (Value-Based Purchasing)**")
            st.write(f"- Status: {financial['hvbp_current']['status']}")
            st.write(f"- Percentage: {financial['hvbp_current']['percent']*100:+.2f}%")
            st.write(f"- Dollar Impact: ${financial['hvbp_current']['dollars']:,.0f}")
            st.write(f"- Opportunity: ${financial['opportunity']['hvbp']:,.0f}")

        with col2:
            st.write("**HRRP (Readmissions)**")
            st.write(f"- Status: {financial['hrrp_current']['status']}")
            st.write(f"- Percentage: {financial['hrrp_current']['percent']*100:+.2f}%")
            st.write(f"- Dollar Impact: ${financial['hrrp_current']['dollars']:,.0f}")
            st.write(f"- Opportunity: ${financial['opportunity']['hrrp']:,.0f}")

        with col3:
            st.write("**HACRP (Hospital-Acquired Conditions)**")
            st.write(f"- Status: {financial['hacrp_current']['status']}")
            st.write(f"- Percentage: {financial['hacrp_current']['percent']*100:+.2f}%")
            st.write(f"- Dollar Impact: ${financial['hacrp_current']['dollars']:,.0f}")
            st.write(f"- Opportunity: ${financial['opportunity']['hacrp']:,.0f}")

    # ========== DETAILED MEASURE TABLE ==========
    st.markdown("---")
    st.subheader("📋 Detailed Measure Performance")

    try:
        star_result = st.session_state.star_results[0] if st.session_state.star_results else None
        if star_result:
            measure_df = create_measure_detail_table(star_result, financial)
            st.dataframe(measure_df, use_container_width=True, height=400)
    except Exception as e:
        st.warning(f"Could not create measure detail table: {e}")

    # ========== STRATEGIC PRIORITIES ==========
    st.markdown("---")
    st.subheader("🎯 Top Strategic Priorities (ROI-Ranked)")

    priorities = hospital_assessment.get('strategic_priorities', [])[:5]
    if priorities:
        for idx, p in enumerate(priorities, 1):
            with st.expander(f"**{idx}. {p.get('name', p['measure_id'])}** - {', '.join(p['programs_affected'])}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Current Performance:** {p.get('score', 0):.2f} SD below national average")
                    st.write(f"**Programs Affected:** {', '.join(p['programs_affected'])}")
                    st.write(f"**Cross-Program Impact:** {'✓ YES' if p.get('is_cross_program') else '✗ No'}")
                with col2:
                    if p.get('financial_impact', 0) > 0:
                        st.write(f"**Estimated Financial Impact:** ${p['financial_impact']:,.0f}")
                    if p.get('roi_ratio', 0) > 0:
                        st.write(f"**ROI Potential:** {p['roi_ratio']:.1f}x return")
                    st.write(f"**Recommendation:** {p.get('roi_recommendation', 'Focus improvement efforts here')}")

    # ========== EXPORT BUTTONS ==========
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Generate Executive Dashboard PDF", use_container_width=True):
            try:
                exec_summary = create_executive_summary_data([star_result], financial)
                pdf_buffer = generate_executive_dashboard_pdf(exec_summary)
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"executive_dashboard_{hospital_assessment['facility_id']}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.success("✅ PDF generated successfully!")
            except Exception as e:
                st.error(f"❌ Error generating PDF: {e}")

    with col2:
        if st.button("📑 Export to Excel", use_container_width=True):
            try:
                excel_buffer = export_to_excel([star_result], exec_summary=None)
                st.download_button(
                    label="📥 Download Excel File",
                    data=excel_buffer,
                    file_name=f"hospital_analysis_{hospital_assessment['facility_id']}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                st.success("✅ Excel file generated successfully!")
            except Exception as e:
                st.error(f"❌ Error generating Excel: {e}")

if __name__ == "__main__":
    main()
