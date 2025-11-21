"""
PDF executive summary generation for CMS Star Rating Calculator
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from typing import Dict
import io
from datetime import datetime
import pandas as pd
from star_rating_calculator import MEASURE_GROUPS


def generate_executive_summary_pdf(result: Dict, filename: str = None, financial_data=None, integrated_assessment=None) -> io.BytesIO:
    """
    Generate integrated assessment PDF for a single hospital including financial and cross-program analysis.

    Args:
        result: Hospital result dictionary
        filename: Optional filename to save to disk
        financial_data: Optional DataFrame with financial impact data
        integrated_assessment: Optional integrated assessment dictionary with strategic priorities

    Returns:
        BytesIO buffer with PDF
    """
    # Create buffer
    buffer = io.BytesIO()

    # Create PDF document
    if filename:
        doc = SimpleDocTemplate(filename, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
    else:
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)

    # Container for elements
    elements = []

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#366092'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#366092'),
        spaceAfter=12,
        spaceBefore=12
    )
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#555555'),
        spaceAfter=8
    )

    # Page 1: Cover Page
    elements.extend(_create_cover_page(result, title_style, styles))
    elements.append(PageBreak())

    # Page 2: Overview
    elements.extend(_create_overview_page(result, heading_style, subheading_style, styles))
    elements.append(PageBreak())

    # Page 3: Financial Impact (if financial data available)
    if financial_data is not None and not financial_data.empty:
        elements.extend(_create_financial_impact_page(result, financial_data, heading_style, subheading_style, styles))
        elements.append(PageBreak())

    # Page 4: Cross-Program Strategic Priorities (if integrated assessment available)
    if integrated_assessment is not None:
        elements.extend(_create_strategic_priorities_page(result, integrated_assessment, heading_style, subheading_style, styles))
        elements.append(PageBreak())

    # Page 5: Group Performance
    elements.extend(_create_group_performance_page(result, heading_style, subheading_style, styles))
    elements.append(PageBreak())

    # Page 6: Detailed Findings
    elements.extend(_create_detailed_findings_page(result, heading_style, subheading_style, styles))
    elements.append(PageBreak())

    # Page 7: Recommendations
    elements.extend(_create_recommendations_page(result, heading_style, subheading_style, styles))

    # Build PDF
    doc.build(elements, onFirstPage=_add_page_number, onLaterPages=_add_page_number)

    if filename:
        print(f"✅ PDF report saved to {filename}")

    buffer.seek(0)
    return buffer


def _add_page_number(canvas, doc):
    """Add page number to footer."""
    canvas.saveState()
    canvas.setFont('Times-Roman', 9)
    page_num = canvas.getPageNumber()
    text = f"Page {page_num}"
    canvas.drawRightString(7.5*inch, 0.5*inch, text)
    canvas.restoreState()


def _create_cover_page(result: Dict, title_style, styles):
    """Create cover page elements."""
    elements = []

    # Title
    elements.append(Spacer(1, 1.5*inch))
    elements.append(Paragraph("CMS Hospital Star Rating", title_style))
    elements.append(Paragraph("Executive Summary", title_style))
    elements.append(Spacer(1, 0.5*inch))

    # Hospital info
    provider_style = ParagraphStyle(
        'Provider',
        parent=styles['Normal'],
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=20
    )
    elements.append(Paragraph(f"<b>Facility ID: {result['provider_id']}</b>", provider_style))

    # Star rating display
    if result['is_eligible']:
        stars = '★' * result['star_rating'] + '☆' * (5 - result['star_rating'])
        star_style = ParagraphStyle(
            'Stars',
            parent=styles['Normal'],
            fontSize=48,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#FFD700'),
            spaceAfter=20
        )
        elements.append(Paragraph(stars, star_style))

        rating_style = ParagraphStyle(
            'Rating',
            parent=styles['Normal'],
            fontSize=36,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#366092'),
            spaceAfter=40
        )
        elements.append(Paragraph(f"<b>{result['star_rating']} Stars</b>", rating_style))
    else:
        not_eligible_style = ParagraphStyle(
            'NotEligible',
            parent=styles['Normal'],
            fontSize=24,
            alignment=TA_CENTER,
            textColor=colors.red,
            spaceAfter=40
        )
        elements.append(Paragraph("<b>Not Eligible for Star Rating</b>", not_eligible_style))

    elements.append(Spacer(1, 1*inch))

    # Date
    date_style = ParagraphStyle(
        'Date',
        parent=styles['Normal'],
        fontSize=12,
        alignment=TA_CENTER
    )
    elements.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}", date_style))
    elements.append(Paragraph("Data Source: CMS Hospital Compare - July 2025", date_style))

    return elements


def _create_overview_page(result: Dict, heading_style, subheading_style, styles):
    """Create overview page elements."""
    elements = []

    elements.append(Paragraph("Overview", heading_style))
    elements.append(Spacer(1, 0.2*inch))

    if not result['is_eligible']:
        elements.append(Paragraph(f"<b>Eligibility Status:</b> Not Eligible", styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph(f"<b>Reason:</b> {result['eligibility_reason']}", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph(
            "This hospital does not meet the minimum requirements for a CMS star rating. "
            "To be eligible, a hospital must have at least 3 measure groups with 3 or more measures, "
            "and must report either Mortality or Safety of Care measures.",
            styles['Normal']
        ))
        return elements

    # Key metrics table
    data = [
        ['Metric', 'Value'],
        ['Star Rating', f"{result['star_rating']} out of 5"],
        ['Summary Score', f"{result['summary_score']:.4f}"],
        ['Peer Group', f"{result['peer_group']} measure groups"],
        ['Total Measures Reported', f"{len(result['measure_details'])} out of 46"],
        ['Eligible Measure Groups', f"{len([g for g, c in result['measures_per_group'].items() if c >= 3])} out of 5"]
    ]

    table = Table(data, colWidths=[3*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#366092')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.3*inch))

    # Interpretation
    elements.append(Paragraph("What This Means", subheading_style))

    interpretation = _get_star_interpretation(result['star_rating'])
    elements.append(Paragraph(interpretation, styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))

    # Summary score explanation
    if result['summary_score'] > 0:
        score_text = f"With a summary score of {result['summary_score']:.4f}, this hospital performs <b>above the national average</b>."
    elif result['summary_score'] < 0:
        score_text = f"With a summary score of {result['summary_score']:.4f}, this hospital performs <b>below the national average</b>."
    else:
        score_text = f"With a summary score of {result['summary_score']:.4f}, this hospital performs at the national average."

    elements.append(Paragraph(score_text, styles['Normal']))

    return elements


def _create_group_performance_page(result: Dict, heading_style, subheading_style, styles):
    """Create group performance page elements."""
    elements = []

    elements.append(Paragraph("Performance by Measure Group", heading_style))
    elements.append(Spacer(1, 0.2*inch))

    if not result['is_eligible']:
        elements.append(Paragraph("Not applicable - hospital not eligible for star rating.", styles['Normal']))
        return elements

    # Group scores table
    data = [['Measure Group', 'Score', 'Measures', 'Weight', 'Performance']]

    for group_name in MEASURE_GROUPS.keys():
        score = result['standardized_group_scores'].get(group_name)
        measures = result['measures_per_group'].get(group_name, 0)
        weight = result['adjusted_weights'].get(group_name)

        if score is not None:
            performance = 'Strong' if score > 0.5 else 'Weak' if score < -0.5 else 'Average'
            data.append([
                group_name,
                f"{score:.3f}",
                str(measures),
                f"{weight:.2%}" if weight else 'N/A',
                performance
            ])
        else:
            data.append([
                group_name,
                'N/A',
                str(measures),
                'N/A',
                'No Data'
            ])

    table = Table(data, colWidths=[2.2*inch, 1*inch, 0.8*inch, 0.8*inch, 1.2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#366092')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
    ]))

    # Add color coding for performance
    for i, row in enumerate(data[1:], start=1):
        if row[4] == 'Strong':
            table.setStyle(TableStyle([('BACKGROUND', (4, i), (4, i), colors.lightgreen)]))
        elif row[4] == 'Weak':
            table.setStyle(TableStyle([('BACKGROUND', (4, i), (4, i), colors.lightcoral)]))
        elif row[4] == 'Average':
            table.setStyle(TableStyle([('BACKGROUND', (4, i), (4, i), colors.lightyellow)]))

    elements.append(table)
    elements.append(Spacer(1, 0.3*inch))

    # Interpretation
    elements.append(Paragraph("Interpretation", subheading_style))
    elements.append(Paragraph(
        "Scores represent standard deviations from the national average. "
        "A score of 0 means the hospital performs at the national average. "
        "Positive scores indicate better-than-average performance, while negative scores indicate below-average performance.",
        styles['Normal']
    ))

    return elements


def _create_detailed_findings_page(result: Dict, heading_style, subheading_style, styles):
    """Create detailed findings page elements."""
    elements = []

    elements.append(Paragraph("Detailed Findings", heading_style))
    elements.append(Spacer(1, 0.2*inch))

    if not result['is_eligible']:
        elements.append(Paragraph("Not applicable - hospital not eligible for star rating.", styles['Normal']))
        return elements

    # Strength areas
    if result['strength_areas']:
        elements.append(Paragraph("Areas of Strength", subheading_style))
        strength_list = "<br/>".join([f"• {area}" for area in result['strength_areas']])
        elements.append(Paragraph(strength_list, styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))

        # Details
        for area in result['strength_areas']:
            score = result['standardized_group_scores'][area]
            elements.append(Paragraph(
                f"<b>{area}</b>: Score of {score:.3f} indicates performance is {abs(score):.2f} standard deviations above the national average.",
                styles['Normal']
            ))
            elements.append(Spacer(1, 0.1*inch))
    else:
        elements.append(Paragraph("Areas of Strength", subheading_style))
        elements.append(Paragraph("No measure groups scoring >0.5 standard deviations above average.", styles['Normal']))

    elements.append(Spacer(1, 0.3*inch))

    # Problem areas
    if result['problem_areas']:
        elements.append(Paragraph("Areas for Improvement", subheading_style))
        problem_list = "<br/>".join([f"• {area}" for area in result['problem_areas']])
        elements.append(Paragraph(problem_list, styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))

        # Details
        for area in result['problem_areas']:
            score = result['standardized_group_scores'][area]
            elements.append(Paragraph(
                f"<b>{area}</b>: Score of {score:.3f} indicates performance is {abs(score):.2f} standard deviations below the national average. "
                f"This represents a significant opportunity for improvement.",
                styles['Normal']
            ))
            elements.append(Spacer(1, 0.1*inch))
    else:
        elements.append(Paragraph("Areas for Improvement", subheading_style))
        elements.append(Paragraph("No measure groups scoring <-0.5 standard deviations below average.", styles['Normal']))

    return elements


def _create_recommendations_page(result: Dict, heading_style, subheading_style, styles):
    """Create recommendations page elements."""
    elements = []

    elements.append(Paragraph("Recommendations", heading_style))
    elements.append(Spacer(1, 0.2*inch))

    if not result['is_eligible']:
        elements.append(Paragraph("Recommendations for Achieving Eligibility", subheading_style))
        elements.append(Paragraph(
            "To become eligible for a CMS star rating, focus on expanding reporting across all measure groups. "
            "Priority should be given to reporting Mortality and Safety of Care measures, as at least one of these groups is required for eligibility.",
            styles['Normal']
        ))
        return elements

    recommendations = _generate_recommendations(result)

    for i, rec in enumerate(recommendations, 1):
        elements.append(Paragraph(f"<b>Recommendation #{i}: {rec['title']}</b>", subheading_style))
        elements.append(Paragraph(rec['description'], styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))

    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("Potential Impact", subheading_style))
    elements.append(Paragraph(
        "Addressing these priority areas could improve the hospital's summary score and potentially increase the star rating. "
        f"The hospital is currently in the {result['peer_group']}-group peer comparison. "
        "Focus on the weakest performing groups first, as these offer the greatest opportunity for improvement.",
        styles['Normal']
    ))

    return elements


def _get_star_interpretation(stars: int) -> str:
    """Get interpretation text for star rating."""
    interpretations = {
        5: "This <b>5-star rating</b> indicates outstanding performance, significantly above the national average across quality measures. This places the hospital in the top tier of U.S. hospitals.",
        4: "This <b>4-star rating</b> indicates above-average performance across quality measures. The hospital performs better than most U.S. hospitals in the same peer group.",
        3: "This <b>3-star rating</b> indicates average performance, comparable to the national average. There are opportunities for improvement in several areas.",
        2: "This <b>2-star rating</b> indicates below-average performance across quality measures. Significant improvement is needed in multiple areas to reach the national average.",
        1: "This <b>1-star rating</b> indicates performance significantly below the national average. Immediate attention and quality improvement initiatives are strongly recommended."
    }
    return interpretations.get(stars, "")


def _generate_recommendations(result: Dict) -> list:
    """Generate actionable recommendations based on results."""
    recommendations = []

    # Recommendation 1: Focus on weakest group
    if result['problem_areas']:
        weakest_group = min(result['problem_areas'],
                           key=lambda g: result['standardized_group_scores'][g])
        score = result['standardized_group_scores'][weakest_group]

        recommendations.append({
            'title': f"Priority Focus on {weakest_group}",
            'description': f"With a score of {score:.3f}, {weakest_group} is the weakest performing area. "
                          f"Implement targeted quality improvement initiatives in this group. "
                          f"Review individual measure performance to identify specific opportunities."
        })

    # Recommendation 2: Expand measure reporting
    missing_groups = [g for g, c in result['measures_per_group'].items() if c < 3]
    if missing_groups:
        recommendations.append({
            'title': "Expand Measure Reporting",
            'description': f"Consider expanding reporting in: {', '.join(missing_groups)}. "
                          f"Additional measures could change your peer group and provide a more complete picture of quality."
        })

    # Recommendation 3: Leverage strengths
    if result['strength_areas']:
        recommendations.append({
            'title': "Share Best Practices from Strong Areas",
            'description': f"Your hospital performs well in {', '.join(result['strength_areas'])}. "
                          f"Document and share best practices from these areas with teams working on weaker measure groups."
        })

    # Recommendation 4: Peer comparison
    recommendations.append({
        'title': "Benchmark Against Top Performers",
        'description': f"Compare your hospital's performance with 5-star hospitals in the {result['peer_group']}-group peer category. "
                      f"Identify specific practices and processes that differentiate top performers."
    })

    # Recommendation 5: Focus on high-weight groups
    if result['star_rating'] and result['star_rating'] < 4:
        high_weight_problems = [g for g in result['problem_areas']
                               if MEASURE_GROUPS[g]['weight'] >= 0.22]
        if high_weight_problems:
            recommendations.append({
                'title': "Address High-Weight Measure Groups",
                'description': f"Focus particularly on {', '.join(high_weight_problems)}, "
                              f"as these groups carry the highest weights (22%) in the star rating calculation. "
                              f"Improvements here will have the greatest impact on your overall score."
            })

    return recommendations[:5]  # Return top 5 recommendations


def generate_system_executive_summary_pdf(results_list: list, exec_summary: Dict, filename: str = None, financial_data=None, integrated_assessment=None) -> io.BytesIO:
    """
    Generate integrated executive PDF summary for multiple hospitals with financial and cross-program analysis.

    Args:
        results_list: List of hospital results
        exec_summary: Executive summary dictionary from create_executive_summary()
        filename: Optional filename to save to disk
        financial_data: Optional DataFrame with financial impact data
        integrated_assessment: Optional system-wide integrated assessment

    Returns:
        BytesIO buffer with PDF
    """
    # Create buffer
    buffer = io.BytesIO()

    # Create PDF document
    if filename:
        doc = SimpleDocTemplate(filename, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
    else:
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)

    # Container for elements
    elements = []

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#366092'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#366092'),
        spaceAfter=12,
        spaceBefore=12
    )
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#555555'),
        spaceAfter=8
    )

    # PAGE 1: COVER PAGE
    elements.append(Paragraph("Integrated Quality & Financial Assessment", title_style))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("Star Ratings, HVBP, HRRP, HACRP Analysis", subheading_style))
    elements.append(Spacer(1, 0.5*inch))

    overview = exec_summary['System_Overview'].iloc[0]
    elements.append(Paragraph(f"Analysis of {overview['Total_Hospitals']} Hospitals", heading_style))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    elements.append(Spacer(1, 0.5*inch))

    # Key metrics summary
    metrics_text = f"""
    <b>Star Rating System Performance Summary:</b><br/>
    <br/>
    <b>Total Hospitals Analyzed:</b> {overview['Total_Hospitals']}<br/>
    <b>Hospitals with Valid Ratings:</b> {overview['Hospitals_with_Valid_Ratings']}<br/>
    <b>Average Star Rating:</b> {overview['Average_Star_Rating']:.2f} stars<br/>
    <br/>
    <b>Star Rating Distribution:</b><br/>
    5 Stars: {overview['5_Star_Hospitals']} hospitals ({overview['5_Star_Hospitals']/overview['Total_Hospitals']*100:.0f}%)<br/>
    4 Stars: {overview['4_Star_Hospitals']} hospitals ({overview['4_Star_Hospitals']/overview['Total_Hospitals']*100:.0f}%)<br/>
    3 Stars: {overview['3_Star_Hospitals']} hospitals ({overview['3_Star_Hospitals']/overview['Total_Hospitals']*100:.0f}%)<br/>
    2 Stars: {overview['2_Star_Hospitals']} hospitals ({overview['2_Star_Hospitals']/overview['Total_Hospitals']*100:.0f}%)<br/>
    1 Star: {overview['1_Star_Hospitals']} hospitals ({overview['1_Star_Hospitals']/overview['Total_Hospitals']*100:.0f}%)
    """
    elements.append(Paragraph(metrics_text, styles['Normal']))

    # Add financial summary preview if available
    if financial_data is not None and not financial_data.empty:
        elements.append(Spacer(1, 0.3*inch))
        total_revenue = financial_data['net_medicare_revenue'].sum()
        total_impact = financial_data.get('hvbp_dollars', pd.Series([0])).sum() + \
                      financial_data.get('hrrp_dollars', pd.Series([0])).sum() + \
                      financial_data.get('hacrp_dollars', pd.Series([0])).sum()
        impact_pct = (total_impact / total_revenue * 100) if total_revenue > 0 else 0

        fin_preview_text = f"""
        <b>Financial System Performance Summary:</b><br/>
        <br/>
        <b>Total Medicare Revenue:</b> ${total_revenue/1e6:.1f}M<br/>
        <b>Current Quality Program Impact:</b> ${total_impact/1e6:+.1f}M ({impact_pct:+.1f}% of revenue)<br/>
        <b>At-Risk Revenue:</b> ${total_revenue * 0.06 / 1e6:.1f}M (6% maximum exposure across all programs)
        """
        elements.append(Paragraph(fin_preview_text, styles['Normal']))
    elements.append(PageBreak())

    # PAGE 2: KEY FINDINGS
    elements.append(Paragraph("Key Findings & Problem Areas", heading_style))
    elements.append(Spacer(1, 0.2*inch))

    # Top problem measures table (with weighted impact scores)
    if not exec_summary['Top_Problem_Measures'].empty:
        elements.append(Paragraph("Top Problem Measures (Weighted Impact Analysis)", subheading_style))
        elements.append(Spacer(1, 0.1*inch))

        problem_data = [['Measure Name', '% System', 'Impact Score', 'Severity']]
        for _, row in exec_summary['Top_Problem_Measures'].head(5).iterrows():
            problem_data.append([
                row['Measure_Name'],
                row['Pct_System'],
                f"{row['Impact_Score']:.1f}",
                row['Severity']
            ])

        problem_table = Table(problem_data, colWidths=[2.5*inch, 0.8*inch, 1*inch, 1*inch])
        problem_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#366092')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        elements.append(problem_table)
        elements.append(Spacer(1, 0.3*inch))

    # Group Performance table
    if not exec_summary['Group_Performance'].empty:
        elements.append(Paragraph("Measure Group Performance", subheading_style))
        elements.append(Spacer(1, 0.1*inch))

        group_data = [['Group', 'System Avg', '% Below Nat\'l', 'Priority']]
        for _, row in exec_summary['Group_Performance'].iterrows():
            group_data.append([
                row['Measure_Group'],
                row['System_Average'],
                row['Pct_Below_National'],
                row['Priority']
            ])

        group_table = Table(group_data, colWidths=[2*inch, 1*inch, 1.2*inch, 1*inch])
        group_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#366092')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        elements.append(group_table)
    elements.append(PageBreak())

    # PAGE 3: RECOMMENDATIONS
    elements.append(Paragraph("Recommended Actions", heading_style))
    elements.append(Spacer(1, 0.2*inch))

    if not exec_summary['Recommendations'].empty:
        for idx, (_, row) in enumerate(exec_summary['Recommendations'].iterrows(), 1):
            priority_text = f"<b>{row['Priority']} PRIORITY</b>: {row['Focus_Area']} (Impact Score: {row['Impact_Score']:.1f})"
            elements.append(Paragraph(priority_text, subheading_style))
            elements.append(Spacer(1, 0.1*inch))

            issue = Paragraph(f"<b>Issue:</b> {row['Issue']}", styles['Normal'])
            elements.append(issue)
            elements.append(Spacer(1, 0.05*inch))

            rec = Paragraph(f"<b>Recommendation:</b> {row['Recommendation']}", styles['Normal'])
            elements.append(rec)
            elements.append(Spacer(1, 0.2*inch))
    else:
        elements.append(Paragraph("No critical recommendations at this time. Continue monitoring performance metrics.", styles['Normal']))

    elements.append(PageBreak())

    # PAGE 4: IMPROVEMENT OPPORTUNITIES
    if not exec_summary['Improvement_Opportunities'].empty:
        elements.append(Paragraph("Star Rating Improvement Opportunities", heading_style))
        elements.append(Spacer(1, 0.2*inch))

        elements.append(Paragraph(
            "The following hospitals are closest to achieving the next star rating level:",
            styles['Normal']
        ))
        elements.append(Spacer(1, 0.1*inch))

        opp_data = [['Hospital', 'Current', 'Gap', 'Focus Area', 'Feasibility']]
        for _, row in exec_summary['Improvement_Opportunities'].head(10).iterrows():
            opp_data.append([
                row['Facility_ID'],
                f"{row['Current_Rating']} ⭐",
                f"{row['Gap_to_Next_Star']:.3f}",
                row['Primary_Focus_Area'],
                row['Feasibility']
            ])

        opp_table = Table(opp_data, colWidths=[1.2*inch, 0.8*inch, 0.6*inch, 2*inch, 0.8*inch])
        opp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#366092')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        elements.append(opp_table)

    # PAGE 5: FINANCIAL IMPACT SUMMARY (if available)
    if financial_data is not None and not financial_data.empty:
        elements.append(PageBreak())
        elements.append(Paragraph("System Financial Impact Summary", heading_style))
        elements.append(Spacer(1, 0.2*inch))

        # Calculate totals
        total_revenue = financial_data['net_medicare_revenue'].sum()
        total_hvbp = financial_data.get('hvbp_dollars', pd.Series([0])).sum()
        total_hrrp = financial_data.get('hrrp_dollars', pd.Series([0])).sum()
        total_hacrp = financial_data.get('hacrp_dollars', pd.Series([0])).sum()
        total_impact = total_hvbp + total_hrrp + total_hacrp
        impact_pct = (total_impact / total_revenue * 100) if total_revenue > 0 else 0

        # Calculate at-risk and opportunity
        at_risk_revenue = total_revenue * 0.06  # 6% max: HVBP 2% + HRRP 3% + HACRP 1%
        total_opportunity = financial_data.get('total_opportunity', pd.Series([0])).sum()

        # Key metrics summary
        metrics_text = f"""<b>System-Wide Quality Performance Financial Impact</b><br/><br/>
        <b>Total Medicare Revenue:</b> ${total_revenue/1e6:.1f}M<br/>
        <b>Current Impact:</b> ${total_impact/1e6:+.1f}M ({impact_pct:+.1f}% of revenue)<br/>
        <b>At-Risk Revenue:</b> ${at_risk_revenue/1e6:.1f}M (6% maximum penalty exposure)<br/>
        <b>Improvement Opportunity:</b> ${total_opportunity/1e6:.1f}M (potential upside from QI initiatives)
        """
        elements.append(Paragraph(metrics_text, styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))

        financial_summary_data = [
            ['Program', 'Total Impact', '% of Revenue'],
            ['Total Medicare Revenue', f"${total_revenue:,.0f}", '100%'],
            ['HVBP (FY 2026)', f"${total_hvbp:,.0f}", f"{(total_hvbp/total_revenue*100):+.2f}%"],
            ['HRRP (FY 2026)', f"${total_hrrp:,.0f}", f"{(total_hrrp/total_revenue*100):.2f}%"],
            ['HACRP (FY 2025)', f"${total_hacrp:,.0f}", f"{(total_hacrp/total_revenue*100):.2f}%"],
            ['Total Current Impact', f"${total_impact:,.0f}", f"{impact_pct:+.2f}%"]
        ]

        fin_summary_table = Table(financial_summary_data, colWidths=[3*inch, 2*inch, 1.5*inch])
        fin_summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#366092')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#FFE6E6') if total_impact < 0 else colors.HexColor('#E6FFE6')),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        elements.append(fin_summary_table)

    # PAGE 6: CROSS-PROGRAM STRATEGIC PRIORITIES (if available)
    if integrated_assessment is not None and 'strategic_priorities' in integrated_assessment:
        elements.append(PageBreak())
        elements.append(Paragraph("Cross-Program Strategic Priorities", heading_style))
        elements.append(Spacer(1, 0.2*inch))

        # Cross-Program Impact Metrics
        integrated_metrics = integrated_assessment.get('integrated_metrics', {})
        metrics_text = f"""<b>Cross-Program Impact Analysis</b><br/><br/>
        <b>Total Problem Measures:</b> {integrated_metrics.get('total_unique_problem_measures', 'N/A')}<br/>
        <b>Cross-Program Measures:</b> {integrated_metrics.get('cross_program_measures', 'N/A')} (affecting 2+ programs)<br/>
        <b>Cascading Impact Measures:</b> {integrated_metrics.get('cascading_impact_measures', 'N/A')} (affecting 3+ programs - highest ROI)
        """
        elements.append(Paragraph(metrics_text, styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))

        intro_text = """<b>System-Wide ROI-Ranked Improvement Opportunities</b><br/><br/>
        These measures affect multiple CMS programs simultaneously across your system, providing the highest
        return on investment for quality improvement efforts.<br/><br/>
        <b>Priority Score Calculation:</b> Programs Affected × Hospitals Affected × |Avg Performance (SD)|<br/>
        Higher scores indicate more urgent improvement opportunities (affecting more programs, more hospitals, with worse performance)."""
        elements.append(Paragraph(intro_text, styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))

        priorities = integrated_assessment['strategic_priorities'][:10]
        if priorities:
            # Top Priority Callout
            if len(priorities) > 0:
                top = priorities[0]
                callout_style = ParagraphStyle(
                    'CalloutStyle',
                    parent=styles['Normal'],
                    fontSize=11,
                    textColor=colors.HexColor('#721c24'),
                    backColor=colors.HexColor('#f8d7da'),
                    borderPadding=10,
                    spaceAfter=12,
                    leftIndent=10,
                    rightIndent=10
                )
                callout_text = f"""<b>TOP PRIORITY:</b> {top.get('measure_name', 'Unknown')}<br/>
                Programs Affected: {top.get('num_programs', 0)} |
                Hospitals Affected: {top.get('hospitals_affected', 0)} ({top.get('pct_system_affected', 0):.0f}%)<br/>
                Avg Performance: {top.get('avg_performance', 0):.2f} SD |
                Priority Score: {top.get('priority_score', 0):.1f}"""
                elements.append(Paragraph(callout_text, callout_style))
                elements.append(Spacer(1, 0.2*inch))
            priority_data = [['Rank', 'Measure', 'Programs Affected', 'Hospitals Affected', 'Avg Performance', 'Priority Score']]
            for rank, priority in enumerate(priorities, 1):
                priority_data.append([
                    str(rank),
                    priority.get('measure_name', 'Unknown')[:40],
                    f"{priority.get('num_programs', 0)} programs",
                    f"{priority.get('hospitals_affected', 0)} ({priority.get('pct_system_affected', 0):.0f}%)",
                    f"{priority.get('avg_performance', 0):.2f} SD",
                    f"{priority.get('priority_score', 0):.1f}"
                ])

            priority_table = Table(priority_data, colWidths=[0.4*inch, 2.2*inch, 1*inch, 1*inch, 0.9*inch, 0.7*inch])
            priority_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#366092')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                ('ALIGN', (4, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            elements.append(priority_table)

    # Build PDF
    doc.build(elements, onFirstPage=_add_page_number, onLaterPages=_add_page_number)

    if filename:
        print(f"✅ System executive summary PDF saved to {filename}")

    buffer.seek(0)
    return buffer


def _create_financial_impact_page(result, financial_data, heading_style, subheading_style, styles):
    """Create financial impact page for PDF"""
    elements = []

    # Get financial data for this hospital
    facility_id = str(result['provider_id'])
    fin_row = financial_data[financial_data['facility_id'].astype(str) == facility_id]

    if fin_row.empty:
        elements.append(Paragraph("Financial Impact Analysis", heading_style))
        elements.append(Paragraph("Financial data not available for this facility.", styles['Normal']))
        return elements

    fin = fin_row.iloc[0]

    # Title
    elements.append(Paragraph("Financial Impact Analysis", heading_style))
    elements.append(Spacer(1, 0.2*inch))

    # Summary metrics
    elements.append(Paragraph("CMS Value-Based Program Performance", subheading_style))

    summary_data = [
        ['Program', 'Measurement Period', 'Current Impact', '% of Revenue'],
        ['Medicare Revenue', 'Annual', f"${fin['net_medicare_revenue']:,.0f}", '100%'],
        ['HVBP (FY 2026)', 'Oct 2023 - Sep 2024',
         f"${fin.get('hvbp_dollars', 0):,.0f}",
         f"{fin.get('hvbp_adjustment_pct', 0):.2f}%"],
        ['HRRP (FY 2026)', 'July 2021 - June 2024',
         f"${fin.get('hrrp_dollars', 0):,.0f}",
         f"{fin.get('hrrp_penalty_pct', 0):.2f}%"],
        ['HACRP (FY 2025)', 'Jan 2022 - Dec 2023',
         f"${fin.get('hacrp_dollars', 0):,.0f}",
         f"{fin.get('hacrp_adjustment_pct', 0):.2f}%"]
    ]

    total_impact = fin.get('total_current_impact_dollars', 0)
    impact_pct = (total_impact / fin['net_medicare_revenue'] * 100) if fin['net_medicare_revenue'] > 0 else 0

    summary_data.append([
        'Total Current Impact', '',
        f"${total_impact:,.0f}",
        f"{impact_pct:+.2f}%"
    ])

    summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 1.3*inch, 1*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#366092')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#FFE6E6') if total_impact < 0 else colors.HexColor('#E6FFE6')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 0.3*inch))

    # Program details
    elements.append(Paragraph("Program-by-Program Breakdown", subheading_style))
    elements.append(Spacer(1, 0.1*inch))

    # HVBP
    elements.append(Paragraph("<b>Hospital Value-Based Purchasing (HVBP)</b>", styles['Normal']))
    hvbp_text = f"FY 2026 adjustment: ${fin.get('hvbp_dollars', 0):,.0f} ({fin.get('hvbp_adjustment_pct', 0):+.2f}%). "
    hvbp_text += "HVBP rewards or penalizes up to ±2% of Medicare payments based on quality performance across clinical outcomes, patient experience, and safety domains."
    elements.append(Paragraph(hvbp_text, styles['Normal']))
    elements.append(Spacer(1, 0.1*inch))

    # HRRP
    elements.append(Paragraph("<b>Hospital Readmissions Reduction Program (HRRP)</b>", styles['Normal']))
    hrrp_text = f"FY 2026 penalty: ${fin.get('hrrp_dollars', 0):,.0f} ({fin.get('hrrp_penalty_pct', 0):.2f}%). "
    hrrp_text += "HRRP applies penalties up to -3% for hospitals with excess readmissions in six clinical conditions."
    elements.append(Paragraph(hrrp_text, styles['Normal']))
    elements.append(Spacer(1, 0.1*inch))

    # HACRP
    elements.append(Paragraph("<b>Hospital-Acquired Condition Reduction Program (HACRP)</b>", styles['Normal']))
    hacrp_text = f"FY 2025 penalty: ${fin.get('hacrp_dollars', 0):,.0f} ({fin.get('hacrp_adjustment_pct', 0):.2f}%). "
    hacrp_text += "HACRP applies a -1% penalty to the worst-performing 25% of hospitals on hospital-acquired infections and patient safety indicators."
    elements.append(Paragraph(hacrp_text, styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))

    # Opportunity
    total_opp = fin.get('total_opportunity', 0)
    if total_opp > 0:
        elements.append(Paragraph("Improvement Opportunity", subheading_style))
        opp_text = f"<b>Potential revenue recovery: ${total_opp:,.0f}</b><br/>"
        opp_text += "This represents the potential financial gain from eliminating all current penalties and maximizing bonuses through quality improvement initiatives."
        elements.append(Paragraph(opp_text, styles['Normal']))

    return elements


def _create_strategic_priorities_page(result, integrated_assessment, heading_style, subheading_style, styles):
    """Create cross-program strategic priorities page for PDF"""
    from strategic_priorities import calculate_strategic_priorities_corrected
    from integrated_assessment import get_hospital_integrated_assessment

    elements = []

    # Title
    elements.append(Paragraph("Cross-Program Strategic Priorities", heading_style))
    elements.append(Spacer(1, 0.2*inch))

    # Introduction
    intro_text = """<b>ROI-Ranked Quality Improvement Priorities</b><br/><br/>
    The following measures have been identified as strategic priorities because they impact MULTIPLE CMS programs simultaneously.
    Improving these measures provides the highest return on investment by affecting both your star rating AND financial programs
    (HVBP, HRRP, HACRP)."""
    elements.append(Paragraph(intro_text, styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))

    # Get strategic priorities for this hospital
    try:
        hospital_assessment = get_hospital_integrated_assessment(
            result,
            integrated_assessment.get('financial_data') if integrated_assessment else None
        )

        if 'strategic_priorities' in hospital_assessment and hospital_assessment['strategic_priorities']:
            priorities = hospital_assessment['strategic_priorities'][:10]  # Top 10

            # Create table
            priority_data = [['Rank', 'Measure', 'Programs', 'Current Score', 'Star Impact', 'Financial Impact']]

            for rank, priority in enumerate(priorities, 1):
                severity_symbol = ''
                if priority.get('severity') == 'CRITICAL':
                    severity_symbol = '[CRITICAL] '
                elif priority.get('severity') == 'HIGH':
                    severity_symbol = '[HIGH] '

                priority_data.append([
                    str(rank),
                    severity_symbol + priority.get('measure_name', 'Unknown')[:40],
                    f"{priority.get('program_count', 0)} pgms",
                    f"{priority.get('current_score', 0):.2f} SD",
                    f"{priority.get('star_impact_points', 0):+.3f} pts",
                    f"${abs(priority.get('financial_impact', 0)):,.0f}"
                ])

            priority_table = Table(priority_data, colWidths=[0.4*inch, 2.5*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.9*inch])
            priority_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#366092')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                ('ALIGN', (3, 0), (-1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
            ]))

            elements.append(priority_table)
            elements.append(Spacer(1, 0.2*inch))

            # Explain top priority
            if len(priorities) > 0:
                top = priorities[0]
                elements.append(Paragraph("Top Priority Detail", subheading_style))
                detail_text = f"""<b>{top.get('measure_name', 'Unknown')}</b><br/><br/>
                <b>Programs Affected:</b> {', '.join(top.get('programs', []))}<br/>
                <b>Current Performance:</b> {top.get('current_score', 0):.2f} standard deviations from national average<br/>
                <b>Star Rating Impact:</b> {top.get('star_impact_points', 0):+.3f} points<br/>
                <b>Estimated Financial Impact:</b> ${abs(top.get('financial_impact', 0)):,.0f}<br/><br/>
                <b>Why This Matters:</b> This measure affects {top.get('program_count', 0)} CMS programs, making it a high-ROI target
                for improvement. Addressing this measure will simultaneously improve your star rating and reduce financial penalties.
                """
                elements.append(Paragraph(detail_text, styles['Normal']))
        else:
            elements.append(Paragraph("No strategic priorities identified. Excellent performance across all programs!", styles['Normal']))

    except Exception as e:
        elements.append(Paragraph(f"Strategic priorities analysis not available. ({str(e)})", styles['Normal']))

    return elements
