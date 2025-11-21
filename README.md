# CMS Hospital Star Rating Calculator

An integrated quality and financial assessment tool for hospital systems, analyzing performance across CMS Star Ratings, HVBP, HRRP, and HACRP programs.

## Features

- **Star Rating Calculator**: Calculate CMS Hospital Star Ratings using July 2025 methodology
- **Financial Impact Analysis**: Assess HVBP, HRRP, and HACRP financial performance
- **Cross-Program Strategic Priorities**: Identify measures affecting multiple CMS programs
- **System-Wide Reporting**: Analyze performance across multi-hospital systems
- **Executive Dashboards**: Generate PDF and Excel reports

## Usage

1. Upload your hospital data file (Excel, CSV, or SAS format)
2. Upload Definitive Healthcare financial data (optional)
3. View integrated quality and financial assessment
4. Download comprehensive PDF or Excel reports

## Requirements

- Python 3.9+
- Streamlit
- Pandas, NumPy
- Plotly (for visualizations)
- ReportLab (for PDF generation)

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data Privacy

This tool processes CMS quality data. Do not upload PHI or sensitive patient information.

---

Built with Streamlit | CMS Methodology v4.1 (July 2025)
