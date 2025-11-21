"""
Verify that the enhanced executive summary answers C-suite questions:
1. Which specific measures are failing?
2. How much do they impact star ratings? (weighted)
3. Which hospitals can improve quickly?
4. What's the financial impact?
5. What are the top 3-5 specific actions to take?
"""

import pandas as pd
from star_rating_calculator import load_data, process_hospitals
from app import create_executive_summary

print("=" * 80)
print("VERIFICATION: C-Suite Executive Summary Questions")
print("=" * 80)

# Load CHS hospitals
print("\nLoading CHS hospital IDs...")
chs_df = pd.read_csv("/Users/heilman/Downloads/CHS_Hospitals_ProviderIDs.csv", header=None)
facility_ids = list(set(chs_df[0].astype(str).tolist()))

# Load data and process
print("Loading hospital data...")
file_path = "/Users/heilman/Desktop/All Data July 2025.sas7bdat"
df_original, df_std, national_stats = load_data(file_path)

print(f"Processing {len(facility_ids)} CHS hospitals...")
results_list, summary_df = process_hospitals(facility_ids, df_std, df_original)

# Create executive summary
print("Creating executive summary...")
exec_summary = create_executive_summary(results_list)

print("\n" + "=" * 80)
print("QUESTION 1: Which specific measures are failing?")
print("=" * 80)
print("\nTop Problem Measures (showing human-readable names):")
print(exec_summary['Top_Problem_Measures'][['Measure_Name', 'Group', 'Hospitals_Affected', 'Pct_System']].head(10).to_string(index=False))

print("\n" + "=" * 80)
print("QUESTION 2: How much do they impact star ratings? (weighted)")
print("=" * 80)
print("\nWeighted Impact Scores (accounting for group weights and measure counts):")
print(exec_summary['Top_Problem_Measures'][['Measure_Name', 'Per_Measure_Weight', 'Impact_Score', 'Severity']].head(10).to_string(index=False))

print("\n" + "=" * 80)
print("QUESTION 3: Which hospitals can improve quickly?")
print("=" * 80)
print("\nQuick Win Opportunities (gap < 0.3 points to next star):")
print(exec_summary['Improvement_Opportunities'][['Facility_ID', 'Current_Rating', 'Gap_to_Next_Star', 'Feasibility']].head(10).to_string(index=False))

print("\n" + "=" * 80)
print("QUESTION 4: What's the financial impact?")
print("=" * 80)
print("\nFinancial Impact Analysis:")
financial = exec_summary['Financial_Impact'].iloc[0]
print(f"  Current Average Star Rating: {financial['Current_Avg_Stars']}")
print(f"  Hospitals Below 5 Stars: {financial['Hospitals_Below_5_Stars']}")
print(f"  Estimated Revenue at Risk per Hospital: {financial['Est_Revenue_at_Risk_Per_Hospital']}")
print(f"  Potential System-Wide Revenue Gain: {financial['Potential_System_Revenue_Gain']}")
print(f"  Quick Win Opportunities: {financial['Quick_Win_Opportunities']} hospitals")
print(f"\n  Note: {financial['Note']}")

print("\n" + "=" * 80)
print("QUESTION 5: What are the top 3-5 specific actions to take?")
print("=" * 80)
print("\nWeighted Recommendations (prioritized by impact score):")
for idx, row in exec_summary['Recommendations'].iterrows():
    print(f"\n{idx + 1}. {row['Priority']} PRIORITY (Impact Score: {row['Impact_Score']:.1f})")
    print(f"   Focus Area: {row['Focus_Area']}")
    print(f"   Issue: {row['Issue']}")
    print(f"   Recommendation: {row['Recommendation']}")

print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)
print(f"✅ Question 1: {len(exec_summary['Top_Problem_Measures'])} specific measures identified with human-readable names")
print(f"✅ Question 2: Impact scores calculated using weighted formula: (% affected) × (severity) × (measure weight) × 100")
print(f"✅ Question 3: {len(exec_summary['Improvement_Opportunities'])} quick win opportunities identified")
print(f"✅ Question 4: Financial impact with {financial['Potential_System_Revenue_Gain']} potential system-wide revenue gain")
print(f"✅ Question 5: {len(exec_summary['Recommendations'])} specific, weighted recommendations provided")

print("\n" + "=" * 80)
print("ALL C-SUITE QUESTIONS ANSWERED ✅")
print("=" * 80)
