"""
Import hospital data from SAS file to Supabase database tables.

This script reads your SAS file and creates Supabase tables with all the data.
Once imported, the app will query data directly from Supabase.
"""

from supabase import create_client
import pandas as pd
import pyreadstat

# Supabase credentials
SUPABASE_URL = input("Enter your Supabase URL: ").strip()
SUPABASE_KEY = input("Enter your Supabase service role key: ").strip()

# File to import
file_path = "/Users/heilman/Desktop/All Data July 2025.sas7bdat"

try:
    # Initialize Supabase client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    print(f"\n📁 Reading SAS file...")

    # Read SAS file
    df, meta = pyreadstat.read_sas7bdat(file_path)

    print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
    print(f"\nColumns: {list(df.columns)[:10]}..." if len(df.columns) > 10 else f"\nColumns: {list(df.columns)}")

    # Clean column names for SQL (remove spaces, special chars)
    df.columns = [col.strip().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace('/', '_') for col in df.columns]

    # Convert to records
    print(f"\n📤 Uploading to Supabase table 'hospital_quality_data'...")

    # Insert in batches (Supabase has limits)
    batch_size = 1000
    total_rows = len(df)

    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i+batch_size]
        records = batch.to_dict('records')

        # Convert NaN to None for JSON compatibility
        for record in records:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None

        try:
            supabase.table('hospital_quality_data').insert(records).execute()
            print(f"  Uploaded rows {i+1}-{min(i+batch_size, total_rows)} of {total_rows}")
        except Exception as e:
            print(f"  ⚠️  Error on batch {i//batch_size + 1}: {str(e)}")
            print(f"  Continuing with next batch...")

    print(f"\n✅ Successfully imported {total_rows} rows to Supabase!")
    print("\n📝 Next steps:")
    print("1. Go to your Streamlit Cloud app settings")
    print("2. Add these secrets:")
    print(f'   SUPABASE_URL = "{SUPABASE_URL}"')
    print(f'   SUPABASE_KEY = "{SUPABASE_KEY}"')
    print("\n🚀 After adding secrets, your app will load data from Supabase!")

except FileNotFoundError:
    print(f"❌ Error: File not found at {file_path}")
    print("Please update the file_path variable in this script.")
except Exception as e:
    print(f"❌ Error: {str(e)}")
    print("\nMake sure you have:")
    print("1. Created a Supabase project at https://supabase.com")
    print("2. Created a table named 'hospital_quality_data' (or let this script create it)")
    print("3. Used your service role key (not anon key)")
    print("\nTo create the table in Supabase:")
    print("Go to SQL Editor and run:")
    print("CREATE TABLE hospital_quality_data (id BIGSERIAL PRIMARY KEY, data JSONB);")
