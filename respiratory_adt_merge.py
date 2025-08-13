import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import json
    from pathlib import Path
    import numpy as np
    return Path, json, pd


@app.cell
def _(Path, json):
    config_path = Path("config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"Site: {config['site']}")
    print(f"CLIF2 Path: {config['clif2_path']}")
    return (config,)


@app.cell
def _(Path, config):
    clif_path = Path(config["clif2_path"])
    respiratory_file = clif_path / "clif_respiratory_support.parquet"
    adt_file = clif_path / "clif_adt.parquet"

    print(f"Respiratory support file: {respiratory_file}")
    print(f"ADT file: {adt_file}")
    return adt_file, respiratory_file


@app.cell
def _(pd, respiratory_file):
    df_respiratory_full = pd.read_parquet(respiratory_file)
    print(f"Loaded {len(df_respiratory_full):,} respiratory support rows")
    print(f"Columns: {df_respiratory_full.columns.tolist()}")

    # Keep only necessary columns
    respiratory_cols = ['hospitalization_id', 'recorded_dttm', 'mode_category']
    df_respiratory = df_respiratory_full[respiratory_cols].copy()

    # Convert datetime
    df_respiratory['recorded_dttm'] = pd.to_datetime(df_respiratory['recorded_dttm'])

    print(f"\nRespiratory data shape: {df_respiratory.shape}")
    df_respiratory.head()
    return (df_respiratory,)


@app.cell
def _(adt_file, pd):
    df_adt_full = pd.read_parquet(adt_file)
    print(f"Loaded {len(df_adt_full):,} ADT rows")
    print(f"Columns: {df_adt_full.columns.tolist()}")

    # Keep only necessary columns
    adt_cols = ['hospitalization_id', 'in_dttm', 'out_dttm', 'location_name']
    df_adt = df_adt_full[adt_cols].copy()

    # Convert datetime columns
    df_adt['in_dttm'] = pd.to_datetime(df_adt['in_dttm'])
    df_adt['out_dttm'] = pd.to_datetime(df_adt['out_dttm'])

    print(f"\nADT data shape: {df_adt.shape}")
    df_adt.head()
    return (df_adt,)


@app.cell
def _(df_adt, df_respiratory):
    # Check unique hospitalizations in each dataset
    resp_hosp_ids = df_respiratory['hospitalization_id'].unique()
    adt_hosp_ids = df_adt['hospitalization_id'].unique()

    print(f"Unique hospitalizations in respiratory: {len(resp_hosp_ids):,}")
    print(f"Unique hospitalizations in ADT: {len(adt_hosp_ids):,}")

    # Find overlap
    common_hosp_ids = set(resp_hosp_ids) & set(adt_hosp_ids)
    print(f"Common hospitalizations: {len(common_hosp_ids):,}")
    return


@app.cell
def _(df_adt, df_respiratory, pd):
    # Perform the merge using merge_asof
    # First sort both dataframes
    df_respiratory_sorted = df_respiratory.sort_values(['hospitalization_id', 'recorded_dttm'])
    df_adt_sorted = df_adt.sort_values(['hospitalization_id', 'in_dttm'])

    # We'll need to do this hospitalization by hospitalization for proper interval matching
    merged_list = []

    # Get unique hospitalization IDs that exist in both datasets
    common_ids = set(df_respiratory_sorted['hospitalization_id'].unique()) & set(df_adt_sorted['hospitalization_id'].unique())

    print(f"Processing {len(common_ids)} hospitalizations...")

    # Process in batches for efficiency
    for hosp_id in list(common_ids)[:1000]:  # Limit to first 1000 for testing
        resp_subset = df_respiratory_sorted[df_respiratory_sorted['hospitalization_id'] == hosp_id]
        adt_subset = df_adt_sorted[df_adt_sorted['hospitalization_id'] == hosp_id]

        # For each respiratory record, find the ADT record it falls within
        for _, resp_row in resp_subset.iterrows():
            recorded_time = resp_row['recorded_dttm']

            # Find ADT records where recorded_dttm is between in_dttm and out_dttm
            matching_adt = adt_subset[
                (adt_subset['in_dttm'] <= recorded_time) & 
                (adt_subset['out_dttm'] >= recorded_time)
            ]

            if not matching_adt.empty:
                # Take the first match if multiple (shouldn't happen often)
                adt_row = matching_adt.iloc[0]
                merged_row = {
                    'hospitalization_id': hosp_id,
                    'recorded_dttm': recorded_time,
                    'in_dttm': adt_row['in_dttm'],
                    'out_dttm': adt_row['out_dttm'],
                    'mode_category': resp_row['mode_category'],
                    'location_name': adt_row['location_name']
                }
                merged_list.append(merged_row)

    # Create final dataframe
    df_merged = pd.DataFrame(merged_list)
    print(f"\nMerged dataset shape: {df_merged.shape}")
    return (df_merged,)


@app.cell
def _(df_merged):
    # Display the final merged dataset
    print("Final merged dataset:")
    print(f"Shape: {df_merged.shape}")
    print(f"Columns: {df_merged.columns.tolist()}")
    print(f"\nFirst 10 rows:")
    df_merged.head(10)
    return


@app.cell
def _(df_merged):
    # Check for missing values
    print("\nMissing values in final dataset:")
    missing = df_merged.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            pct = (count / len(df_merged)) * 100
            print(f"{col}: {count:,} ({pct:.2f}%)")
    return


if __name__ == "__main__":
    app.run()
