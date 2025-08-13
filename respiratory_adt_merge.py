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
    # Optimized merge using inner join and vectorized filtering
    print("Starting optimized merge...")
    
    # Inner join on hospitalization_id to get all combinations
    df_merged = pd.merge(
        df_respiratory,
        df_adt,
        on='hospitalization_id',
        how='inner'
    )
    
    print(f"After inner join: {len(df_merged):,} rows")
    
    # Filter to keep only rows where recorded_dttm falls within ADT interval
    df_merged = df_merged[
        (df_merged['recorded_dttm'] >= df_merged['in_dttm']) &
        (df_merged['recorded_dttm'] <= df_merged['out_dttm'])
    ]
    
    print(f"After filtering by time intervals: {len(df_merged):,} rows")
    
    # Sort by hospitalization_id and recorded_dttm for consistency
    df_merged = df_merged.sort_values(['hospitalization_id', 'recorded_dttm'])
    
    # If there are duplicates (respiratory record matches multiple ADT intervals), keep first
    df_merged = df_merged.drop_duplicates(
        subset=['hospitalization_id', 'recorded_dttm', 'mode_category'],
        keep='first'
    )
    
    print(f"\nFinal merged dataset shape: {df_merged.shape}")
    print(f"Unique hospitalizations: {df_merged['hospitalization_id'].nunique():,}")
    
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


@app.cell
def _(df_merged):
    # Save the merged dataset to parquet
    output_file = "respiratory_adt_merged.parquet"
    df_merged.to_parquet(output_file, index=False)
    print(f"\nMerged data saved to: {output_file}")
    print(f"File contains {len(df_merged):,} rows")
    return


if __name__ == "__main__":
    app.run()
