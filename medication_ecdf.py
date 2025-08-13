import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import json
    from pathlib import Path
    import plotly.graph_objects as go
    return Path, go, json, pd


@app.cell
def _(Path, json):
    config_path = Path("config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"Site: {config['site']}")
    print(f"CLIF2 Path: {config['clif2_path']}")
    return (config,)


@app.cell
def _(Path, config, pd):
    clif_path = Path(config["clif2_path"])
    medication_file = clif_path / "clif_medication_admin_continuous.parquet"

    print(f"Loading medication admin data from: {medication_file}")
    df_med = pd.read_parquet(medication_file)
    print(f"Loaded {len(df_med):,} rows")
    print(f"Columns: {df_med.columns.tolist()}")
    return (df_med,)


@app.cell
def _(df_med):
    df_med.columns
    return


@app.cell
def _(df_med):
    # Filter for vasoactives only
    df_vasoactives = df_med[df_med['med_group'] == 'vasoactives'].copy()
    print(f"Filtered to vasoactives: {len(df_vasoactives):,} rows")

    grouped = df_vasoactives.groupby(['med_category']).size().reset_index(name='count')
    print(f"Number of unique med_category values in vasoactives: {len(grouped)}")
    print(f"\nAll vasoactive categories by count:")
    print(grouped.sort_values('count', ascending=False))
    return (df_vasoactives,)


@app.cell
def _(df_vasoactives, go):
    import numpy as np
    from plotly.subplots import make_subplots

    # Get unique categories
    categories = df_vasoactives['med_category'].value_counts().index[:9]  # Limit to 9 for 3x3 grid

    # Create subplots
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[cat for cat in categories],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Add ECDF for each category
    for idx, med_cat in enumerate(categories):
        row = idx // 3 + 1
        col = idx % 3 + 1

        # Get values for this category
        values = df_vasoactives[df_vasoactives['med_category'] == med_cat]['med_dose'].dropna()
        
        # Remove outliers using IQR method
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        values = values[(values >= lower_bound) & (values <= upper_bound)].sort_values()

        if len(values) > 0:
            # Downsample if too many points
            if len(values) > 500:
                indices = np.linspace(0, len(values)-1, 500, dtype=int)
                values = values.iloc[indices]

            # Calculate ECDF
            y_ecdf = np.arange(1, len(values) + 1) / len(values)

            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=values,
                    y=y_ecdf,
                    mode='lines',
                    name=med_cat,
                    showlegend=False,
                    line=dict(width=2)
                ),
                row=row, col=col
            )

    # Update layout
    fig.update_layout(
        title_text='ECDF of Vasoactive Medications by Category',
        height=800,
        showlegend=False
    )

    # Update axes labels and formatting
    fig.update_xaxes(title_text="Unit Doses", row=3, col=2)
    fig.update_xaxes(tickformat=".3f")  # Format all x-axes to 3 decimal places
    fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)
    fig.update_yaxes(tickformat=".1%")  # Format y-axes as percentages

    fig
    return


@app.cell
def _(df_vasoactives):
    # Check dose statistics for each category
    for cat in df_vasoactives['med_category'].unique()[:5]:
        doses = df_vasoactives[df_vasoactives['med_category']==cat]['med_dose'].dropna()
        print(f"\n{cat}:")
        print(f"  Min: {doses.min():.3f}, Max: {doses.max():.3f}")
        print(f"  Median: {doses.median():.3f}, Mean: {doses.mean():.3f}")
        print(f"  95th percentile: {doses.quantile(0.95):.3f}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
