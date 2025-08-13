import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import json
    from pathlib import Path
    import plotly.express as px
    import plotly.graph_objects as go
    return Path, go, json, pd, px


@app.cell
def _(Path, json):
    config_path = Path("config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"Site: {config['site']}")
    print(f"CLIF2 Path: {config['clif2_path']}")
    print(f"File Type: {config['filetype']}")
    return (config,)


@app.cell
def _(Path, config):
    clif_path = Path(config["clif2_path"])
    respiratory_file = clif_path / "clif_respiratory_support.parquet"

    print(f"Loading respiratory support data from: {respiratory_file}")
    return (respiratory_file,)


@app.cell
def _(pd, respiratory_file):
    df_full = pd.read_parquet(respiratory_file)
    print(f"Loaded {len(df_full):,} rows")

    # Keep only specified columns
    columns_to_keep = ['hospitalization_id', 'recorded_dttm', 'mode_name', 'mode_category']
    df_respiratory = df_full[columns_to_keep].copy()

    print(f"Keeping columns: {df_respiratory.columns.tolist()}")
    return (df_respiratory,)


@app.cell
def _(df_respiratory):
    mode_name_counts = df_respiratory['mode_name'].value_counts()
    print("Ventilation Mode Distribution:")
    print(mode_name_counts.head(15))
    return (mode_name_counts,)


@app.cell
def _(mode_name_counts, px):
    fig_modes = px.bar(
        x=mode_name_counts.values[:15], 
        y=mode_name_counts.index[:15],
        orientation='h',
        title="Top 15 Ventilation Modes",
        labels={'x': 'Count', 'y': 'Mode Name'}
    )
    fig_modes.update_layout(height=500)
    fig_modes
    return


@app.cell
def _(df_respiratory):
    mode_category_counts = df_respiratory['mode_category'].value_counts()
    print("Mode Category Distribution:")
    print(mode_category_counts)
    return (mode_category_counts,)


@app.cell
def _(mode_category_counts, px):
    fig_categories = px.pie(
        values=mode_category_counts.values,
        names=mode_category_counts.index,
        title="Distribution by Mode Category"
    )
    fig_categories.update_layout(height=400)
    fig_categories
    return


@app.cell
def _(df_respiratory):
    df_respiratory['mode_category'].unique()
    return


@app.cell
def _(df_respiratory, pd):
    df_with_dates = df_respiratory.copy()
    df_with_dates['recorded_dttm'] = pd.to_datetime(df_with_dates['recorded_dttm'])
    df_with_dates['date'] = df_with_dates['recorded_dttm'].dt.date
    daily_counts = df_with_dates.groupby('date').size()
    print(f"Data spans from {daily_counts.index.min()} to {daily_counts.index.max()}")
    return daily_counts, df_with_dates


@app.cell
def _(daily_counts, px):
    fig_daily_timeline = px.line(
        x=daily_counts.index, 
        y=daily_counts.values,
        title="Daily Respiratory Support Records",
        labels={'x': 'Date', 'y': 'Number of Records'}
    )
    fig_daily_timeline.update_layout(height=400)
    fig_daily_timeline
    return


@app.cell
def _(df_respiratory):
    missing_values = df_respiratory.isnull().sum()
    missing_pct = (missing_values / len(df_respiratory) * 100).round(2)
    print("\nMissing Values Summary:")
    for col in df_respiratory.columns:
        if missing_values[col] > 0:
            print(f"{col}: {missing_values[col]:,} ({missing_pct[col]:.2f}%)")
    return


@app.cell
def _(df_respiratory):
    # Mode transitions analysis
    hosp_mode_counts = df_respiratory.groupby('hospitalization_id')['mode_name'].nunique()
    print(f"Average number of mode changes per hospitalization: {hosp_mode_counts.mean():.2f}")
    print(f"Max mode changes in a single hospitalization: {hosp_mode_counts.max()}")
    return (hosp_mode_counts,)


@app.cell
def _(hosp_mode_counts, px):
    fig_transitions = px.histogram(
        hosp_mode_counts.values,
        nbins=30,
        title="Distribution of Mode Changes per Hospitalization",
        labels={'value': 'Number of Unique Modes', 'count': 'Number of Hospitalizations'}
    )
    fig_transitions.update_layout(showlegend=False, height=400)
    fig_transitions
    return


@app.cell
def _(df_with_dates):
    # Mode usage over time - first find most used mode per hospitalization per day
    mode_time_df = df_with_dates.groupby(['hospitalization_id', 'date', 'mode_category']).size().reset_index(name='count')

    # For each hospitalization-date combo, find the dominant mode category
    idx_mode_time = mode_time_df.groupby(['hospitalization_id', 'date'])['count'].idxmax()
    dominant_modes = mode_time_df.loc[idx_mode_time]

    # Now count dominant modes by date and category
    daily_dominant_counts = dominant_modes.groupby(['date', 'mode_category']).size().reset_index(name='count')
    pivot_df = daily_dominant_counts.pivot(index='date', columns='mode_category', values='count').fillna(0)

    return (pivot_df,)


@app.cell
def _(pivot_df):
    pivot_df
    return


@app.cell
def _(df_with_dates):
    # Filter for the 3 specific modes
    target_modes = [
        'Assist Control-Volume Control',
        'Pressure Support/CPAP',
        'Pressure-Regulated Volume Control'
    ]

    df_three_modes = df_with_dates[df_with_dates['mode_category'].isin(target_modes)].copy()

    return (df_three_modes,)


@app.cell
def _(df_three_modes, pd):


    # Extract year from recorded_dttm
    df_three_modes['year'] = df_three_modes['recorded_dttm'].dt.year

    # For each hospitalization and day, find the most used mode
    hosp_daily_mode = df_three_modes.groupby(['hospitalization_id', 'date', 'mode_category']).size().reset_index(name='count')

    # Get the dominant mode for each hospitalization-date combination
    idx_hosp_daily = hosp_daily_mode.groupby(['hospitalization_id', 'date'])['count'].idxmax()
    hosp_daily_dominant = hosp_daily_mode.loc[idx_hosp_daily]

    # Extract year from date for aggregation
    hosp_daily_dominant['year'] = pd.to_datetime(hosp_daily_dominant['date']).dt.year

    # Count unique hospitalization-days per mode per year
    yearly_modes = hosp_daily_dominant.groupby(['year', 'mode_category']).size().reset_index(name='hosp_days')

    # Calculate total hospitalization-days per year for percentage
    yearly_totals = yearly_modes.groupby('year')['hosp_days'].sum()
    yearly_modes['total'] = yearly_modes['year'].map(yearly_totals)
    yearly_modes['percentage'] = (yearly_modes['hosp_days'] / yearly_modes['total'] * 100).round(1)

    print("Yearly dominant mode distribution (hospitalization-days):")
    print(yearly_modes)
    print(f"\nTotal unique hospitalizations analyzed: {df_three_modes['hospitalization_id'].nunique()}")

    return (yearly_modes,)


@app.cell
def _(go, yearly_modes):
    # Create stacked bar chart with percentages
    years = sorted(yearly_modes['year'].unique())

    # Define colors for each mode
    mode_colors = {
        'Assist Control-Volume Control': '#66c266',  # green
        'Pressure Support/CPAP': '#ffcc00',  # yellow
        'Pressure-Regulated Volume Control': '#fdc086'  # R color
    }

    fig_yearly = go.Figure()

    # Add trace for each mode
    for mode in mode_colors.keys():
        mode_data = yearly_modes[yearly_modes['mode_category'] == mode]

        # Get values for all years (fill 0 if missing)
        values = []
        percentages = []
        for year in years:
            year_mode = mode_data[mode_data['year'] == year]
            if not year_mode.empty:
                values.append(year_mode['hosp_days'].values[0])
                percentages.append(year_mode['percentage'].values[0])
            else:
                values.append(0)
                percentages.append(0)

        # Replace display name for Pressure Support/CPAP
        display_name = 'Pressure Control' if mode == 'Pressure Support/CPAP' else mode

        fig_yearly.add_trace(go.Bar(
            name=display_name,
            x=years,
            y=values,
            text=[f'{p:.1f}%' if p > 0 else '' for p in percentages],
            textposition='inside',
            textfont=dict(color='white', size=12),
            marker_color=mode_colors[mode],
            hovertemplate='%{x}<br>' + display_name + '<br>Hosp-Days: %{y}<br>Percentage: %{text}<extra></extra>'
        ))

    fig_yearly.update_layout(
        barmode='stack',
        title='Yearly Usage of Top 3 Ventilation Modes (Most Used Mode per Hospitalization-Day)',
        xaxis_title='Year',
        yaxis_title='Number of Hospitalization-Days',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    fig_yearly
    return


@app.cell
def _(go, yearly_modes):
    # Create 100% stacked bar chart
    years_list_pct = sorted(yearly_modes['year'].unique())

    mode_colors_pct = {
        'Assist Control-Volume Control': '#66c266',  # green
        'Pressure Support/CPAP': '#ffcc00',  # yellow
        'Pressure-Regulated Volume Control': '#fdc086'  # R color
    }

    fig_yearly_pct = go.Figure()

    for mode_name_pct in mode_colors_pct.keys():
        mode_data_pct = yearly_modes[yearly_modes['mode_category'] == mode_name_pct]

        percentages_pct = []
        for yr in years_list_pct:
            year_mode_pct = mode_data_pct[mode_data_pct['year'] == yr]
            if not year_mode_pct.empty:
                percentages_pct.append(year_mode_pct['percentage'].values[0])
            else:
                percentages_pct.append(0)

        # Replace display name for Pressure Support/CPAP
        display_name_pct = 'Pressure Control' if mode_name_pct == 'Pressure Support/CPAP' else mode_name_pct

        fig_yearly_pct.add_trace(go.Bar(
            name=display_name_pct,
            x=years_list_pct,
            y=percentages_pct,
            text=[f'{p:.1f}%' if p > 0 else '' for p in percentages_pct],
            textposition='inside',
            textfont=dict(color='white', size=14, family='Arial Black'),
            marker_color=mode_colors_pct[mode_name_pct],
            hovertemplate='%{x}<br>' + display_name_pct + '<br>Percentage: %{text}<extra></extra>'
        ))

    fig_yearly_pct.update_layout(
        barmode='stack',
        title='Yearly Ventilation Mode Distribution (Percentage)',
        xaxis_title='Year',
        yaxis_title='Percentage (%)',
        yaxis=dict(range=[0, 100]),
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        bargap=0.15
    )

    fig_yearly_pct
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
