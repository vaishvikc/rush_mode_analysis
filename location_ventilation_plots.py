import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import plotly.graph_objects as go
    from pathlib import Path
    return Path, go, pd


@app.cell
def _(Path):
    # Create plots folder if it doesn't exist
    plots_folder = Path("plots")
    plots_folder.mkdir(exist_ok=True)
    print(f"Plots will be saved to: {plots_folder.absolute()}")
    return plots_folder,


@app.cell
def _(pd):
    # Load the merged data
    df_merged_raw = pd.read_parquet("respiratory_adt_merged.parquet")
    print(f"Loaded {len(df_merged_raw):,} rows")
    print(f"Columns: {df_merged_raw.columns.tolist()}")
    print(f"Unique locations: {df_merged_raw['location_name'].nunique()}")
    return df_merged_raw,


@app.cell
def _(df_merged_raw, pd):
    # Convert datetime columns and add date column
    df_merged = df_merged_raw.copy()
    df_merged['recorded_dttm'] = pd.to_datetime(df_merged['recorded_dttm'])
    df_merged['date'] = df_merged['recorded_dttm'].dt.date
    
    # Sort by hospitalization_id and recorded_dttm
    df_merged = df_merged.sort_values(['hospitalization_id', 'recorded_dttm'])
    
    # Forward fill mode_category within each hospitalization
    df_merged['mode_category'] = df_merged.groupby('hospitalization_id')['mode_category'].ffill()
    
    print(f"\nAfter forward fill:")
    print(f"  - Null mode_category: {df_merged['mode_category'].isnull().sum()}")
    print(f"  - Total records: {len(df_merged)}")
    
    return df_merged,


@app.cell
def _(df_merged):
    # Filter for the 3 specific modes
    target_modes = [
        'Assist Control-Volume Control',
        'Pressure Support/CPAP',
        'Pressure-Regulated Volume Control'
    ]

    df_three_modes = df_merged[df_merged['mode_category'].isin(target_modes)].copy()
    print(f"\nFiltered to {len(df_three_modes):,} rows with target modes")

    # Extract year from recorded_dttm
    df_three_modes['year'] = df_three_modes['recorded_dttm'].dt.year
    return df_three_modes, target_modes


@app.cell
def _():
    # Define colors for each mode
    mode_colors = {
        'Assist Control-Volume Control': '#66c266',  # green
        'Pressure Support/CPAP': '#ffcc00',  # yellow
        'Pressure-Regulated Volume Control': '#fdc086'  # R color
    }
    return mode_colors,


@app.cell
def _(df_three_modes):
    # Get all unique locations
    locations = sorted(df_three_modes['location_name'].unique())
    print(f"\nWill process {len(locations)} locations:")
    for loc in locations:
        count = len(df_three_modes[df_three_modes['location_name'] == loc])
        print(f"  - {loc}: {count:,} records")
    return locations,


@app.cell
def _(df_three_modes, go, locations, mode_colors, pd, plots_folder):
    # Process each location and create plots
    all_figures = {}
    location_stats = []

    for location in locations:
        # Filter data for this location
        df_location = df_three_modes[df_three_modes['location_name'] == location].copy()
        
        if len(df_location) == 0:
            continue
            
        # For each hospitalization and day, find the most used mode
        hosp_daily_mode = df_location.groupby(['hospitalization_id', 'date', 'mode_category']).size().reset_index(name='count')
        
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
        
        # Create 100% stacked bar chart
        years_list = sorted(yearly_modes['year'].unique())
        
        fig = go.Figure()
        
        for mode_name in mode_colors.keys():
            mode_data = yearly_modes[yearly_modes['mode_category'] == mode_name]
            
            percentages = []
            for yr in years_list:
                year_mode = mode_data[mode_data['year'] == yr]
                if not year_mode.empty:
                    percentages.append(year_mode['percentage'].values[0])
                else:
                    percentages.append(0)
            
            # Replace display name for Pressure Support/CPAP
            display_name = 'Pressure Control' if mode_name == 'Pressure Support/CPAP' else mode_name
            
            fig.add_trace(go.Bar(
                name=display_name,
                x=years_list,
                y=percentages,
                text=[f'{p:.1f}%' if p > 0 else '' for p in percentages],
                textposition='inside',
                textfont=dict(color='white', size=14, family='Arial Black'),
                marker_color=mode_colors[mode_name],
                hovertemplate='%{x}<br>' + display_name + '<br>Percentage: %{text}<extra></extra>'
            ))
        
        fig.update_layout(
            barmode='stack',
            title=f'Ventilation Mode Distribution - {location}',
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
        
        # Save the plot as PNG
        filename = f"{location.replace('/', '_').replace(' ', '_')}.png"
        filepath = plots_folder / filename
        fig.write_image(str(filepath), width=1200, height=800, scale=2)
        
        # Store figure for display
        all_figures[location] = fig
        
        # Collect statistics
        location_stats.append({
            'location': location,
            'records': len(df_location),
            'hospitalizations': df_location['hospitalization_id'].nunique(),
            'years': len(years_list),
            'file': filename
        })

    print(f"\n✅ Generated {len(all_figures)} plots")
    return all_figures, location_stats


@app.cell
def _(location_stats, pd):
    # Display summary statistics
    stats_df = pd.DataFrame(location_stats)
    print("\nLocation Statistics Summary:")
    print(stats_df.to_string(index=False))
    stats_df
    return stats_df,


@app.cell
def _(all_figures):
    # Display the last plot (or any specific plot)
    if all_figures:
        last_location = list(all_figures.keys())[-1]
        print(f"\nDisplaying plot for: {last_location}")
        all_figures[last_location]
    return last_location,


@app.cell
def _(plots_folder):
    # List all saved plots
    import os
    saved_files = sorted([f for f in os.listdir(plots_folder) if f.endswith('.png')])
    print(f"\nSaved {len(saved_files)} plot files:")
    for f in saved_files:
        print(f"  - {f}")
    return os, saved_files


if __name__ == "__main__":
    app.run()