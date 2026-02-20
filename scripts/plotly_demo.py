"""Quick demo: Polars + Plotly Express / .plot accessor → browser."""

import plotly.express as px
import polars as pl

df = pl.DataFrame(
    {
        'hour': list(range(24)),
        'solar': [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.1,
            0.4,
            0.7,
            0.9,
            1.0,
            1.0,
            0.95,
            0.9,
            0.85,
            0.7,
            0.5,
            0.2,
            0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        'wind': [
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.5,
            0.4,
            0.3,
            0.25,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.5,
            0.6,
            0.65,
            0.7,
            0.65,
            0.55,
            0.45,
            0.4,
            0.35,
        ],
    }
)

long = df.unpivot(index='hour', on=['solar', 'wind'], variable_name='source', value_name='capacity_factor')

# --- Plotly Express (opens in browser) ---
fig = px.line(long, x='hour', y='capacity_factor', color='source', title='Plotly: Renewable Capacity Factors')
fig.show()

# --- Built-in .plot accessor (hvPlot/Bokeh) ---
long.plot.line(x='hour', y='capacity_factor', color='source')
