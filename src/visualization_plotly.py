from __future__ import annotations
from typing import Any
import numpy as np
from scipy.stats import gaussian_kde
from src.visualization_static import _DARK_LAYOUT, PALETTE

# KDE precomputation cache
_KDE_CACHE: dict = {}

def _compute_kde_density(pts_array: np.ndarray, grid: np.ndarray, bw: float):
    """Returns normalised KDE density on the (grid x grid) meshgrid
    Result is cached by (n_pts, bw), safe because the dataset doesn't change mid-session and the cache key captures both size and bandwidth."""
    key = (len(pts_array), bw)
    if key in _KDE_CACHE:
        return _KDE_CACHE[key]
    XX, YY = np.meshgrid(grid, grid)
    pos = np.vstack([XX.ravel(), YY.ravel()])
    kde = gaussian_kde(pts_array.T, bw_method=bw)
    Z = kde(pos).reshape(XX.shape)
    Z = Z / Z.max()
    _KDE_CACHE[key] = (XX, YY, Z)
    return XX, YY, Z


def plot_phase_space_plotly(df, E_critical: float = 0.25):
    """ Interactive Plotly phase space: KDE-smoothed density heatmap with scatter overlay and four annotated quadrants"""
    import plotly.graph_objects as go

    churners = df[df['Churn_bin'] == 1][['E_eq', 'E0']].dropna()
    non_churners = df[df['Churn_bin'] == 0][['E_eq', 'E0']].dropna()

    fig = go.Figure()

    # KDE density for each class (cached)
    grid = np.linspace(0.01, 0.99, 80)

    for pts, colorscale, name in [
        (non_churners, [[0,'rgba(0,0,0,0)'],[1,'rgba(76,201,240,0.35)']], 'No churn density'),
        (churners, [[0,'rgba(0,0,0,0)'],[1,'rgba(255,77,109,0.40)']], 'Churn density'),]:
        if len(pts) < 5:
            continue
        try:
            XX, YY, Z = _compute_kde_density(pts.values, grid, bw=0.20)
            fig.add_trace(go.Contour(
                x=grid, y=grid, z=Z, colorscale=colorscale, showscale=False,
                contours=dict(start=0.10, end=1.0, size=0.18,coloring='fill', showlines=True),
                line=dict(width=0.6), name=name, hoverinfo='skip',))
        except Exception:
            pass

    # Scatter 
    for pts, color, name in [
        (non_churners, PALETTE['no_churn'], 'No churn'),
        (churners, PALETTE['churn'], 'Churn'),]:
        fig.add_trace(go.Scatter(
            x=pts['E_eq'], y=pts['E0'],
            mode='markers',
            marker=dict(color=color, size=4, opacity=0.35),
            name=name,))

    # Critical lines 
    fig.add_vline(x=E_critical, line_dash='dash', line_color='rgba(255,255,255,0.4)', line_width=1.2)
    fig.add_hline(y=E_critical, line_dash='dash', line_color='rgba(255,255,255,0.4)', line_width=1.2)

    # Quadrant annotations
    ec = E_critical
    annotations = [
        (ec/2, ec/2, 'Structural Risk', PALETTE['churn']),
        ((ec+1)/2, ec/2, 'Fragile', PALETTE['yellow']),
        (ec/2, (ec+1)/2, 'Low E_eq Resilient', PALETTE['accent']),
        ((ec+1)/2, (ec+1)/2, 'Stable', PALETTE['no_churn']),]
    
    for ax_, ay_, text, color in annotations:
        fig.add_annotation(x=ax_, y=ay_, text=f'<b>{text}</b>', showarrow=False, font=dict(color=color, size=10), opacity=0.7)

    fig.update_layout(**{**_DARK_LAYOUT, 'title': 'Phase Space — Customer Loyalty Dynamics', 'height': 420,})
    fig.update_xaxes(title_text='E_eq  (structural loyalty)', range=[0, 1])
    fig.update_yaxes(title_text='E0  (current loyalty)', range=[0, 1])
    return fig


# plot_trajectory_plotly (interactive, with uncertainty band)
def plot_trajectory_plotly(t: np.ndarray, E: np.ndarray, E_eq: float, E_critical: float = 0.25,
    sigma_upper: np.ndarray | None = None, sigma_lower: np.ndarray | None = None, perturbation_t: float | None = None,
    title: str = 'Customer Energy Trajectory',):
    """ Interactive Plotly trajectory with:
      - ±σ uncertainty band 
      - Churn danger zone fill
      - E_eq and E_critical reference lines
      - Optional vertical line at perturbation event
      - Colour-coded segment: safe (above E_critical) vs at-risk (below)"""
    import plotly.graph_objects as go

    fig = go.Figure()

    # Danger zone fill 
    fig.add_hrect(y0=0, y1=E_critical, fillcolor=PALETTE['churn'], opacity=0.07, layer='below', line_width=0)

    # Uncertainty band 
    if sigma_upper is not None and sigma_lower is not None:
        fig.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]), y=np.concatenate([sigma_upper, sigma_lower[::-1]]),
            fill='toself', fillcolor='rgba(248,150,30,0.12)', line=dict(color='rgba(0,0,0,0)'),
            name='±σ uncertainty', hoverinfo='skip',))

    # Trajectory, split by risk zone 
    # Colour segment green above E_critical, red below
    safe_mask = E >= E_critical
    churn_mask = E < E_critical

    if safe_mask.any():
        safe_idx = np.where(safe_mask)[0]
        fig.add_trace(go.Scatter(
            x=t[safe_idx], y=E[safe_idx], mode='lines',
            line=dict(color=PALETTE['accent'], width=2.5), name='E(t) - safe', showlegend=True,))
    if churn_mask.any():
        churn_idx = np.where(churn_mask)[0]
        fig.add_trace(go.Scatter(
            x=t[churn_idx], y=E[churn_idx], mode='lines',
            line=dict(color=PALETTE['churn'], width=2.5), name='E(t) - at risk', showlegend=True,))

    # Reference lines 
    fig.add_hline(y=float(E_critical), line_dash='dashdot', line_color=PALETTE['churn'], line_width=1.5, opacity=0.9)
    fig.add_annotation(x=1, xref='paper', y=float(E_critical), text=f'E_critical = {E_critical}', showarrow=False,
                       xanchor='right', yanchor='bottom', font=dict(color=PALETTE['churn'], size=10))

    fig.add_hline(y=float(E_eq), line_dash='dot', line_color=PALETTE['green'], line_width=1.5, opacity=0.8)
    fig.add_annotation(x=1, xref='paper', y=float(E_eq), text=f'E_eq = {E_eq:.2f}', showarrow=False,
                       xanchor='right', yanchor='top', font=dict(color=PALETTE['green'], size=10))

    if perturbation_t is not None:
        fig.add_vline(x=float(perturbation_t), line_dash='dot', line_color=PALETTE['yellow'], line_width=1.5)
        fig.add_annotation(x=float(perturbation_t), y=1, yref='paper', text='Perturbation', showarrow=False,
                           xanchor='left', yanchor='top', font=dict(color=PALETTE['yellow'], size=10))

    # Final point marker 
    final_color = PALETTE['churn'] if E[-1] < E_critical else PALETTE['green']
    fig.add_trace(go.Scatter(
        x=[t[-1]], y=[E[-1]],
        mode='markers+text',
        marker=dict(color=final_color, size=10, symbol='circle', line=dict(color='white', width=1.5)),
        text=[f'  E(t={t[-1]:.0f}) = {E[-1]:.3f}'],
        textposition='middle right', textfont=dict(color=final_color, size=10), showlegend=False, hoverinfo='skip',))

    fig.update_layout(**{**_DARK_LAYOUT, 'title': title, 'height': 400, 'legend': {**_DARK_LAYOUT['legend'], 'x': 0.01, 'y': 0.99},})
    fig.update_xaxes(title_text='Time (months)')
    fig.update_yaxes(title_text='Energy E(t)', range=[-0.03, 1.08])
    return fig


# plot_waterfall_roi (Plotly Waterfall)
def plot_waterfall_roi(revenue_at_risk: float, revenue_recovered: float, campaign_cost: float,
    net_roi: float, title: str = 'ROI Breakdown',) -> Any:
    """ Plotly Waterfall chart decomposing ROI into its components:
    Revenue at risk -> Revenue recovered -> Campaign cost -> Net ROI"""
    import plotly.graph_objects as go

    fig = go.Figure(go.Waterfall(
        name='ROI', orientation='v',
        measure=['absolute', 'relative', 'relative', 'total'],
        x=['Revenue<br>at risk', 'Revenue<br>recovered', 'Campaign<br>cost', 'Net ROI'],
        y=[revenue_at_risk, revenue_recovered - revenue_at_risk, -campaign_cost, 0],
        text=[f'${revenue_at_risk:,.0f}', f'+${revenue_recovered:,.0f}', f'−${campaign_cost:,.0f}', f'${net_roi:,.0f}'],
        textposition='outside', textfont=dict(color='#dcdcf0', size=12),
        connector=dict(line=dict(color='#333355', width=1.5, dash='dot')),
        increasing=dict(marker=dict(color=PALETTE['green'])),
        decreasing=dict(marker=dict(color=PALETTE['churn'])),
        totals=dict(marker=dict(color=PALETTE['green'] if net_roi >= 0 else PALETTE['churn'], line=dict(color='white', width=1.5),)),))

    fig.update_layout(**_DARK_LAYOUT, title=dict(text=title, font=dict(size=15)), yaxis_title='USD ($)',
        yaxis_tickprefix='$', yaxis_tickformat=',.0f', showlegend=False, height=380,)
    return fig


# plot_customer_funnel (Plotly Funnel)
def plot_customer_funnel(n_total: int, n_flagged: int,n_intervened: int, n_retained: int, title: str = 'Retention Campaign Funnel',) -> Any:
    """ Plotly Funnel chart: total customers -> flagged -> intervened -> retained"""
    import plotly.graph_objects as go

    stages = ['Total customers', 'Flagged as at-risk', 'Campaign sent', 'Successfully retained']
    values = [n_total, n_flagged, n_intervened, n_retained]
    colors  = [PALETTE['neutral'], PALETTE['accent'], PALETTE['yellow'], PALETTE['green']]

    # Conversion labels
    texts = [f'{n_total:,}']
    for prev, curr in zip(values, values[1:]):
        rate = curr / prev if prev > 0 else 0
        texts.append(f'{curr:,} ({rate:.0%} of prev.)')

    fig = go.Figure(go.Funnel(
        y=stages, x=values, textinfo='text', text=texts, textfont=dict(color='white', size=12),
        marker=dict(color=colors, line=dict(color=['rgba(255,255,255,0.2)'] * 4, width=1.5)),
        connector=dict(line=dict(color='rgba(255,255,255,0.15)', dash='dot', width=2)),opacity=0.85,))

    fig.update_layout(**{**_DARK_LAYOUT, 'title': dict(text=title, font=dict(size=15)), 'height': 370, 'margin': dict(l=160, r=30, t=60, b=30), 'funnelmode': 'stack',})
    return fig


# plot_heatmap_annotated (Plotly, with cell values) 
def plot_heatmap_annotated(z: np.ndarray, x_labels, y_labels, title: str = 'Heatmap', x_title: str = '',
    y_title: str = '', colorscale: str = 'RdYlGn', fmt: str = '${:,.0f}', vline_x: float | None = None,) -> Any:
    """ Plotly heatmap with cell value annotations"""
    import plotly.graph_objects as go

    # Cell annotations
    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            try:
                text = fmt.format(val)
            except Exception:
                text = str(round(val, 2))

            norm = (val - z.min()) / (z.max() - z.min() + 1e-15)
            font_color = 'white' if norm < 0.6 else '#0f0f1e'
            annotations.append(dict(
                x=j, y=i, text=text,
                showarrow=False,
                font=dict(size=9, color=font_color),
                xref='x', yref='y',))

    fig = go.Figure(go.Heatmap(z=z, x=list(range(len(x_labels))), y=list(range(len(y_labels))),
        colorscale=colorscale, showscale=True, hovertemplate='%{text}<extra></extra>',
        text=[[fmt.format(v) for v in row] for row in z],
        colorbar=dict(tickfont=dict(color='#dcdcf0', size=9), title=dict(text=title, font=dict(color='#dcdcf0')),),))

    if vline_x is not None:
        closest_idx = int(np.argmin([abs(float(str(lbl)) - vline_x)
                                      if str(lbl).replace('.','',1).isdigit()
                                      else 999
                                      for lbl in x_labels]))
        fig.add_vline(x=int(closest_idx), line_dash='dash', line_color=PALETTE['accent'], line_width=2)
        fig.add_annotation(x=int(closest_idx), y=1, yref='paper', text=f'Assumed: {vline_x}', showarrow=False, xanchor='left', yanchor='top', font=dict(color=PALETTE['accent'], size=10))

    fig.update_layout(**{**_DARK_LAYOUT, 'title': dict(text=title, font=dict(size=14)), 'annotations': annotations, 'height': 400,})
    fig.update_xaxes(title_text=x_title, ticktext=[str(lbl) for lbl in x_labels], tickvals=list(range(len(x_labels))))
    fig.update_yaxes(title_text=y_title, ticktext=[str(lbl) for lbl in y_labels], tickvals=list(range(len(y_labels))))
    return fig


