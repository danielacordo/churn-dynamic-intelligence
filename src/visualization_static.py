from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

PALETTE = {
    'churn': '#ff4d6d',
    'no_churn': '#4cc9f0',
    'accent': '#f8961e',
    'neutral': '#8d8dbd',
    'bg': '#0f0f1a',
    'panel': '#1a1a2e',
    'text': '#dcdcf0',
    'border': '#333355',
    'green': '#06ffa5',
    'purple': '#9b5de5',
    'yellow': '#ffbe0b',}

_DARK_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0f0f1e',
    font=dict(color='#dcdcf0', family='JetBrains Mono, monospace'),
    xaxis=dict(gridcolor='#1e1e3a', zerolinecolor='#1e1e3a'),
    yaxis=dict(gridcolor='#1e1e3a', zerolinecolor='#1e1e3a'),
    legend=dict(bgcolor='#0f0f1e', bordercolor='#1e1e3a', font=dict(color='#dcdcf0')),
    margin=dict(l=40, r=20, t=50, b=40),)


def set_style():
    """Applys dark theme to all subsequent matplotlib plots"""
    plt.rcParams.update({
        'figure.facecolor': PALETTE['bg'],
        'axes.facecolor': PALETTE['panel'],
        'axes.edgecolor': PALETTE['border'],
        'axes.labelcolor': PALETTE['text'],
        'axes.titlecolor': PALETTE['text'],
        'xtick.color': PALETTE['text'],
        'ytick.color': PALETTE['text'],
        'text.color': PALETTE['text'],
        'legend.facecolor': PALETTE['panel'],
        'legend.edgecolor': PALETTE['border'],
        'grid.color': PALETTE['border'],
        'grid.alpha': 0.4,
        'figure.dpi': 120,
        'axes.grid': True,
        'font.family': 'monospace',})


# plot_trajectory (matplotlib, with uncertainty band)
def plot_trajectory(sol, E_critical: float = 0.25, E_eq: float | None = None,
    sigma_E: float | None = None, title: str = 'Customer Energy Trajectory', ax=None,):
    """Energy trajectory with optional ±σ uncertainty band.

    The churn zone (E < E_critical) is filled with a subtle red gradient.
    If sigma_E is provided, an additional ±1σ band is drawn, showing the range of possible trajectories under parameter uncertainty"""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))

    t, E = sol.t, sol.y[0]

    # Churn danger zone 
    ax.fill_between(t, 0, E_critical, alpha=0.07, color=PALETTE['churn'], zorder=0)

    # Uncertainty band 
    if sigma_E is not None and sigma_E > 0:
        ax.fill_between(t, np.clip(E - sigma_E, 0, 1), np.clip(E + sigma_E, 0, 1),
                        alpha=0.18, color=PALETTE['accent'], label=f'±{sigma_E:.2f}σ band', zorder=1)

    # Main trajectory
    ax.plot(t, E, color=PALETTE['accent'], linewidth=2.5, label='E(t)', zorder=3)

    # Highlight churn crossing
    below = E < E_critical
    if below.any():
        ax.fill_between(t, E, E_critical, where=below, alpha=0.35, color=PALETTE['churn'], zorder=2, label='Churn zone crossed')

    # Reference lines 
    ax.axhline(E_critical, color=PALETTE['churn'], linestyle='--', linewidth=1.4, alpha=0.85, label=f'E_critical = {E_critical}', zorder=4)

    if E_eq is not None:
        ax.axhline(E_eq, color=PALETTE['no_churn'], linestyle=':', linewidth=1.4, alpha=0.85, label=f'E_eq = {E_eq:.2f}', zorder=4)
        # Arrow annotation showing direction of movement
        mid = len(t) // 2
        arrow_y = float(E[mid])
        dy = float(E_eq - arrow_y)
        if abs(dy) > 0.02:
            ax.annotate('', xy=(t[mid], arrow_y + dy * 0.6), xytext=(t[mid], arrow_y), arrowprops=dict(arrowstyle='->', color=PALETTE['no_churn'], lw=1.2, alpha=0.6,),)

    ax.set_xlabel('Time (months)', fontsize=10)
    ax.set_ylabel('Energy E(t)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=8, loc='upper right')
    return ax


# plot_phase_space (with KDE contours) 
def plot_phase_space(df, x: str = 'E_eq', y: str = 'E0', color_col: str = 'Churn_bin', ax=None,
    show_kde: bool = True, show_quadrants: bool = True,):
    """Phase space E0 vs E_eq with KDE density contours per class"""
    from src.physics import E_CRITICAL, E0_AT_RISK_THRESHOLD

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    churners = df[df[color_col] == 1][[x, y]].dropna()
    non_churners = df[df[color_col] == 0][[x, y]].dropna()

    x_critical = E_CRITICAL if x == 'E_eq' else (E0_AT_RISK_THRESHOLD if x == 'E0' else E_CRITICAL)
    y_critical = E_CRITICAL if y == 'E_eq' else (E0_AT_RISK_THRESHOLD if y == 'E0' else E_CRITICAL)

    # Quadrant shading 
    if show_quadrants:
        ecx, ecy = x_critical, y_critical
        quad_colors = [
            ([0, ecx, 0, ecy], PALETTE['churn'], 0.06),  # bottom-left: structural risk
            ([ecx, 1, 0, ecy], PALETTE['yellow'], 0.04),  # bottom-right: fragile
            ([0, ecx, ecy, 1], PALETTE['accent'], 0.04),  # top-left: resilient but low Eeq
            ([ecx, 1, ecy, 1], PALETTE['no_churn'], 0.05),  # top-right: stable
        ]
        labels = [
            (ecx/2, ecy/2, 'Structural\nRisk', PALETTE['churn']),
            ((ecx+1)/2, ecy/2, 'Fragile', PALETTE['yellow']),
            (ecx/2, (ecy+1)/2, 'Low Eeq\nResilient',PALETTE['accent']),
            ((ecx+1)/2, (ecy+1)/2, 'Stable', PALETTE['no_churn']),]
        
        for (x0, x1, y0, y1), color, alpha in quad_colors:
            ax.fill_betweenx([y0, y1], x0, x1, alpha=alpha, color=color)
        for lx, ly, label, color in labels:
            ax.text(lx, ly, label, ha='center', va='center', fontsize=7, color=color, alpha=0.7, fontweight='bold', style='italic')

    # KDE contours 
    if show_kde:
        grid_x = np.linspace(0, 1, 60)
        grid_y = np.linspace(0, 1, 60)
        XX, YY = np.meshgrid(grid_x, grid_y)
        positions = np.vstack([XX.ravel(), YY.ravel()])

        for pts, color in [(non_churners, PALETTE['no_churn']), (churners, PALETTE['churn'])]:
            if len(pts) < 5:
                continue
            try:
                kde = gaussian_kde(pts[[x, y]].values.T, bw_method=0.25)
                Z = kde(positions).reshape(XX.shape)
                Z = Z / Z.max()
                ax.contourf(XX, YY, Z, levels=[0.15, 0.35, 0.60, 0.85, 1.0], colors=[color] * 5, alpha=0.08, zorder=1)
                ax.contour(XX, YY, Z, levels=[0.35, 0.70], colors=[color], linewidths=0.8, alpha=0.55, zorder=2)
            except Exception:
                pass

    # Scatter points 
    ax.scatter(non_churners[x], non_churners[y], c=PALETTE['no_churn'], alpha=0.30, s=10, label='No churn', linewidths=0, zorder=3)
    ax.scatter(churners[x], churners[y], c=PALETTE['churn'], alpha=0.40, s=10, label='Churn', linewidths=0, zorder=4)

    # Critical threshold lines 
    ax.axvline(x_critical, color='white', linestyle='--', alpha=0.45, linewidth=1, zorder=5)
    ax.axhline(y_critical, color='white', linestyle='--', alpha=0.45, linewidth=1, zorder=5)

    ax.set_xlabel('E_eq - Equilibrium energy (structural loyalty)', fontsize=9)
    ax.set_ylabel('E0 - Current energy (observed loyalty)', fontsize=9)
    ax.set_title('Phase space: E0 vs E_eq', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


def plot_relaxation(df, ax=None):
    """Relaxation time (τ) distribution by contract type"""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))

    contracts = ['Month-to-month', 'One year', 'Two year']
    colors = [PALETTE['churn'], PALETTE['accent'], PALETTE['no_churn']]
    col = 'Contract' if 'Contract' in df.columns else None

    tau_global = df['tau'].dropna().values
    tau_range  = np.linspace(max(0, tau_global.min() - 1), min(tau_global.max() + 5, 100), 300)

    for contract, color in zip(contracts, colors):
        if col is not None:
            sub = df[df[col] == contract]['tau'].dropna().values
        else:
            sub = tau_global

        if len(sub) < 5:
            continue

        try:
            kde = gaussian_kde(sub, bw_method=0.3)
            dens = kde(tau_range)
            dens = dens / dens.max()
            ax.plot(tau_range, dens, color=color, linewidth=2.2, label=f'{contract} (n={len(sub):,})', zorder=3)
            ax.fill_between(tau_range, 0, dens, alpha=0.10, color=color, zorder=2)
            mu = sub.mean()
            ax.axvline(mu, color=color, linewidth=1.0, linestyle=':', alpha=0.7, zorder=4)
            ax.text(mu, dens[np.argmin(np.abs(tau_range - mu))] + 0.03,
                    f'μ={mu:.1f}', color=color, fontsize=8, ha='center', va='bottom', zorder=5)
        except Exception:
            ax.hist(sub, bins=25, alpha=0.4, color=color, label=contract, density=True, edgecolor='none')

    ax.set_xlabel('τ - Relaxation time (months)', fontsize=10)
    ax.set_ylabel('Density (normalised)', fontsize=10)
    ax.set_title('Relaxation time distribution by contract type', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    return ax


# plot_prediction_uncertainty (hexbin density + CI bars)
def plot_prediction_uncertainty(df, ax=None):
    """Uncertainty scatter: P(churn) vs σ with hexbin density overlay"""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    all_prob = df['prob_churn'].values
    all_sigma = df['sigma_prob'].values

    # Hexbin density 
    hb = ax.hexbin(all_prob, all_sigma, gridsize=35, cmap='YlOrRd', mincnt=1, linewidths=0.2, alpha=0.75, zorder=1)
    cb = plt.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label('Count', fontsize=8)

    # Per-risk mean ± std markers 
    risk_styles = [
        ('HIGH', PALETTE['churn'], 'D', 9),
        ('MEDIUM', PALETTE['accent'], 's', 9),
        ('LOW', PALETTE['no_churn'], 'o', 9),
        ('UNCERTAIN', PALETTE['purple'], '^', 9),]
    
    col = 'risk_level' if 'risk_level' in df.columns else None
    for risk, color, marker, ms in risk_styles:
        sub = df[df[col] == risk] if col else df
        if len(sub) < 3:
            continue
        mx, my = sub['prob_churn'].mean(), sub['sigma_prob'].mean()
        sx, sy = sub['prob_churn'].std(), sub['sigma_prob'].std()
        ax.errorbar(mx, my, xerr=sx, yerr=sy, fmt=marker, color=color, markersize=ms, elinewidth=1.2, capsize=4, capthick=1.2, label=f'{risk} (n={len(sub):,})', zorder=4)

    ax.set_xlabel('P(churn)', fontsize=10)
    ax.set_ylabel('σ - Prediction uncertainty', fontsize=10)
    ax.set_title('Prediction uncertainty by risk level (hexbin + mean ± std)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, markerscale=1.2)
    return ax


# plot_state_distribution 
def state_distribution(df, ax=None):
    """Bar chart of customer distribution by physical state or risk level """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    col = next((c for c in ('physical_state', 'estado_fisico', 'risk_level')
                if c in df.columns), None)
    if col is None:
        ax.set_title('No state column found')
        return ax

    counts = df[col].value_counts()
    palette_vals = [PALETTE['churn'], PALETTE['accent'], PALETTE['no_churn'], PALETTE['neutral']]
    colors = (palette_vals * 4)[:len(counts)]
    counts.plot(kind='bar', ax=ax, color=colors, edgecolor='none')
    ax.set_xlabel('State')
    ax.set_ylabel('Customers')
    ax.set_title('Customer distribution by state')
    ax.tick_params(axis='x', rotation=15)
    return ax


def plot_physical_states(df, ax=None):
    """Bar chart of physical states"""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    col = 'physical_state' if 'physical_state' in df.columns else 'estado_fisico'
    if col not in df.columns:
        ax.set_title('No physical_state column')
        return ax
    counts = df[col].value_counts()
    colors = [PALETTE['churn'], PALETTE['accent'], PALETTE['no_churn']][:len(counts)]
    counts.plot(kind='bar', ax=ax, color=colors, edgecolor='none')
    ax.set_xlabel('Physical state')
    ax.set_ylabel('Customers')
    ax.set_title('Customer distribution by physical state')
    ax.tick_params(axis='x', rotation=0)
    return ax


# Aliases for notebook compatibility
def state_distribution_v1(df): return state_distribution(df)
def plot_trajectory_v1(sol, **kw): return plot_trajectory(sol, **kw)
def plot_phase_space_v1(df, **kw): return plot_phase_space(df, **kw)
