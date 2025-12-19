
import plotly.graph_objects as go
import numpy as np

def add_dim_line(fig, p1, p2, label, offset=10, color="gray", row=None, col=None):
    """
    Draws a dimension line between p1 and p2 with an offset.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    v = p2 - p1
    length = np.linalg.norm(v)
    if length < 1e-3: return

    u = v / length
    n = np.array([-u[1], u[0]]) # Normal vector 

    # Offset points
    p1_off = p1 + n * offset
    p2_off = p2 + n * offset

    # Main Dim Line
    fig.add_trace(go.Scatter(
        x=[p1_off[0], p2_off[0]], y=[p1_off[1], p2_off[1]],
        mode='lines+text',
        line=dict(color=color, width=1, dash='solid'),
        text=[None], # No text on endpoints
        hoverinfo='skip', showlegend=False
    ), row=row, col=col)

    # Extension lines (from object to dim line)
    # Gap from object
    gap = 2
    p1_gap = p1 + n * gap
    p2_gap = p2 + n * gap
    
    fig.add_trace(go.Scatter(
        x=[p1_gap[0], p1_off[0] + n[0]*2], y=[p1_gap[1], p1_off[1] + n[1]*2], # Little overshoot
        mode='lines', line=dict(color=color, width=1), hoverinfo='skip', showlegend=False
    ), row=row, col=col)
    
    fig.add_trace(go.Scatter(
        x=[p2_gap[0], p2_off[0] + n[0]*2], y=[p2_gap[1], p2_off[1] + n[1]*2],
        mode='lines', line=dict(color=color, width=1), hoverinfo='skip', showlegend=False
    ), row=row, col=col)

    # Arrows/Ticks (Simple dots for now or small arrows)
    # Plotly has arrowheads on lines, but hard to control middle. 
    # Let's just use the label.
    
    mid = (p1_off + p2_off) / 2
    
    # Text Annotation
    fig.add_annotation(
        x=mid[0], y=mid[1],
        text=label,
        showarrow=False,
        font=dict(size=10, color=color),
        bgcolor="rgba(0,0,0,0.5)",
        row=row, col=col
    )

def add_force_arrow(fig, start, force_vec, label, row=None, col=None, color="red", scale_factor=0.2):
    """
    Draws a fixed-length force arrow with label showing magnitude.
    Fixed length logic: Using a constant pixel length for visualization (e.g., 60px equivalent).
    """
    fx, fy = force_vec
    mag = np.hypot(fx, fy)
    if mag < 1e-2: return

    # Fixed Length Vector
    L_VISUAL = 60 
    ux, uy = fx/mag, fy/mag
    
    end = (start[0] + ux * L_VISUAL, start[1] + uy * L_VISUAL)
    
    # Arrow Line
    fig.add_trace(go.Scatter(
        x=[start[0], end[0]], y=[start[1], end[1]],
        mode='lines',
        line=dict(color=color, width=3),
        hoverinfo='skip', showlegend=False
    ), row=row, col=col)
    
    # Arrow Head
    angle = np.degrees(np.arctan2(uy, ux))
    fig.add_trace(go.Scatter(
        x=[end[0]], y=[end[1]],
        mode='markers',
        marker=dict(symbol="triangle-up", size=10, color=color, angle=angle-90),
        hoverinfo='skip', showlegend=False
    ), row=row, col=col)

    # Label (F = 123 N)
    lbl_text = f"<b>{label}</b> = {mag:.1f} N"
    
    # Smart Text Placement to avoid overlap
    # Vector direction: ux, uy
    
    if "W" in label:
        # Weight: Strictly BELOW the arrow tip
        # Assuming Weight vector points down, put label further down.
        # Use simple coordinate offset
        # Note: Plotly coords, assume y-up? Yes usually.
        # But if W points down (uy ~ -1), we want text below end_y.
        txt_x = end[0]
        txt_y = end[1] - 20 # 20 units below tip
        xanchor = "center"
        yanchor = "top" # Text box's top is at txt_y
        
    else:
        # Force or others (F)
        # If pointing down (uy < 0), avoid the "Below" space reserved for W.
        if uy < -0.3:
            # Shift Laterally
            if ux >= 0:
                # Pointing Down-Right or Down -> Place Right
                txt_x = end[0] + 20
                txt_y = end[1] + 10 # Slightly up to clear W label zone
                xanchor = "left"
                yanchor = "bottom"
            else:
                # Pointing Down-Left -> Place Left
                txt_x = end[0] - 20
                txt_y = end[1] + 10
                xanchor = "right"
                yanchor = "bottom"
        else:
            # Pointing Up or Horizontal -> Place at Tip extension
            txt_x = end[0] + ux * 15
            txt_y = end[1] + uy * 15
            xanchor = "center"
            yanchor = "middle"

    fig.add_annotation(
        x=txt_x, y=txt_y,
        text=lbl_text,
        showarrow=False,
        font=dict(color=color, size=12),
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor=color, borderwidth=1, borderpad=2,
        xanchor=xanchor, yanchor=yanchor,
        row=row, col=col
    )
