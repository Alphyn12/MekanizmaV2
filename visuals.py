import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- STYLE CONSTANTS ---

# --- STYLE CONSTANTS ---
COLOR_L2 = '#0074D9'    # Parlak Mavi
COLOR_L3 = '#2ECC40'    # Parlak Yeşil
COLOR_L4 = '#FF4136'    # Parlak Kırmızı
COLOR_GROUND = '#444444'
COLOR_PISTON = '#FF851B' # Turuncu
COLOR_JOINT_FILL = 'white'
COLOR_JOINT_LINE = 'black'

WIDTH_LINK = 15
WIDTH_GROUND = 8
SIZE_JOINT = 20
SIZE_PISTON = 40

def calc_angle_deg(p1, p2):
    """Calculates angle of vector p1->p2 in degrees."""
    return np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0]))

def create_angle_arc(center, radius, start_angle, end_angle, color, label=None, fill_color='rgba(255, 255, 255, 0.1)', text_size=12, row=None, col=None):
    """Creates a filled arc sector for angle visualization."""
    # Normalize angles
    t = np.linspace(start_angle, end_angle, 20)
    x = center[0] + radius * np.cos(np.radians(t))
    y = center[1] + radius * np.sin(np.radians(t))
    
    # Close the polygon for filling
    x_poly = np.concatenate(([center[0]], x, [center[0]]))
    y_poly = np.concatenate(([center[1]], y, [center[1]]))
    
    # Text position (midpoint of arc)
    mid_angle = (start_angle + end_angle) / 2
    txt_x = center[0] + (radius * 1.3) * np.cos(np.radians(mid_angle))
    txt_y = center[1] + (radius * 1.3) * np.sin(np.radians(mid_angle))
    
    sector = go.Scatter(
        x=x_poly, y=y_poly, 
        mode='lines', 
        fill='toself', 
        fillcolor=fill_color, 
        line=dict(color=color, width=1, dash='dot'), 
        hoverinfo='skip',
        showlegend=False
    )
    
    annotation = None
    if label:
        annotation = dict(
            x=txt_x, y=txt_y,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(color=color, size=text_size),
            bgcolor="rgba(0,0,0,0.5)"
        )
        if row is not None and col is not None:
             annotation['row'] = row
             annotation['col'] = col
             
    return sector, annotation


def add_dim_line(fig, p1, p2, label, offset=20, color="gray", row=None, col=None):
    """Draws a dimension line with offset and label."""
    p1 = np.array(p1)
    p2 = np.array(p2)
    v = p2 - p1
    L = np.linalg.norm(v)
    if L < 1e-3: return

    u = v / L
    n = np.array([-u[1], u[0]])
    
    p1_off = p1 + n * offset
    p2_off = p2 + n * offset
    
    # Main Line
    fig.add_trace(go.Scatter(
        x=[p1_off[0], p2_off[0]], y=[p1_off[1], p2_off[1]],
        mode='lines', line=dict(color=color, width=1),
        hoverinfo='skip', showlegend=False
    ), row=row, col=col)
    
    # Extension Lines
    fig.add_trace(go.Scatter(x=[p1[0], p1_off[0]], y=[p1[1], p1_off[1]], mode='lines', line=dict(color=color, width=1, dash='dot'), hoverinfo='skip', showlegend=False), row=row, col=col)
    fig.add_trace(go.Scatter(x=[p2[0], p2_off[0]], y=[p2[1], p2_off[1]], mode='lines', line=dict(color=color, width=1, dash='dot'), hoverinfo='skip', showlegend=False), row=row, col=col)
    
    # Label
    mid = (p1_off + p2_off) / 2
    fig.add_annotation(x=mid[0], y=mid[1], text=label, showarrow=False, font=dict(size=10, color=color), bgcolor="rgba(0,0,0,0.5)", row=row, col=col)


def add_vector_arrow(fig, start, components, label, color='#FFD700', fixed_length=60):
    """
    Adds a fixed-length force vector arrow to the plot.
    Length is INDEPENDENT of magnitude.
    Label shows magnitude.
    """
    # Normalize components
    mag = np.hypot(components[0], components[1])
    if mag < 1e-1: return # Zero force
    
    ux = components[0] / mag
    uy = components[1] / mag
    
    end = (start[0] + ux * fixed_length, start[1] + uy * fixed_length)
    
    # Line
    fig.add_trace(go.Scatter(
        x=[start[0], end[0]], y=[start[1], end[1]],
        mode='lines', line=dict(color=color, width=4),
        showlegend=False, hoverinfo='skip'
    ))
    
    # Arrow Head (Triangle Marker)
    arrow_angle = np.degrees(np.arctan2(uy, ux))
    fig.add_trace(go.Scatter(
        x=[end[0]], y=[end[1]],
        mode='markers',
        marker=dict(symbol="triangle-up", size=12, color=color, angle=arrow_angle-90),
        showlegend=False, hoverinfo='skip'
    ))
    
    # Label Placement (Shift along vector to avoid overlap)
    shift_dist = 30
    lx = end[0] + ux * shift_dist
    ly = end[1] + uy * shift_dist
    
    annot_text = f"<b>{label}</b> = {mag:.1f} N"
    fig.add_annotation(
        x=lx, y=ly, text=annot_text,
        showarrow=False, font=dict(color=color, size=14, family="Arial"),
        bgcolor="rgba(0,0,0,0.85)", bordercolor=color, borderwidth=1, borderpad=3
    )

def create_single_fbd(title, link_pts, link_color, forces, gravity_info, angle_info=None):
    """
    Helper to create a single Clean FBD figure.
    link_pts: list of (x,y) tuples defining the link line/shape.
    forces: list of (pos_tuple, vec_tuple, label, color).
    gravity_info: (G_pos, W_label).
    """
    fig = go.Figure()
    
    # Link Body
    x_vals = [p[0] for p in link_pts]
    y_vals = [p[1] for p in link_pts]
    
    if len(link_pts) > 2: # Polygon (e.g. Piston)
         fig.add_trace(go.Scatter(
            x=x_vals + [x_vals[0]], y=y_vals + [y_vals[0]],
            fill='toself', fillcolor=link_color,
            mode='lines', line=dict(color='white', width=2),
            showlegend=False
        ))
    else: # Line Segment
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines+markers',
            line=dict(color=link_color, width=12), # Thick link
            marker=dict(size=15, color=link_color, line=dict(width=2, color='white')),
            showlegend=False
        ))

    # Angle Arc
    if angle_info:
        center, radius, t_start, t_end, lbl = angle_info
        arc, ann = create_angle_arc(center, radius, t_start, t_end, 'white', lbl)
        fig.add_trace(arc)
        if ann: fig.add_annotation(ann)

    # Forces
    for pos, vec, lbl, col in forces:
        add_vector_arrow(fig, pos, vec, lbl, color=col, fixed_length=60)
        
    # Gravity & CG
    min_dim = min(max(x_vals)-min(x_vals) if len(x_vals)>1 else 100, 100)
    
    if gravity_info:
        G_pos, W_lbl = gravity_info
        # CG Marker (Black Dot)
        fig.add_trace(go.Scatter(
            x=[G_pos[0]], y=[G_pos[1]],
            mode='markers', marker=dict(color='black', size=8, symbol='circle'),
            showlegend=False
        ))
        # Gravity Vector (Purple)
        add_vector_arrow(fig, G_pos, (0, -1), W_lbl, color='#AA00FF', fixed_length=50)
        
        # Dimension Line (From first point to G)
        if len(link_pts) > 0:
            add_dim_line(fig, link_pts[0], G_pos, "r_G", offset=15, color="#AAAAAA")

    # Clean Layout
    fig.update_layout(
        title=dict(text=title, font=dict(color='white', size=16), x=0.5),
        template="plotly_dark",
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def draw_fbd_separate(joints, dyn_state, mech_type):
    """
    Generates 3 separate FBD figures for Crank, Coupler, Output.
    """
    if not dyn_state or not joints: 
        return go.Figure(), go.Figure(), go.Figure()

    # Colors
    C_CRANK = '#2962FF'
    C_COUPLER = '#00C853'
    C_OUTPUT = '#FF3D00'
    C_FORCE = '#FFD700' # Gold

    # Forces Extraction
    F12 = dyn_state.get('F12', (0,0)); F32 = (-dyn_state.get('F23', (0,0))[0], -dyn_state.get('F23', (0,0))[1])
    F23 = dyn_state.get('F23', (0,0)); F43 = (-dyn_state.get('F34', (0,0))[0], -dyn_state.get('F34', (0,0))[1])
    F34 = dyn_state.get('F34', (0,0)); F14 = dyn_state.get('F14', (0,0))

    # --- 1. CRANK FBD ---
    O2, A = joints["O2"], joints["A"]
    G2 = ((O2[0]+A[0])/2, (O2[1]+A[1])/2)
    theta2 = calc_angle_deg(O2, A)
    
    fig2 = create_single_fbd(
        "Krank (Uzuv 2)", [O2, A], C_CRANK,
        [ (O2, F12, "F₁₂", C_FORCE), (A, F32, "F₃₂", C_FORCE) ],
        (G2, "W₂"),
        (O2, 20, 0, theta2, f"{theta2:.0f}°")
    )
    
    # --- 2. COUPLER FBD ---
    B = joints["B"]
    G3 = ((A[0]+B[0])/2, (A[1]+B[1])/2)
    theta3 = calc_angle_deg(A, B)
    
    fig3 = create_single_fbd(
        "Biyel (Uzuv 3)", [A, B], C_COUPLER,
        [ (A, F23, "F₂₃", C_FORCE), (B, F43, "F₄₃", C_FORCE) ],
        (G3, "W₃"),
        (A, 20, 0, theta3, f"{theta3:.0f}°")
    )
    
    # --- 3. OUTPUT FBD ---
    fig4 = None
    if mech_type == "Krank-Biyel Mekanizması" and "C" in joints:
        C = joints["C"]
        # Piston box
        box = [ (C[0]-20, C[1]-12), (C[0]+20, C[1]-12), (C[0]+20, C[1]+12), (C[0]-20, C[1]+12) ]
        
        fig4 = create_single_fbd(
            "Piston (Uzuv 4)", box, C_OUTPUT,
            [ (C, F34, "F₃₄", C_FORCE), (C, F14, "F₁₄", C_FORCE) ],
            (C, "W₄"),
            None
        )
    else:
        O4 = joints["O4"]
        G4 = ((B[0]+O4[0])/2, (B[1]+O4[1])/2)
        theta4 = calc_angle_deg(O4, B)
        
        fig4 = create_single_fbd(
            "Sarkaç (Uzuv 4)", [O4, B], C_OUTPUT,
            [ (B, F34, "F₃₄", C_FORCE), (O4, F14, "F₁₄", C_FORCE) ],
            (G4, "W₄"),
            (O4, 20, 0, theta4, f"{theta4:.0f}°")
        )

    return fig2, fig3, fig4


def create_motion_arrow(center, radius, value, color, label, offset_angle=135):
    """
    Creates a curved arrow indicating rotation direction.
    value: positive for CCW, negative for CW.
    offset_angle: angle where the arrow is placed (0-360).
    """
    if abs(value) < 0.001: return [] # No motion
    
    # Direction
    is_ccw = value > 0
    
    # Arc span (e.g. 60 degrees)
    span = 60
    if is_ccw:
        start = offset_angle - span/2
        end = offset_angle + span/2
        arrow_pos_angle = end
        # Arrow points tangent value + 90
        marker_angle = (arrow_pos_angle + 90) % 360
    else:
        start = offset_angle + span/2
        end = offset_angle - span/2
        arrow_pos_angle = end
        # Arrow points tangent value - 90
        marker_angle = (arrow_pos_angle - 90) % 360
        
    # Generate Arc Points
    t = np.linspace(start, end, 15)
    x = center[0] + radius * np.cos(np.radians(t))
    y = center[1] + radius * np.sin(np.radians(t))
    
    # Line Trace
    traces = []
    traces.append(go.Scatter(
        x=x, y=y, mode='lines', 
        line=dict(color=color, width=3), 
        hoverinfo='skip', showlegend=False
    ))
    
    # Arrow Head (Triangle)
    ax = center[0] + radius * np.cos(np.radians(arrow_pos_angle))
    ay = center[1] + radius * np.sin(np.radians(arrow_pos_angle))
    
    traces.append(go.Scatter(
        x=[ax], y=[ay], mode='markers',
        marker=dict(symbol="triangle-up", size=12, color=color, angle=marker_angle-90), # -90 correction for 'triangle-up' base
        hoverinfo='skip', showlegend=False
    ))
    
    # Label
    lx = center[0] + (radius * 1.3) * np.cos(np.radians(offset_angle))
    ly = center[1] + (radius * 1.3) * np.sin(np.radians(offset_angle))
    
    traces.append(go.Scatter(
        x=[lx], y=[ly], mode='text',
        text=[f"<b>{label}</b>"],
        textfont=dict(size=14, color=color),
        hoverinfo='skip', showlegend=False
    ))
    
    return traces

def draw_mechanism(joints, L1, L2, L3, L4, omega2=0, alpha2=0, assembly_mode=1, show_labels=True, show_angles=True, show_motion=True, trace_points=None, current_cp=None, instant_centers=None):
    """
    Draws the 4-bar linkage using Plotly (Static Frame).
    trace_points: tuple (x_list, y_list) for full cycle trace
    current_cp: tuple (px, py) for current coupler point
    instant_centers: dict {'I13': (x,y), 'I24': (x,y)} to draw construction lines
    """
    if joints is None:
        fig = go.Figure()
        return fig

    O2 = joints["O2"]
    
    fig = go.Figure()

    # --- TRACE (Coupler Curve) ---
    if trace_points:
        tx, ty = trace_points
        fig.add_trace(go.Scatter(
            x=tx, y=ty,
            mode='lines',
            line=dict(color='#FFDC00', width=2, dash='dot'), # Yellow dash
            name='Yörünge',
            hoverinfo='skip'
        ))
        
    # --- CURRENT CP ---
    if current_cp:
        fig.add_trace(go.Scatter(
            x=[current_cp[0]], y=[current_cp[1]],
            mode='markers',
            marker=dict(size=10, color='#FFDC00', symbol='x'),
            name='Biyel Noktası'
        ))

    # --- INSTANT CENTERS (Kennedy's Theorem) ---
    if instant_centers:
        ic_color = "#E91E63" # Pink/Magenta for ICs
        
        # I13 Construction Lines
        I13 = instant_centers.get("I13")
        if I13:
            # Line O2 -> I13 (passes through A)
            fig.add_trace(go.Scatter(
                x=[O2[0], I13[0]], y=[O2[1], I13[1]],
                mode='lines',
                line=dict(color=ic_color, width=1, dash='dot'),
                hoverinfo='skip', showlegend=False
            ))
            # Line O4 -> I13 (passes through B)
            if "O4" in joints:
                O4 = joints["O4"]
                fig.add_trace(go.Scatter(
                    x=[O4[0], I13[0]], y=[O4[1], I13[1]],
                    mode='lines',
                    line=dict(color=ic_color, width=1, dash='dot'),
                    hoverinfo='skip', showlegend=False
                ))
            # Marker
            fig.add_trace(go.Scatter(
                x=[I13[0]], y=[I13[1]],
                mode='markers+text',
                marker=dict(size=10, color=ic_color, symbol='diamond'),
                text=["I13"], textposition="top center",
                name='Ani Dönme Merkezi (I13)'
            ))

        # I24 Construction Lines (Coupler Extension)
        I24 = instant_centers.get("I24")
        if I24:
            # Line A -> I24 (passes through B)
            if "A" in joints:
                A = joints["A"]
                fig.add_trace(go.Scatter(
                    x=[A[0], I24[0]], y=[A[1], I24[1]],
                    mode='lines',
                    line=dict(color=ic_color, width=1, dash='dot'),
                    hoverinfo='skip', showlegend=False
                ))
            # Line O2 -> I24 (Ground Extension)
            fig.add_trace(go.Scatter(
                x=[O2[0], I24[0]], y=[O2[1], I24[1]],
                mode='lines',
                line=dict(color=ic_color, width=1, dash='dot'),
                hoverinfo='skip', showlegend=False
            ))
            # Marker
            fig.add_trace(go.Scatter(
                x=[I24[0]], y=[I24[1]],
                mode='markers+text',
                marker=dict(size=10, color=ic_color, symbol='diamond'),
                text=["I24"], textposition="top center",
                name='Ani Dönme Merkezi (I24)'
            ))

    # --- INPUT MOTION ARROWS (Applicable to both types) ---
    # Will append at end if show_motion is True
    
    if "C" in joints: # Slider-Crank
        B, C = joints["B"], joints["C"]
        
        # Piston Dimensions (mm)
        piston_h = 30 
        piston_w = 60 
        
        # Ground Line Logic
        ground_y = 0 
        
        # Piston Shape Coordinates (Centered at C)
        px = [C[0] - piston_w/2, C[0] + piston_w/2, C[0] + piston_w/2, C[0] - piston_w/2, C[0] - piston_w/2]
        py = [C[1] - piston_h/2, C[1] - piston_h/2, C[1] + piston_h/2, C[1] + piston_h/2, C[1] - piston_h/2]
        
        # Ground Line Length
        gx_start = -50 
        gx_end = L2 + L3 + 50 
        
        # 0. Ground
        fig.add_trace(go.Scatter(x=[gx_start, gx_end], y=[ground_y, ground_y], mode='lines', line=dict(color='gray', width=4, dash='dash'), name='Zemin'))
        
        # 1. Piston Body
        fig.add_trace(go.Scatter(x=px, y=py, mode='lines', fill='toself', fillcolor=COLOR_PISTON, line=dict(color='black', width=2), name='Piston'))
        
        # 2. Links (Order: L2, L4, L3 - Coupler on Top)
        fig.add_trace(go.Scatter(x=[O2[0], B[0]], y=[O2[1], B[1]], mode='lines+markers', line=dict(color=COLOR_L2, width=WIDTH_LINK), marker=dict(size=WIDTH_LINK, color=COLOR_L2), name='Krank'))
        if "C" not in joints: # 4-Bar Rocker (Just in case logic flows here erroneously, but controlled by if/else)
             fig.add_trace(go.Scatter(x=[B[0], O4[0]], y=[B[1], O4[1]], mode='lines+markers', line=dict(color=COLOR_L4, width=WIDTH_LINK), marker=dict(size=WIDTH_LINK, color=COLOR_L4), name='Sarkaç'))
        
        # Coupler (L3) - Top Link
        if "C" in joints:
             fig.add_trace(go.Scatter(x=[B[0], C[0]], y=[B[1], C[1]], mode='lines+markers', line=dict(color=COLOR_L3, width=WIDTH_LINK), marker=dict(size=WIDTH_LINK, color=COLOR_L3), name='Biyel'))
        else:
             fig.add_trace(go.Scatter(x=[A[0], B[0]], y=[A[1], B[1]], mode='lines+markers', line=dict(color=COLOR_L3, width=WIDTH_LINK), marker=dict(size=WIDTH_LINK, color=COLOR_L3), name='Biyel'))

        # 3. Joints (All - Top Layer)
        if "C" in joints:
             fig.add_trace(go.Scatter(x=[O2[0], B[0], C[0]], y=[O2[1], B[1], C[1]], mode='markers', marker=dict(size=SIZE_JOINT, color=COLOR_JOINT_FILL, line=dict(width=3, color=COLOR_JOINT_LINE)), name='Mafsallar'))
        else:
             fig.add_trace(go.Scatter(x=[O2[0], A[0], B[0], O4[0]], y=[O2[1], A[1], B[1], O4[1]], mode='markers', marker=dict(size=SIZE_JOINT, color=COLOR_JOINT_FILL, line=dict(width=3, color=COLOR_JOINT_LINE)), name='Mafsallar'))
        
        # ANNOTATIONS (Slider-Crank) - Modern & Scientific
        font_style = dict(family="Arial, sans-serif", size=14, color="white")
        
        # Joints
        fig.add_annotation(x=O2[0], y=O2[1], text="<b>O₂</b>", showarrow=False, yshift=-25, font=font_style)
        fig.add_annotation(x=B[0], y=B[1], text="<b>B</b>", showarrow=False, yshift=40, font=font_style)
        fig.add_annotation(x=C[0], y=C[1], text="<b>C</b>", showarrow=False, yshift=25, font=font_style)
        
        # Links
        if show_labels:
            fig.add_annotation(x=(O2[0]+B[0])/2, y=(O2[1]+B[1])/2, text="<b>Krank</b> (L2)", showarrow=True, arrowhead=0, ax=20, ay=-20, font=dict(color=COLOR_L2, size=14, family="Arial"), bgcolor="rgba(20,20,20,0.8)", bordercolor=COLOR_L2, borderwidth=1, borderpad=4)
            fig.add_annotation(x=(B[0]+C[0])/2, y=(B[1]+C[1])/2, text="<b>Biyel</b> (L3)", showarrow=True, arrowhead=0, ax=20, ay=20, font=dict(color=COLOR_L3, size=14, family="Arial"), bgcolor="rgba(20,20,20,0.8)", bordercolor=COLOR_L3, borderwidth=1, borderpad=4)
            fig.add_annotation(x=C[0], y=C[1]-piston_h, text="<b>Piston</b>", showarrow=False, yshift=-10, font=dict(color=COLOR_PISTON, size=14, family="Arial"), bgcolor="rgba(20,20,20,0.8)", bordercolor=COLOR_PISTON, borderwidth=1, borderpad=4)

        if show_angles:
            # --- ANGLES (Slider-Crank) ---
            # Dynamic Scaled Radius (approx 20% of smallest moving link, but clamped)
            scale_ref = min(L2, L3)
            arc_radius = max(scale_ref * 0.25, 15)
            
            # 1. Theta 2 (at O2)
            theta2_deg = calc_angle_deg(O2, B)
            # Handle wraparound if needed for nicer arc (0 to theta2)
            start_t2, end_t2 = 0, theta2_deg
            if end_t2 < 0: end_t2 += 360
            
            arc_t2, ann_t2 = create_angle_arc(O2, arc_radius, 0, end_t2, COLOR_L2, "θ₂", text_size=16)
            fig.add_trace(arc_t2)
            if ann_t2: 
                 # Manual adjustment to prevent overlap if angle is small
                 if abs(end_t2) < 20: ann_t2['x'] += 10; ann_t2['y'] += 10
                 fig.add_annotation(ann_t2)

            # 2. Theta 3 (Show at C for better visual "Ground" reference)
            # Vector C->B (Connecting Rod)
            t3_radius = arc_radius * 3.0
            
            fig.add_shape(type="line", x0=C[0]-t3_radius*1.2, y0=C[1], x1=C[0]+t3_radius*0.5, y1=C[1], line=dict(color="gray", width=1, dash="dot"))
            
            angle_CB = calc_angle_deg(C, B) # Angle of Rod pointing back to Crank
            
            # Smart Arc Logic for Theta 3 (ALWAYS Acute with Horizontal)
            # 1. Normalize angle to 0-360
            ang = angle_CB
            if ang < 0: ang += 360
            
            # 2. Find Closest Horizontal Ref
            # If in Q2/Q3 (90-270), closest is 180.
            # If in Q1/Q4 (<90 or >270), closest is 0 (or 360).
            
            if 90 <= ang <= 270:
                ref = 180
                # Arc between 180 and ang
                start_t3, end_t3 = min(ang, ref), max(ang, ref)
            else:
                # Closer to 0/360
                if ang > 270: # Q4
                    ref = 360
                    # Draw from ang to 360
                    start_t3, end_t3 = ang, 360 # 360 is effectively 0
                else: # Q1
                    ref = 0
                    start_t3, end_t3 = 0, ang
                
            arc_t3, ann_t3 = create_angle_arc(C, t3_radius, start_t3, end_t3, COLOR_L3, "θ₃", fill_color="rgba(46, 204, 64, 0.1)", text_size=16)
            fig.add_trace(arc_t3)
            if ann_t3: fig.add_annotation(ann_t3)

    else: # 4-Bar OR Inverted Slider
        A, B, O4 = joints["A"], joints["B"], joints["O4"]
        
        # Ground Line
        fig.add_trace(go.Scatter(x=[O2[0], O4[0]], y=[O2[1], O4[1]], mode='lines', line=dict(color=COLOR_GROUND, width=WIDTH_GROUND), name='Zemin'))
        
        is_inverted = (L3 == 0) # Detection for Inverted Slider
        
        if is_inverted:
             # --- INVERTED SLIDER DRAWING ---
             # 1. Crank (O2-A)
             fig.add_trace(go.Scatter(x=[O2[0], A[0]], y=[O2[1], A[1]], mode='lines+markers', line=dict(color=COLOR_L2, width=WIDTH_LINK), marker=dict(size=WIDTH_LINK, color=COLOR_L2), name='Krank'))
             # 2. Rocker (O4-B) where B is tip
             fig.add_trace(go.Scatter(x=[O4[0], B[0]], y=[O4[1], B[1]], mode='lines+markers', line=dict(color=COLOR_L4, width=WIDTH_LINK), marker=dict(size=WIDTH_LINK, color=COLOR_L4), name='Kızak Kolu'))
             
             # 3. Slider Block at A
             sl_w, sl_h = 30, 20
             angle_rad = np.arctan2(B[1]-O4[1], B[0]-O4[0])
             corners = np.array([[-sl_w/2, -sl_h/2], [sl_w/2, -sl_h/2], [sl_w/2, sl_h/2], [-sl_w/2, sl_h/2], [-sl_w/2, -sl_h/2]])
             c, s = np.cos(angle_rad), np.sin(angle_rad)
             rot_corners = corners @ np.array([[c, -s], [s, c]]).T
             box_x = rot_corners[:, 0] + A[0]
             box_y = rot_corners[:, 1] + A[1]
             fig.add_trace(go.Scatter(x=box_x, y=box_y, mode='lines', fill='toself', fillcolor=COLOR_L3, line=dict(color='black', width=2), name='Kızak'))
             
             # Joints
             fig.add_trace(go.Scatter(x=[O2[0], A[0], O4[0]], y=[O2[1], A[1], O4[1]], mode='markers', marker=dict(size=SIZE_JOINT, color=COLOR_JOINT_FILL, line=dict(width=3, color=COLOR_JOINT_LINE)), name='Mafsallar'))
        
        else:
             # --- NORMAL 4-BAR ---
             # 1. Links (L2, L4, L3)
             fig.add_trace(go.Scatter(x=[O2[0], A[0]], y=[O2[1], A[1]], mode='lines+markers', line=dict(color=COLOR_L2, width=WIDTH_LINK), marker=dict(size=WIDTH_LINK, color=COLOR_L2), name='Krank'))
             fig.add_trace(go.Scatter(x=[B[0], O4[0]], y=[B[1], O4[1]], mode='lines+markers', line=dict(color=COLOR_L4, width=WIDTH_LINK), marker=dict(size=WIDTH_LINK, color=COLOR_L4), name='Sarkaç'))
             fig.add_trace(go.Scatter(x=[A[0], B[0]], y=[A[1], B[1]], mode='lines+markers', line=dict(color=COLOR_L3, width=WIDTH_LINK), marker=dict(size=WIDTH_LINK, color=COLOR_L3), name='Biyel'))
             # 2. Joints
             fig.add_trace(go.Scatter(x=[O2[0], A[0], B[0], O4[0]], y=[O2[1], A[1], B[1], O4[1]], mode='markers', marker=dict(size=SIZE_JOINT, color=COLOR_JOINT_FILL, line=dict(width=3, color=COLOR_JOINT_LINE)), name='Mafsallar'))

        # ANNOTATIONS (4-Bar) - Modern & Scientific
        font_style = dict(family="Arial, sans-serif", size=14, color="white")
        
        # B Label Logic
        yshift_B = 25 if assembly_mode == -1 else -40
        
        fig.add_annotation(x=O2[0], y=O2[1], text="<b>O₂</b>", showarrow=False, yshift=-25, font=font_style)
        fig.add_annotation(x=A[0], y=A[1], text="<b>A</b>", showarrow=False, yshift=25, font=font_style)
        fig.add_annotation(x=B[0], y=B[1], text="<b>B</b>", showarrow=False, yshift=yshift_B, font=font_style)
        fig.add_annotation(x=O4[0], y=O4[1], text="<b>O₄</b>", showarrow=False, yshift=-25, font=font_style)

        # Link Labels
        if show_labels:
            fig.add_annotation(x=(O2[0]+A[0])/2, y=(O2[1]+A[1])/2, text="<b>Krank</b> (L2)", showarrow=True, arrowhead=0, ax=-30, ay=-30, font=dict(color=COLOR_L2, size=14, family="Arial"), bgcolor="rgba(20,20,20,0.8)", bordercolor=COLOR_L2, borderwidth=1, borderpad=4)
            fig.add_annotation(x=(A[0]+B[0])/2, y=(A[1]+B[1])/2, text="<b>Biyel</b> (L3)", showarrow=True, arrowhead=0, ax=0, ay=-40, font=dict(color=COLOR_L3, size=14, family="Arial"), bgcolor="rgba(20,20,20,0.8)", bordercolor=COLOR_L3, borderwidth=1, borderpad=4)
            fig.add_annotation(x=(B[0]+O4[0])/2, y=(B[1]+O4[1])/2, text="<b>Sarkaç</b> (L4)", showarrow=True, arrowhead=0, ax=30, ay=-30, font=dict(color=COLOR_L4, size=14, family="Arial"), bgcolor="rgba(20,20,20,0.8)", bordercolor=COLOR_L4, borderwidth=1, borderpad=4)

        if show_angles:
            # --- ANGLES (4-Bar) ---
            # Dynamic Radius
            scale_ref = min(L2, L3, L4)
            arc_radius = max(scale_ref * 0.25, 15)
            
            # 1. Theta 2 (At O2)
            theta2_deg = calc_angle_deg(O2, A)
            start_t2, end_t2 = 0, theta2_deg
            if end_t2 < 0: end_t2 += 360
            
            arc_t2, ann_t2 = create_angle_arc(O2, arc_radius, 0, end_t2, COLOR_L2, "θ₂")
            fig.add_trace(arc_t2)
            if ann_t2: fig.add_annotation(ann_t2)
            
            # 2. Theta 4 (At O4)
            theta4_deg = calc_angle_deg(O4, B)
            # Measured from Horizontal (0)
            start_t4, end_t4 = 0, theta4_deg
            if end_t4 < 0: end_t4 += 360
            
            arc_t4, ann_t4 = create_angle_arc(O4, arc_radius, 0, end_t4, COLOR_L4, "θ₄")
            fig.add_trace(arc_t4)
            if ann_t4: fig.add_annotation(ann_t4)
            
            # 3. Transmission Angle (Mu) at B
            mu_radius = arc_radius * 0.8 # Slightly smaller
            ang_BA = calc_angle_deg(B, A)
            ang_BO4 = calc_angle_deg(B, O4) # Vector B->O4
            
            # Calculate shortest arc
            diff = (ang_BA - ang_BO4) % 360
            if diff <= 180:
                 start_mu, end_mu = ang_BO4, ang_BA
            else:
                 start_mu, end_mu = ang_BA, ang_BO4
                 
            arc_mu, ann_mu = create_angle_arc(B, mu_radius, start_mu, end_mu, "#E040FB", "μ", fill_color="rgba(224, 64, 251, 0.2)")
            fig.add_trace(arc_mu)
            if ann_t4: fig.add_annotation(ann_mu)

    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False, showticklabels=True),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False, showticklabels=True),
        width=700, height=500,
        margin=dict(l=20, r=20, t=10, b=20), 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # --- DRAW MOTION ARROWS ---
    if show_motion:
        # Calculate Theta 2 for alignment
        # End point of crank is A (4-bar) or B (Slider)
        crank_end = joints.get("A", joints.get("B"))
        theta2_current = calc_angle_deg(O2, crank_end)
        
        # Radius depends on L2
        motion_radius_w = L2 * 0.5 if L2 else 30
        motion_radius_a = L2 * 0.75 if L2 else 50
        
        # Omega2 (Blue/Cyan) - Position relative to Crank
        # We place it "along" the crank at an offset so it moves with it
        traces_w = create_motion_arrow(O2, motion_radius_w, omega2, "#00E5FF", "ω₂", offset_angle=theta2_current + 45)
        for t in traces_w: fig.add_trace(t)
        
        # Alpha2 (Red/Pink)
        traces_a = create_motion_arrow(O2, motion_radius_a, alpha2, "#FF4081", "α₂", offset_angle=theta2_current + 45)
        for t in traces_a: fig.add_trace(t)

    return fig

def create_animation_figure(mechanism, theta_start, L1, L2, L3, L4, omega2=1.0, alpha2=0, assembly_mode=1):
    
    # 1. Visual Steps
    direction = 1 if omega2 >= 0 else -1
    slider_steps_val = list(range(0, 361, 2)) 
    
    valid_frames_data = [] 
    all_x, all_y = [], []
    
    piston_h, piston_w = 30, 60

    for step in slider_steps_val:
        angle_delta = step * direction
        mech_angle = theta_start + angle_delta
        
        joints = None
        if hasattr(mechanism, 'get_position'):
            joints = mechanism.get_position(mech_angle, assembly_mode=assembly_mode)
            
            if joints:
                w3, w4 = mechanism.get_velocity(mech_angle, omega2, assembly_mode)
                a3, a4 = mechanism.get_acceleration(mech_angle, omega2, alpha2, assembly_mode)
                
                if "C" in joints: # Slider Crank
                   v_p = w4 
                   a_p = a4
                   kin = mechanism.calculate_kinematics(mech_angle, omega2, alpha2, assembly_mode)
                   theta3_val = kin.get('theta3', 0)
                   
                   p_col_m = ["Krank Açısı (θ₂)", "Biyel Açısı (θ₃)", "Biyel Hızı (ω₃)", "Biyel İvmesi (α₃)", "Piston Hızı (Vₚ)", "Piston İvmesi (aₚ)"]
                   v_col_m = [f"{mech_angle%360:.2f}", f"{abs(theta3_val):.2f}", f"{abs(w3):.2f}", f"{abs(a3):.2f}", f"{abs(v_p):.2f}", f"{abs(a_p):.2f}"]
                   u_col_m = ["°", "°", "rad/s", "rad/s²", "mm/s", "mm/s²"]

                   p_col_c = ["B (x, y)", "C (x, y)"]
                   v_col_c = [f"({joints['B'][0]:.2f}, {joints['B'][1]:.2f})", f"({joints['C'][0]:.2f}, {joints['C'][1]:.2f})"]
                   u_col_c = ["mm", "mm"]
                   
                else: # Four Bar
                   theta4_val = joints.get('theta4', 0)
                   mu_val = mechanism.calculate_transmission_angle(mech_angle, assembly_mode)
                   
                   p_col_m = ["Krank Açısı (θ₂)", "Sarkaç Açısı (θ₄)", "Bağlama Açısı (μ)", "Biyel Hızı (ω₃)", "Biyel İvmesi (α₃)", "Sarkaç Hızı (ω₄)", "Sarkaç İvmesi (α₄)"]
                   v_col_m = [f"{mech_angle%360:.2f}", f"{theta4_val:.2f}", f"{mu_val:.2f}", f"{abs(w3):.2f}", f"{abs(a3):.2f}", f"{abs(w4):.2f}", f"{abs(a4):.2f}"]
                   u_col_m = ["°", "°", "°", "rad/s", "rad/s²", "rad/s", "rad/s²"]

                   p_col_c = ["A (x, y)", "B (x, y)"]
                   v_col_c = [f"({joints['A'][0]:.2f}, {joints['A'][1]:.2f})", f"({joints['B'][0]:.2f}, {joints['B'][1]:.2f})"]
                   u_col_c = ["mm", "mm"]
            
            if joints:
                valid_frames_data.append((step, joints, mech_angle, [p_col_m, v_col_m, u_col_m], [p_col_c, v_col_c, u_col_c]))
            
            O2 = joints["O2"]
            if "C" in joints: 
                B, C = joints["B"], joints["C"]
                all_x.extend([O2[0], B[0], C[0]-piston_w/2, C[0]+piston_w/2])
                all_y.extend([O2[1], B[1], C[1]-piston_h/2, C[1]+piston_h/2])
            else: 
                A, B, O4 = joints["A"], joints["B"], joints["O4"]
                all_x.extend([O2[0], A[0], B[0], O4[0]])
                all_y.extend([O2[1], A[1], B[1], O4[1]])


    if not all_x: return go.Figure()
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    span = max(max_x - min_x, max_y - min_y) * 1.5
    cx, cy = (min_x + max_x)/2, (min_y + max_y)/2
    fixed_x = [cx - span/2, cx + span/2]
    fixed_y = [cy - span/2, cy + span/2]
    
    is_slider = ("C" in valid_frames_data[0][1]) if valid_frames_data else False
    
    initial_joints = valid_frames_data[0][1]
    initial_main_data = valid_frames_data[0][3]
    initial_coord_data = valid_frames_data[0][4]
    O2 = initial_joints["O2"]

    # Ground (Static)
    gx, gy = [], []
    ground_kwargs = dict(color=COLOR_GROUND, width=WIDTH_GROUND)
    
    if is_slider:
        gx = [-50, L2 + L3 + 50]; gy = [0, 0]
        ground_kwargs = dict(color='gray', width=4, dash='dash')
    else:
        gx = [O2[0], initial_joints["O4"][0]]; gy = [O2[1], initial_joints["O4"][1]]

    fig = make_subplots(
        rows=3, cols=2, 
        column_widths=[0.65, 0.35],
        row_heights=[0.04, 0.56, 0.40],
        specs=[[{"type": "xy", "rowspan": 3}, None], 
               [None,                         {"type": "table"}],
               [None,                         {"type": "table"}]],
        horizontal_spacing=0.02,
        vertical_spacing=0.10,
        subplot_titles=("", "TABLO 1 : Anlık Kinematik Büyüklükler", "TABLO 2 : Mafsalların Koordinatları")
    )
    fig.update_annotations(font=dict(family="Segoe UI, Arial, sans-serif", size=22, color="#ECEFF1"))

    fig.add_trace(go.Scatter(x=gx, y=gy, mode='lines', line=dict(color=ground_kwargs.get('color', COLOR_GROUND), width=ground_kwargs.get('width', 4), dash=ground_kwargs.get('dash')), name='Zemin', hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', marker=dict(size=SIZE_JOINT, color=COLOR_JOINT_FILL, line=dict(width=3, color=COLOR_JOINT_LINE)), name='Sabit'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', line=dict(color=COLOR_L2, width=WIDTH_LINK), marker=dict(size=WIDTH_LINK, color=COLOR_L2), name='Krank'), row=1, col=1)
    
    if is_slider:
         fig.add_trace(go.Scatter(x=[], y=[], mode='lines', fill='toself', fillcolor=COLOR_PISTON, line=dict(color='black', width=2), name='Piston'), row=1, col=1)
    else:
         fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', line=dict(color=COLOR_L4, width=WIDTH_LINK), marker=dict(size=WIDTH_LINK, color=COLOR_L4), name='Sarkaç'), row=1, col=1)

    fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', line=dict(color=COLOR_L3, width=WIDTH_LINK), marker=dict(size=WIDTH_LINK, color=COLOR_L3), name='Biyel'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', marker=dict(size=SIZE_JOINT, color=COLOR_JOINT_FILL, line=dict(width=3, color=COLOR_JOINT_LINE)), name='Hareketli'), row=1, col=1)

    fig.add_trace(go.Table(
            columnwidth=[140, 70, 50],
            header=dict(values=["<b>PARAMETRE</b>", "<b>DEĞER</b>", "<b>BİRİM</b>"], fill_color='#FFD740', align='left', font=dict(color='#263238', size=18, family="Segoe UI"), height=36),
            cells=dict(values=initial_main_data, fill_color='#263238', align='left', font=dict(color='white', size=17, family="Segoe UI"), height=34, line_color='#37474F')
        ), row=2, col=2)

    fig.add_trace(go.Table(
            columnwidth=[100, 120, 50],
            header=dict(values=["<b>MAFSAL</b>", "<b>KONUM (x,y)</b>", "<b>BİRİM</b>"], fill_color='#CFD8DC', align='left', font=dict(color='#263238', size=18, family="Segoe UI"), height=36),
            cells=dict(values=initial_coord_data, fill_color='#37474F', align='left', font=dict(color='white', size=17, family="Segoe UI"), height=34, line_color='#546E7A')
        ), row=3, col=2)

    plotly_frames = []
    
    for i, (offset, joints, true_angle, m_data, c_data) in enumerate(valid_frames_data):
        d_fixed, d_moving = {}, {}
        d_l2, d_l3, d_l4 = {}, {}, {}
        
        O2 = joints["O2"]
        if is_slider:
            B, C = joints["B"], joints["C"]
            d_fixed = dict(x=[O2[0]], y=[O2[1]])
            d_moving = dict(x=[B[0], C[0]], y=[B[1], C[1]])
            d_l2 = dict(x=[O2[0], B[0]], y=[O2[1], B[1]])
            d_l3 = dict(x=[B[0], C[0]], y=[B[1], C[1]])
            px = [C[0] - piston_w/2, C[0] + piston_w/2, C[0] + piston_w/2, C[0] - piston_w/2, C[0] - piston_w/2]
            py = [C[1] - piston_h/2, C[1] - piston_h/2, C[1] + piston_h/2, C[1] + piston_h/2, C[1] - piston_h/2]
            d_l4 = dict(x=px, y=py)
        else:
            A, B, O4 = joints["A"], joints["B"], joints["O4"]
            d_fixed = dict(x=[O2[0], O4[0]], y=[O2[1], O4[1]])
            d_moving = dict(x=[A[0], B[0]], y=[A[1], B[1]])
            d_l2 = dict(x=[O2[0], A[0]], y=[O2[1], A[1]])
            d_l3 = dict(x=[A[0], B[0]], y=[A[1], B[1]])
            d_l4 = dict(x=[B[0], O4[0]], y=[B[1], O4[1]])
            
        d_tbl_main = dict(type='table', cells=dict(values=m_data))
        d_tbl_coord = dict(type='table', cells=dict(values=c_data))
        
        plotly_frames.append(go.Frame(data=[d_fixed, d_l2, d_l4, d_l3, d_moving, d_tbl_main, d_tbl_coord], traces=[1, 2, 3, 4, 5, 6, 7], name=str(offset)))

    fig.frames = plotly_frames

    slider_steps = []
    for f in plotly_frames:
        slider_steps.append(dict(method='animate', args=[[f.name], dict(mode='immediate', frame=dict(duration=20, redraw=True), transition=dict(duration=0))], label=""))

    fig.update_layout(
        template="plotly_dark",
        width=1200, height=800,
        margin=dict(l=20, r=20, t=60, b=50), 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.3), 
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=fixed_x, scaleanchor="y", scaleratio=1, showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False, showticklabels=True),
        yaxis=dict(range=fixed_y, showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False, showticklabels=True),
        updatemenus=[dict(type="buttons", showactive=False, bgcolor="rgba(0,0,0,0.5)", y=-0.1, x=0.0, xanchor="left", yanchor="top", direction="left", pad=dict(r=10, t=0), buttons=[dict(label="▶", method="animate", args=[None, dict(frame=dict(duration=20, redraw=True), fromcurrent=True, mode="immediate")]), dict(label="⏸", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))])])],
        sliders=[dict(steps=slider_steps, active=0, currentvalue={"visible": False}, pad=dict(t=0, b=0), bgcolor="rgba(0,0,0,0.5)", font=dict(color="white"), len=0.5, x=0.1, y=-0.1, minorticklen=0, ticklen=0)]
    )

    if plotly_frames:
        first_data = plotly_frames[0].data
        for k in range(1, 6): fig.data[k].update(first_data[k-1]) # fixed, l2, l4, l3, mov. 5 traces. Indices 1-5 in fig.data match 0-4 in frame data.

    return fig


def plot_kinematic_curves(theta2_range, omega3_list, omega4_list, alpha3_list, alpha4_list, ma_list=None, mu_list=None, mech_type="Dört Çubuk Mekanizması"):
    # 1. Determine Labels & Units
    lbl_w3 = "Biyel Açısal Hızı (ω₃)"
    lbl_a3 = "Biyel Açısal İvmesi (α₃)"
    
    if mech_type == "Dört Çubuk Mekanizması":
        lbl_v_out = "Çıkış Açısal Hızı (ω₄)"
        lbl_a_out = "Çıkış Açısal İvmesi (α₄)"
        unit_v = "[rad/s]"
        unit_a = "[rad/s²]"
    elif mech_type == "Krank-Biyel Mekanizması":
        lbl_v_out = "Piston Hızı (Vₚ)"
        lbl_a_out = "Piston İvmesi (aₚ)"
        unit_v = "[mm/s]"
        unit_a = "[mm/s²]"
    else: # Kol-Kızak
        lbl_v_out = "Çıkış Açısal Hızı (ω₄)"
        lbl_a_out = "Çıkış Açısal İvmesi (α₄)"
        unit_v = "[rad/s]"
        unit_a = "[rad/s²]"

    layout_props = dict(
        template="plotly_dark",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(title="Giriş Açısı θ₂ [Derece]", showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # FIGURE 1: VELOCITIES
    fig_v = go.Figure()
    if omega3_list is not None and len(omega3_list) > 0 and not all(x is None for x in omega3_list):
        fig_v.add_trace(go.Scatter(x=theta2_range, y=omega3_list, mode='lines', name=f"{lbl_w3}", line=dict(color='#00CC96')))
    fig_v.add_trace(go.Scatter(x=theta2_range, y=omega4_list, mode='lines', name=f"{lbl_v_out}", line=dict(color='#EF553B')))
    fig_v.update_layout(title=f"Hız Analizi {unit_v}", **layout_props)
    fig_v.update_yaxes(title=f"Hız {unit_v}", showgrid=True, gridcolor='rgba(255,255,255,0.1)')

    # FIGURE 2: ACCELERATIONS
    fig_a = go.Figure()
    if alpha3_list is not None and len(alpha3_list) > 0 and not all(x is None for x in alpha3_list):
        fig_a.add_trace(go.Scatter(x=theta2_range, y=alpha3_list, mode='lines', name=f"{lbl_a3}", line=dict(color='#AB63FA')))
    fig_a.add_trace(go.Scatter(x=theta2_range, y=alpha4_list, mode='lines', name=f"{lbl_a_out}", line=dict(color='#FFA15A')))
    fig_a.update_layout(title=f"İvme Analizi {unit_a}", **layout_props)
    fig_a.update_yaxes(title=f"İvme {unit_a}", showgrid=True, gridcolor='rgba(255,255,255,0.1)')

    # FIGURE 3: SYSTEM PROPERTIES (MA & Mu)
    fig_p = go.Figure()
    # Left Axis: Transmission Angle (Mu)
    if mu_list is not None and len(mu_list) > 0 and not all(x is None for x in mu_list):
        fig_p.add_trace(go.Scatter(x=theta2_range, y=mu_list, mode='lines', name="Bağlama Açısı (μ)", line=dict(color='cyan')))
    
    # Right Axis: Mechanical Advantage (MA)
    if ma_list is not None and len(ma_list) > 0 and not all(x is None for x in ma_list):
        fig_p.add_trace(go.Scatter(x=theta2_range, y=ma_list, mode='lines', name="Mekanik Avantaj (MA)", line=dict(color='yellow'), yaxis="y2"))

    fig_p.update_layout(
        title="Sistem Özellikleri (μ & MA)",
        yaxis=dict(title="Bağlama Açısı [Derece]", side="left", showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[0, 180]),
        yaxis2=dict(title="Mekanik Avantaj (Boyutsuz)", side="right", overlaying="y", showgrid=False),
        **layout_props
    )

    return fig_v, fig_a, fig_p


def plot_transmission_angle(theta2_range, mu_list):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theta2_range, y=mu_list, name="Mu", line=dict(color='#ab47bc', width=3)))
    fig.update_layout(title="<b>Bağlama Açısı</b>", template="plotly_dark")
    return fig

def plot_sn_curve(Sut, Se, a, b, operating_sigma, operating_cycles, material_name):
    """
    Plots the S-N (Wöhler) Curve using pre-calculated parameters.
    """
    import plotly.graph_objects as go
    import numpy as np 
    
    fig = go.Figure()
    
    # 1. Generate S-N Line (10^3 to 10^7)
    N_range = np.logspace(3, 7, 100)
    S_range = a * (N_range ** b)
    
    # Clip S range to not go below Se (theoretically constant after 10^6)
    S_curve = []
    for n_val, s_val in zip(N_range, S_range):
        if s_val < Se:
            S_curve.append(Se)
        else:
            S_curve.append(s_val)
            
    # Main Curve
    fig.add_trace(go.Scatter(
        x=N_range, y=S_curve,
        mode='lines',
        name=f'{material_name} Limit',
        line=dict(color='#00E676', width=3)
    ))
    
    # 2. Operating Point
    # Handle Infinite Life for plotting
    plot_cycles = operating_cycles
    if operating_cycles == float('inf') or operating_cycles > 1e7:
        plot_cycles = 1e7 # Cap at graph end
        point_text = "Sonsuz Ömür"
    else:
        point_text = f"Ömür: {int(operating_cycles):,} Çevrim"
        
    fig.add_trace(go.Scatter(
        x=[plot_cycles], y=[operating_sigma],
        mode='markers+text',
        marker=dict(color='#FF1744', size=15, symbol='x'),
        text=[point_text], textposition="top right",
        name='Çalışma Noktası'
    ))
    
    # 3. Limit Lines
    fig.add_hline(y=Se, line_dash="dash", line_color="gray", annotation_text="Sürekli Mukavemet (Se)")
    fig.add_hline(y=Sut, line_dash="dash", line_color="#FF9100", annotation_text="Çekme Dayanımı (Sut)")
    
    fig.update_layout(
        title="Yorulma Analizi (S-N Diyagramı)",
        xaxis_title="Ömür (Çevrim) - Logaritmik",
        yaxis_title="Gerilme Genliği (MPa)",
        xaxis_type="log",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    return fig
