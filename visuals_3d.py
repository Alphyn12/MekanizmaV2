import plotly.graph_objects as go
import numpy as np

# --- 1. MATERIALS (Matte & Dark Mode) ---
MATERIALS = {
    'Crank': {
        'color': '#34495E', # Matte Dark Blue/Grey
        'specular': 0.0, 'roughness': 1.0, 'ambient': 0.9, 'diffuse': 0.5
    },
    'Coupler': {
        'color': '#C0392B', # Matte Deep Red
        'specular': 0.0, 'roughness': 1.0, 'ambient': 0.9, 'diffuse': 0.5
    },
    'Rocker': {
        'color': '#27AE60', # Matte Green
        'specular': 0.0, 'roughness': 1.0, 'ambient': 0.9, 'diffuse': 0.5
    },
    'Ground': {
        'color': '#505050', # Matte Grey
        'specular': 0.0, 'roughness': 1.0, 'ambient': 0.8
    },
    'Pin': {
        'color': '#BDC3C7', # Matte Silver (No shine)
        'specular': 0.1, 'roughness': 0.9, 'ambient': 0.9, 'diffuse': 0.5
    }
}

def create_extruded_polygon(x_profile, y_profile, z_min, z_max, color, name="Part", mat_props=None):
    """
    Generic function to extrude a 2D 8-point profile into a 3D prism.
    """
    x = np.tile(x_profile, 2) 
    y = np.tile(y_profile, 2)
    z = np.concatenate([np.full(len(x_profile), z_min), np.full(len(x_profile), z_max)])
    
    i, j, k = [], [], []
    N = len(x_profile)
    
    # Side Walls
    for idx in range(N):
        nxt = (idx + 1) % N
        b_c, b_n = idx, nxt
        t_c, t_n = idx + N, nxt + N
        i.extend([b_c, b_n])
        j.extend([b_n, t_n])
        k.extend([t_c, t_c])
        
    # Bottom (CW)
    c_b = 0
    for idx in range(1, N-1):
        i.append(c_b); j.append(idx+1); k.append(idx)
        
    # Top (CCW)
    c_t = N
    for idx in range(1, N-1):
        i.append(c_t); j.append(idx + N); k.append(idx + 1 + N)
        
    default_mat = MATERIALS['Ground']
    if mat_props is None: mat_props = default_mat
        
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, name=name,
        flatshading=True,
        lighting=dict(
            ambient=mat_props.get('ambient', 0.5),
            diffuse=mat_props.get('diffuse', 0.6),
            specular=mat_props.get('specular', 0.5),
            roughness=mat_props.get('roughness', 0.5)
        )
    )

def create_cylinder_mesh(x_c, y_c, z_min, z_max, radius, color, name="Pin", mat_props=None):
    res = 20
    theta = np.linspace(0, 2*np.pi, res)
    x_prof = radius * np.cos(theta) + x_c
    y_prof = radius * np.sin(theta) + y_c
    return create_extruded_polygon(x_prof, y_prof, z_min, z_max, color, name, mat_props)

def create_bar(P1, P2, z_start, z_end, width, material_key, name="Link"):
    P1 = np.array(P1)
    P2 = np.array(P2)
    V = P2 - P1
    L = np.linalg.norm(V)
    if L < 1e-6: return []
    
    mat = MATERIALS.get(material_key, MATERIALS['Ground'])
    traces = []
    
    N = np.array([-V[1], V[0]])
    N = N / np.linalg.norm(N)
    w = width / 2.0
    
    c1 = P1 + N * w
    c2 = P1 - N * w
    c3 = P2 - N * w
    c4 = P2 + N * w
    
    x_prof = [c1[0], c2[0], c3[0], c4[0]]
    y_prof = [c1[1], c2[1], c3[1], c4[1]]
    
    traces.append(create_extruded_polygon(x_prof, y_prof, z_start, z_end, mat['color'], name, mat))
    traces.append(create_cylinder_mesh(P1[0], P1[1], z_start, z_end, w, mat['color'], name+"_Joint1", mat))
    traces.append(create_cylinder_mesh(P2[0], P2[1], z_start, z_end, w, mat['color'], name+"_Joint2", mat))
    
    return traces

def create_pin_hardware(x, y, z_min, z_max, radius):
    mat = MATERIALS['Pin']
    return create_cylinder_mesh(x, y, z_min, z_max, radius, mat['color'], "Pin", mat)

def draw_mechanism_3d(joints, mech_type, show_pins=True, show_grid=True, thickness=5.0, camera_view="ISO"):
    fig = go.Figure()
    
    w_link = thickness * 2.5
    pin_rad = thickness * 0.6
    
    z_L1_bot = 0
    z_L1_top = thickness
    z_L2_bot = thickness
    z_L2_top = thickness * 2
    z_pin_bot = -5
    z_pin_top = z_L2_top + 5
    
    traces = []
    
    if mech_type == "Dört Çubuk Mekanizması":
        if 'O2' in joints:
            O2, A, B, O4 = joints['O2'], joints['A'], joints['B'], joints['O4']
            traces.extend(create_bar(O2, A, z_L1_bot, z_L1_top, w_link, 'Crank', "Krank L2"))
            traces.extend(create_bar(O4, B, z_L1_bot, z_L1_top, w_link, 'Rocker', "Sarkaç L4"))
            traces.extend(create_bar(O2, O4, -thickness/2, 0, w_link, 'Ground', "Zemin L1"))
            traces.extend(create_bar(A, B, z_L2_bot, z_L2_top, w_link, 'Coupler', "Biyel L3"))
            if show_pins:
                for p in [O2, A, B, O4]:
                    traces.append(create_pin_hardware(p[0], p[1], z_pin_bot, z_pin_top, pin_rad))

    elif mech_type == "Krank-Biyel Mekanizması":
         if 'O2' in joints:
            O2, B, C = joints['O2'], joints['B'], joints['C']
            traces.extend(create_bar(O2, B, z_L1_bot, z_L1_top, w_link, 'Crank', "Krank"))
            traces.extend(create_bar(B, C, z_L2_bot, z_L2_top, w_link, 'Coupler', "Biyel"))
            sz = w_link * 1.5
            p_start = np.array(C) - np.array([sz/2, 0])
            p_end   = np.array(C) + np.array([sz/2, 0])
            traces.extend(create_bar(p_start, p_end, 0, thickness, sz, 'Crank', "Piston"))
            if show_pins:
                for p in [O2, B, C]:
                    traces.append(create_pin_hardware(p[0], p[1], z_pin_bot, z_pin_top, pin_rad))

    elif mech_type == "Kol-Kızak (Whitworth) Mekanizması":
         if 'O2' in joints:
            O2, O4, A, B = joints['O2'], joints['O4'], joints['A'], joints['B']
            traces.extend(create_bar(O2, A, z_L1_bot, z_L1_top, w_link, 'Crank', "Krank"))
            traces.extend(create_bar(O2, O4, -thickness/2, 0, w_link, 'Ground', "Zemin"))
            traces.extend(create_bar(O4, B, z_L2_bot, z_L2_top, w_link, 'Rocker', "Kol"))
            if show_pins:
                for p in [O2, O4, A]:
                    traces.append(create_pin_hardware(p[0], p[1], z_pin_bot, z_pin_top, pin_rad))

    for t in traces:
        if t: fig.add_trace(t)

    # --- CAMERAS ---
    cameras = {
        "ISO": dict(eye=dict(x=0.8, y=-1.5, z=0.8), up=dict(x=0, y=0, z=1)),
        "ON":  dict(eye=dict(x=0, y=0, z=2.5), up=dict(x=0, y=1, z=0)),
        "UST": dict(eye=dict(x=0, y=-2.5, z=0.1), up=dict(x=0, y=0, z=1)),
        "YAN": dict(eye=dict(x=2.5, y=0, z=0), up=dict(x=0, y=0, z=1))
    }
    sel_cam = cameras.get(camera_view, cameras["ISO"])

    # --- PERFECT GRID ROOM LOGIC (REVERTED TO STEP 118 LOGIC) ---
    
    GRID_STEP = 20 # 20mm grid spacing
    
    # Explicit Ranges defined in prompt
    X_RANGE = [-50, 350] 
    Y_RANGE = [-100, 200]
    Z_RANGE = [-20, 100]  
    
    grid_style = dict(
        showbackground=True,
        backgroundcolor='#0E1117',
        gridcolor='#333333',
        gridwidth=1,
        zerolinecolor='#555555',
        zerolinewidth=2,
        visible=show_grid,
        dtick=GRID_STEP, # FORCE SYNCHRONIZED TICKS
        range=None,      # Will be set per axis
        showticklabels=True,
        tickfont=dict(color='#666666', size=10),
        title=''
    )
    
    # Clone and set ranges
    x_ax = grid_style.copy(); x_ax['range'] = X_RANGE
    y_ax = grid_style.copy(); y_ax['range'] = Y_RANGE
    z_ax = grid_style.copy(); z_ax['range'] = Z_RANGE
    
    fig.update_layout(
        scene=dict(
            xaxis=x_ax,
            yaxis=y_ax,
            zaxis=z_ax,
            bgcolor='#0E1117',
            aspectmode='data', # Keeps the 1:1:1 ratio
            camera=dict(up=sel_cam['up'], center=dict(x=0, y=0, z=0), eye=sel_cam['eye'])
        ),
        paper_bgcolor='#0E1117',
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    
    return fig
