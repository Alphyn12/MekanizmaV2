import streamlit as st
import numpy as np
from kinematics import FourBarLinkage, SliderCrankMechanism, InvertedSliderCrankMechanism, analyze_cycle, calculate_full_cycle
from visuals import draw_mechanism, create_animation_figure, plot_kinematic_curves, plot_transmission_angle, draw_fbd_separate, plot_sn_curve
from visuals_3d import draw_mechanism_3d
from report_generator import create_pdf, generate_matlab_code, generate_excel_report
import pandas as pd
from solvers.fourbar_solver import FourBarSolver
import stress
import exporters
import fatigue
import materials
import arduino_gen
from solvers.slidercrank_solver import SliderCrankSolver
import itertools
# ... existing imports ...

# ... existing imports ...

# ... existing imports ...

from solvers.slidercrank_solver import SliderCrankSolver
import itertools

st.set_page_config(page_title="Mekanizma Analiz Platformu", layout="wide")

# --- CUSTOM CSS FOR METRIC STYLING ---
st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: #1E1E1E;
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #333;
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("Mekanizma Analiz Platformu")

# --- SIDEBAR (INPUTS) ---
st.sidebar.header("ANALİZ SENARYOSU")

SCENARIOS = {
    "Özel (Manuel Giriş)": {},
    "Örnek 1 (Hız Analizi - 4 Çubuk)": {
        "type": "Dört Çubuk Mekanizması",
        "L1": 100.0, "L2": 50.0, "L3": 151.0, "L4": 111.0, 
        "theta2": 120.0, "omega2": 95.0, "alpha2": 0.0, "mode": "Çapraz"
    },
    "Örnek 7 (Dinamik Analiz - 4 Çubuk)": {
        "type": "Dört Çubuk Mekanizması",
        "L1": 100.0, "L2": 50.0, "L3": 151.0, "L4": 111.0, 
        "theta2": 120.0, "omega2": 95.0, "alpha2": 0.0, "mode": "Çapraz", "enable_dyn": True,
        "m3": 0.5, "J3": 0.00121, "BG3": 75.5, "m4": 0.4, "J4": 0.00091, "O4G4": 55.5
    },
    "Örnek 2 (Hız Analizi - Krank Biyel)": {
        "type": "Krank-Biyel Mekanizması",
        "r": 51.0, "l": 200.0, "e": 0.0, "theta2": 60.0, "omega2": 314.0, "alpha2": 0.0
    },
    "Örnek 9 (Dinamik Analiz - Krank Biyel)": {
        "type": "Krank-Biyel Mekanizması",
        "r": 51.0, "l": 200.0, "e": 0.0, "theta2": 60.0, "omega2": 314.0, "alpha2": 0.0,
        "enable_dyn": True, "m3": 1.36, "J3": 0.0102, "BG3": 51.0, "m4": 0.91, "P_gas": -6300.0
    }
}

selected_scenario = st.sidebar.selectbox("Senaryo Yükle", list(SCENARIOS.keys()))

def get_default(key, fallback):
    if selected_scenario != "Özel (Manuel Giriş)" and key in SCENARIOS[selected_scenario]: return SCENARIOS[selected_scenario][key]
    return fallback

mech_type = st.sidebar.selectbox("Mekanizma Tipi", ["Dört Çubuk Mekanizması", "Krank-Biyel Mekanizması", "Kol-Kızak (Whitworth) Mekanizması"], index=0 if get_default("type", "Dört Çubuk Mekanizması") == "Dört Çubuk Mekanizması" else (1 if get_default("type", "Dört Çubuk Mekanizması") == "Krank-Biyel Mekanizması" else 2))

assembly_mode_val = 1
if mech_type == "Dört Çubuk Mekanizması":
    default_conf = get_default("mode", "Çapraz")
    config_label = st.sidebar.selectbox("Montaj Konfigürasyonu", ["Açık Konfigürasyon", "Çapraz Konfigürasyon"], index=1 if default_conf == "Çapraz" else 0)
    assembly_mode_val = -1 if config_label == "Çapraz Konfigürasyon" else 1

if mech_type == "Dört Çubuk Mekanizması":
    L1 = st.sidebar.number_input("L1 (Sabit)", value=get_default("L1", 100.0))
    L2 = st.sidebar.number_input("L2 (Krank)", value=get_default("L2", 40.0))
    L3 = st.sidebar.number_input("L3 (Biyel)", value=get_default("L3", 80.0))
    L4 = st.sidebar.number_input("L4 (Sarkaç)", value=get_default("L4", 70.0))
    mechanism = FourBarLinkage(L1, L2, L3, L4)
    solver = FourBarSolver(L1, L2, L3, L4, assembly_mode=assembly_mode_val)
elif mech_type == "Krank-Biyel Mekanizması":
    r = st.sidebar.number_input("r (Krank)", value=get_default("r", 50.0))
    l = st.sidebar.number_input("l (Biyel)", value=get_default("l", 200.0))
    e = st.sidebar.number_input("e (Offset)", value=get_default("e", 0.0))
    L1, L2, L3, L4 = 0, r, l, 0
    mechanism = SliderCrankMechanism(r, l, e)
    solver = SliderCrankSolver(r, l, e, assembly_mode=assembly_mode_val)
else: # Kol-Kızak
    r1 = st.sidebar.number_input("O2-O4 Mesafesi (r1)", value=get_default("r1", 30.0))
    r2 = st.sidebar.number_input("Krank Uzunluğu (r2)", value=get_default("r2", 60.0))
    L1, L2, L3, L4 = r1, r2, 0, 0
    mechanism = InvertedSliderCrankMechanism(r1, r2)
    solver = mechanism # Mechanism itself has solve_kinematics

theta2 = st.sidebar.number_input("Giriş Açısı (theta2)", value=get_default("theta2", 45.0), step=1.0)
omega2 = st.sidebar.number_input("Açısal Hız (w2)", value=get_default("omega2", 10.0))
alpha2 = st.sidebar.number_input("Açısal İvme (alp2)", value=get_default("alpha2", 0.0))

enable_dynamics = st.sidebar.checkbox("Dinamik Analiz", value=get_default("enable_dyn", False))
show_instant_centers = False
if mech_type == "Dört Çubuk Mekanizması":
    show_instant_centers = st.sidebar.checkbox("Ani Dönme Merkezlerini Göster ($I_{13}, I_{24}$)", value=False)

dyn_params = {}
if enable_dynamics:
# ... (Keep existing dynamic params logic) ...
    if mech_type == "Dört Çubuk Mekanizması":
        m2 = st.sidebar.number_input("m2 (kg)", value=get_default("m2", 0.3))
        J2 = st.sidebar.number_input("J2 (kgm2)", value=get_default("J2", 0.0005), format="%.5f")
        O2G2 = st.sidebar.number_input("O2G2 (mm)", value=get_default("O2G2", 20.0))
        
        m3 = st.sidebar.number_input("m3 (kg)", value=get_default("m3", 0.5))
        J3 = st.sidebar.number_input("J3 (kgm2)", value=get_default("J3", 0.00121), format="%.5f")
        BG3 = st.sidebar.number_input("BG3 (mm)", value=get_default("BG3", 75.5))
        m4 = st.sidebar.number_input("m4 (kg)", value=get_default("m4", 0.4))
        J4 = st.sidebar.number_input("J4 (kgm2)", value=get_default("J4", 0.00091), format="%.5f")
        O4G4 = st.sidebar.number_input("O4G4 (mm)", value=get_default("O4G4", 55.5))
        dyn_params = {"m2":m2, "J2":J2, "O2G2":O2G2, "m3":m3, "J3":J3, "BG3":BG3, "m4":m4, "J4":J4, "O4G4":O4G4}
    else:
        m2 = st.sidebar.number_input("m2 (kg)", value=get_default("m2", 0.5))
        J2 = st.sidebar.number_input("J2 (kgm2)", value=get_default("J2", 0.01), format="%.5f")
        O2G2 = st.sidebar.number_input("O2G2 (mm)", value=get_default("O2G2", 25.0))
        
        m3 = st.sidebar.number_input("m3 (kg)", value=get_default("m3", 1.36))
        J3 = st.sidebar.number_input("J3 (kgm2)", value=get_default("J3", 0.0102), format="%.5f")
        BG3 = st.sidebar.number_input("BG3 (mm)", value=get_default("BG3", 51.0))
        m4 = st.sidebar.number_input("m4 (kg)", value=get_default("m4", 0.91))
        P_gas = st.sidebar.number_input("P_gas (N)", value=get_default("P_gas", 0.0))
        dyn_params = {"m2":m2, "J2":J2, "O2G2":O2G2, "m3":m3, "J3":J3, "BG3":BG3, "m4":m4, "P_gas":P_gas}

# --- NEW: ENGINEERING MODULES SIDEBAR ---
with st.sidebar.expander("Mukavemet & Malzeme", expanded=True):
    enable_stress = st.checkbox("Analizi Aktif Et", value=False)
    stress_params = {}
    if enable_stress:
         sel_mat = st.selectbox("Malzeme Seçimi", list(materials.MATERIALS_DB.keys()))
         mat_props = materials.MATERIALS_DB[sel_mat]
         st.caption(f"**{mat_props['name']}**")
         st.caption(f"Akma: {mat_props['Sy']} MPa | Çekme: {mat_props['Sut']} MPa")
         
         area_biyel = st.number_input("Biyel Kesit Alanı (mm²)", value=50.0)
         
         st.markdown("---")
         st.markdown("**Yorulma**")
         op_rpm = st.number_input("Çalışma Hızı (RPM)", value=1500.0, step=100.0)
         
         stress_params = {
             "material_key": sel_mat,
             "area_mm2": area_biyel,
             "rpm": op_rpm
         }

# Execute Solver
kin_state = solver.solve_kinematics(theta2, omega2, alpha2)
dyn_state = None
if kin_state and enable_dynamics:
    try:
        dyn_state = solver.solve_dynamics(kin_state, **dyn_params)
    except Exception as e:
        st.error(f"Dinamik Hata: {e}")

current_joints = mechanism.get_position(theta2, assembly_mode_val)

# --- OPTIMIZATION: CACHED CALCULATION ---
# --- GLOBAL CYCLE ANALYSIS ---
r_val = locals().get('r', 0)
l_val = locals().get('l', 0)
e_val = locals().get('e', 0)
r1_val = locals().get('r1', 0)
r2_val = locals().get('r2', 0)

stats, c_data = calculate_full_cycle(mech_type, omega2, alpha2, assembly_mode_val, dyn_params,
                                      L1, L2, L3, L4, r_val, l_val, e_val, r1_val, r2_val)

# --- MAIN TABS ---
# --- MAIN TABS ---
# Reordered: Static First
# --- MAIN TABS ---
# Reordered: Static First
tab_static, tab_sim, tab_3d, tab_graphs, tab_step, tab_data, tab_report = st.tabs(["STATİK GÖRÜNÜM & FBD", "SİMÜLASYON", "3D STÜDYO", "KİNEMATİK GRAFİKLER", "DEVRE KAPALILIK DENKLEMLERİ ÇÖZÜMÜ", "VERİ TABLOLARI", "RAPOR İNDİR"])

# 0. 3D STUDIO
with tab_3d:
    st.markdown("### 3D Mekanizma Stüdyosu")
    
    col_3d_main, col_3d_ctrl = st.columns([3, 1])
    
    with col_3d_ctrl:
        # Wrapped in a container for visual distinction (Panel effect)
        with st.container(border=True):
            st.markdown("**Görünüm Ayarları**")
            
            c1, c2, c3, c4 = st.columns(4)
            # Default View: ON (Front)
            if "view_3d" not in st.session_state: st.session_state.view_3d = "ON"
            
            if c1.button("📐 ISO", key="v_iso", help="İzometrik"): st.session_state.view_3d = "ISO"
            if c2.button("⏹️ ÖN", key="v_front", help="Ön (XY)"): st.session_state.view_3d = "ON"
            if c3.button("⬇️ ÜST", key="v_top", help="Üst (XZ)"): st.session_state.view_3d = "UST"
            if c4.button("➡️ YAN", key="v_side", help="Yan (YZ)"): st.session_state.view_3d = "YAN"
            
            st.write("") # Spacer
            
            p_val = st.slider("Krank Açısı (°)", 0.0, 360.0, float(theta2), step=1.0)
            thick = st.slider("Parça Kalınlığı (mm)", 1, 20, 5, step=1)
            
            c_p, c_g = st.columns(2)
            show_pins = c_p.checkbox("Pimleri Göster", value=True)
            show_grid = c_g.checkbox("Izgarayı Göster", value=False) 
            
            # Trace Toggle
            show_trace = st.toggle("Yörüngeyi Göster (Biyel)", value=False) 
            
            st.caption("💡 **İpucu:** Sol Tık: Çevir | Sağ Tık: Pan | Tekerlek: Zoom")

    with col_3d_main:
        # Calculate specialized 3D Position
        joints_3d = mechanism.get_position(p_val, assembly_mode_val)
        
        # Trace Path Construction
        trace_path_points = None
        if show_trace and c_data and 'joints' in c_data and 'B' in c_data['joints']:
            # Using point B for trace
            raw_pts = c_data['joints']['B']
            # Convert to 3D with correct Z-depth (e.g. at coupler level)
            z_trace = thick * 1.5 # Middle of coupler layer
            trace_path_points = [(p[0], p[1], z_trace) for p in raw_pts if p]

        if joints_3d:
            fig_3d = draw_mechanism_3d(
                joints_3d, mech_type, 
                show_pins=show_pins, show_grid=show_grid, thickness=thick, 
                camera_view=st.session_state.view_3d,
                trace_path=trace_path_points if show_trace else None
            )
            # Enable scrollZoom and ModeBar to allow better navigation
            st.plotly_chart(fig_3d, use_container_width=True, key="chart_3d_studio", 
                            config={'scrollZoom': True, 'displayModeBar': True})
        else:
            st.error("⚠️ Mekanizma bu açıda birleştirilemiyor/kilitlendi.")

# 1. STATIC VIEW (Clean, Information Rich)
with tab_static:
    if current_joints:
        # Calculate Instant Centers if requested
        ics = None
        if show_instant_centers and hasattr(mechanism, 'calculate_instant_centers'):
            ics = mechanism.calculate_instant_centers(theta2, assembly_mode_val)

        # 1. Main Visual
        fig_static = draw_mechanism(current_joints, L1, L2, L3, L4, omega2, alpha2, assembly_mode_val, 
                                    show_labels=True, show_angles=True, show_motion=True,
                                    instant_centers=ics)
        st.plotly_chart(fig_static, use_container_width=True, key="static_view_mech")
        
        # 2. Kinematic Metrics (Image 1 Style)
        st.markdown("### Kinematik Analiz Sonuçları")
        
        # Row 1: Angles
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Giriş Açısı ($\\theta_2$)", f"{theta2:.1f}°")
        
        if mech_type == "Dört Çubuk Mekanizması":
            c2.metric("Çıkış Açısı ($\\theta_4$)", f"{kin_state['theta4']:.1f}°" if kin_state else "-")
            c3.metric("Biyel Açısı ($\\theta_3$)", f"{kin_state['theta3']:.1f}°" if kin_state else "-")
            
            gs = mechanism.check_grashof()
            gs_text = "Grashof" if "Non" not in gs else "Grashof Değil"
            c4.metric("Durum", gs_text)
            
            st.markdown("---")
            
            # Row 2: Velocities & Accelerations
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Biyel Hızı ($\\omega_3$)", f"{kin_state['omega3']:.2f} rad/s" if kin_state else "-")
            v2.metric("Çıkış Hızı ($\\omega_4$)", f"{kin_state['omega4']:.2f} rad/s" if kin_state else "-")
            v3.metric("Biyel İvmesi ($\\alpha_3$)", f"{kin_state['alpha3']:.2f} rad/s²" if kin_state else "-")
            v4.metric("Çıkış İvmesi ($\\alpha_4$)", f"{kin_state['alpha4']:.2f} rad/s²" if kin_state else "-")
            
            st.markdown("---")
            st.markdown("**Mafsal Koordinatları [mm]**")
            k1, k2 = st.columns(2)
            k1.metric("A Noktası (Krank)", f"({current_joints['A'][0]:.1f}, {current_joints['A'][1]:.1f})")
            k2.metric("B Noktası (Biyel-Sarkaç)", f"({current_joints['B'][0]:.1f}, {current_joints['B'][1]:.1f})")

        elif mech_type == "Krank-Biyel Mekanizması":
            c2.metric("Çıkış Açısı", "0° (Piston)")
            c3.metric("Biyel Açısı ($\\theta_3$)", f"{kin_state['theta3']:.1f}°" if kin_state else "-")
            c4.metric("Durum", "Çözülebilir" if kin_state else "Kilitli")
            
            st.markdown("---")
            
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Biyel Hızı ($\\omega_3$)", f"{kin_state['omega3']:.2f} rad/s" if kin_state else "-")
            v2.metric("Piston Hızı ($V_P$)", f"{kin_state['v_piston']:.1f} mm/s" if kin_state else "-")
            v3.metric("Biyel İvmesi ($\\alpha_3$)", f"{kin_state['alpha3']:.2f} rad/s²" if kin_state else "-")
            v4.metric("Piston İvmesi ($a_P$)", f"{kin_state['a_piston']:.1f} mm/s²" if kin_state else "-")
            
            st.markdown("---")
            k1, k2 = st.columns(2)
            k1.metric("B Noktası (Krank)", f"({current_joints['B'][0]:.1f}, {current_joints['B'][1]:.1f})")
            k2.metric("C Noktası (Piston)", f"({current_joints['C'][0]:.1f}, {current_joints['C'][1]:.1f})")

        else: # Kol-Kızak (Whitworth)
            c2.metric("Çıkış Açısı ($\\theta_4$)", f"{kin_state['theta4']:.1f}°" if kin_state else "-")
            c3.metric("Kızak Konumu (r4)", f"{current_joints['r4_val']:.1f} mm" if current_joints else "-")
            c4.metric("Durum", "Çözülebilir")
            
            st.markdown("---")
            
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Kızak Hızı ($\\omega_4$)", f"{kin_state['omega4']:.2f} rad/s" if kin_state else "-")
            v2.metric("Kızak İvmesi ($\\alpha_4$)", f"{kin_state['alpha4']:.2f} rad/s²" if kin_state else "-")
            v3.metric("-", "-")
            v4.metric("-", "-")
            
            st.markdown("---")
            k1, k2 = st.columns(2)
            k1.metric("A Noktası (Krank/Piston)", f"({current_joints['A'][0]:.1f}, {current_joints['A'][1]:.1f})")
            k2.metric("B Noktası (Kol Ucu)", f"({current_joints['B'][0]:.1f}, {current_joints['B'][1]:.1f})")

        # 3. Dynamic Analysis (If Enabled)
        if enable_dynamics and dyn_state:
            st.markdown("---")
            st.subheader("Serbest Cisim Diyagramları (FBD)")
            
            # FBD Visuals (3 Separate Panels)
            fig2, fig3, fig4 = draw_fbd_separate(current_joints, dyn_state, mech_type)
            
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.plotly_chart(fig2, use_container_width=True, key="fbd_crank")
                st.markdown("**Krank (Uzuv 2) Dengesi**")
                st.latex(r"\sum \vec{F} = \vec{F}_{12} + \vec{F}_{32} + \vec{W}_2 = m_2 \vec{a}_{G2}")
                st.latex(r"\sum M_{G2} = T_{giriş} + (\vec{r} \times \vec{F}) = I_2 \alpha_2")

            with c2:
                st.plotly_chart(fig3, use_container_width=True, key="fbd_coupler")
                st.markdown("**Biyel (Uzuv 3) Dengesi**")
                st.latex(r"\sum \vec{F} = \vec{F}_{23} + \vec{F}_{43} + \vec{W}_3 = m_3 \vec{a}_{G3}")
                st.latex(r"\sum M_{G3} = (\vec{r} \times \vec{F}) = I_3 \alpha_3")

            with c3:
                st.plotly_chart(fig4, use_container_width=True, key="fbd_output")
                st.markdown("**Çıkış (Uzuv 4) Dengesi**")
                st.latex(r"\sum \vec{F} = \vec{F}_{34} + \vec{F}_{14} + \vec{W}_4 = m_4 \vec{a}_{G4}")
                if mech_type == "Krank-Biyel Mekanizması":
                     st.latex(r"\sum M_{G4} \approx 0 \quad (\text{Öteleme})")
                else:
                     st.latex(r"\sum M_{G4} = I_4 \alpha_4")
            
            st.markdown("### 📊 Dinamik Kuvvet & Tork Analizi")
            
            col_t, col_tbl = st.columns([1, 2])
            
            with col_t:
                T_val = dyn_state['T_input']
                t_dir = "↺ CCW" if T_val >= 0 else "↻ CW"
                st.metric(
                    label="Giriş Torku (Tork)", 
                    value=f"{abs(T_val):.1f} Nmm", 
                    delta=t_dir,
                    delta_color="off"
                )
                st.caption("Dönme Yönü: " + t_dir)

            with col_tbl:
                # Prepare Force Data with Angles
                forces_list = []
                if mech_type == "Krank-Biyel Mekanizması":
                     # Slider Forces, note F14 is strictly vertical usually if mu=0? No, F14 is Normal force.
                     # F12, F23, F34, F14
                     keys = [('F12', 'O2 (Krank-Zemin)'), ('F23', 'B (Krank-Biyel)'), ('F34', 'C (Biyel-Piston)'), ('F14', 'Piston-Duvar')]
                else:
                     # Four Bar Forces
                     keys = [('F12', 'O2 (Krank-Zemin)'), ('F23', 'A (Krank-Biyel)'), ('F34', 'B (Biyel-Sarkaç)'), ('F14', 'O4 (Sarkaç-Zemin)')]
                
                for key, desc in keys:
                    fx, fy = dyn_state.get(key, (0,0))
                    mag = np.hypot(fx, fy)
                    ang = np.degrees(np.arctan2(fy, fx)) % 360
                    forces_list.append({
                        "Kuvvet": key, 
                        "Konum": desc, 
                        "Büyüklük [N]": f"{mag:.1f}", 
                        "Yön (∠)": f"{ang:.1f}°"
                    })
                
                df_forces = pd.DataFrame(forces_list)
                # Apply styling is automatic with dataframe
                st.dataframe(df_forces, use_container_width=True, hide_index=True)
                
                # --- NEW: CG Accelerations ---
                st.markdown("### Kütle Merkezi İvmeleri (m/s²)")
                ag2 = dyn_state.get('aG2', (0,0))
                ag3 = dyn_state.get('aG3', (0,0))
                ag4 = dyn_state.get('aG4', (0,0))
                
                # Convert mm/s^2 to m/s^2
                ag2_m = (ag2[0]/1000.0, ag2[1]/1000.0)
                ag3_m = (ag3[0]/1000.0, ag3[1]/1000.0)
                ag4_m = (ag4[0]/1000.0, ag4[1]/1000.0)
                
                # Calculate angles (Degrees, 0-360)
                ang2 = np.degrees(np.arctan2(ag2[1], ag2[0])) % 360
                ang3 = np.degrees(np.arctan2(ag3[1], ag3[0])) % 360
                ang4 = np.degrees(np.arctan2(ag4[1], ag4[0])) % 360
                
                ag_data = {
                    "Kütle Merkezi": ["G2 (Krank)", "G3 (Biyel)", "G4 (Çıkış/Piston)"],
                    "Toplam İvme [m/s²]": [
                         f"{np.hypot(*ag2_m):.2f}", 
                         f"{np.hypot(*ag3_m):.2f}", 
                         f"{np.hypot(*ag4_m):.2f}"
                    ],
                    "Yön (∠)": [f"{ang2:.1f}°", f"{ang3:.1f}°", f"{ang4:.1f}°"],
                    "ax [m/s²]": [f"{ag2_m[0]:.2f}", f"{ag3_m[0]:.2f}", f"{ag4_m[0]:.2f}"],
                    "ay [m/s²]": [f"{ag2_m[1]:.2f}", f"{ag3_m[1]:.2f}", f"{ag4_m[1]:.2f}"]
                }
                st.dataframe(pd.DataFrame(ag_data), use_container_width=True, hide_index=True)

    else:
        st.error("Mekanizma bu konumda oluşturulamıyor.")

# 2. SIMULATION (Direct Animation)
with tab_sim:
    st.markdown("### Hareket Simülasyonu")
    
    # Direct Animation Display (No Button)
    # The figure from create_animation_figure is heavy but requested to be direct.
    # It contains its own Play/Pause buttons in the subplot HUD.
    
    with st.spinner("Simülasyon yükleniyor..."):
        # We need to make sure mechanism is valid
        if current_joints:
             anim_fig = create_animation_figure(mechanism, theta2, L1, L2, L3, L4, omega2, alpha2, assembly_mode_val)
             if anim_fig:
                 st.plotly_chart(anim_fig, use_container_width=True, key="anim_direct_chart")
                 st.info("Simülasyonu başlatmak için grafik üzerindeki '▶' düğmesine basınız.")
        else:
             st.error("Mekanizma bu konumda geçersiz, animasyon oluşturulamadı.")

# 3. KINEMATIC GRAPHS
with tab_graphs:
    # Use global c_data
    f_v, f_a, f_p = plot_kinematic_curves(
        c_data["theta2"], c_data["omega3"], c_data["output_vel"], 
        c_data["alpha3"], c_data["output_acc"], 
        ma_list=c_data.get("ma"), mu_list=c_data.get("mu"), 
        mech_type=mech_type
    )
    st.plotly_chart(f_v, use_container_width=True, key="g_vel")
    st.plotly_chart(f_a, use_container_width=True, key="g_acc")
    st.plotly_chart(f_p, use_container_width=True, key="g_props")
    
    # --- ENGINEERING ANALYSIS (NEW) ---
    if enable_dynamics and enable_stress and 'F23' in c_data:
        st.markdown("---")
        st.header("🛡️ Mühendislik Analizi Raporu")
        
        # 0. Physical Properties
        st.subheader("1. Fiziksel Özellikler (Tahmini)")
        if 'material_key' in stress_params:
            mat_key = stress_params['material_key']
            den = materials.MATERIALS_DB[mat_key]['density'] # g/cm3
            
            # Simple Mass Calc (g)
            l_biyel = L3 if mech_type=="Dört Çubuk Mekanizması" else l
            # Volume (cm3) = Length(cm) * Area(cm2)
            # Area input is mm2 -> /100 = cm2
            vol_coupler_cm3 = (l_biyel / 10.0) * (stress_params['area_mm2'] / 100.0) 
            mass_coupler = vol_coupler_cm3 * den
            
            # Total Mass Estimate (Rough)
            # Assuming Crank/Rocker have similar cross-section for simplicity or user ignored
            
            c_mat1, c_mat2 = st.columns(2)
            c_mat1.metric(f"Biyel Kütlesi ({materials.MATERIALS_DB[mat_key]['name']})", f"{mass_coupler:.1f} g")
            c_mat2.metric(f"Malzeme Yoğunluğu", f"{den} g/cm³")

        # 1. Stress Analysis
        st.subheader("2. Mukavemet ve Güvenlik Analizi (Biyel Uzvu)")
        
        forces_biyel = c_data['F23'] # Forces on Coupler (Joint A/B)
        
        # Calculate Logic
        s_res = stress.calculate_stress_safety(
            forces_biyel, 
            stress_params['area_mm2'], 
            stress_params['material_key']
        )
        
        if s_res:
            c1, c2, c3 = st.columns(3)
            c1.metric("Maksimum Gerilme", f"{s_res['sigma_max_abs']:.1f} MPa")
            
            delta_col = "normal"
            if s_res['fos'] < 1.0: delta_col = "inverse"
            c2.metric("Güvenlik Faktörü (FOS)", f"{s_res['fos']:.2f}", delta="Riskli" if s_res['fos'] < 1.0 else "Güvenli", delta_color=delta_col)
            
            c3.metric("Malzeme Akma Limiti", f"{s_res['Sy']} MPa")
            
            if not s_res['is_safe']:
                st.error(f"⚠️ DİKKAT: Biyel uzvu üzerindeki gerilme ({s_res['sigma_max_abs']:.1f} MPa), malzeme akma sınırını aşıyor! Lütfen kesit alanını artırın.")

            # Plot Stress Cycle
            import plotly.graph_objects as go
            fig_stress = go.Figure()
            fig_stress.add_trace(go.Scatter(x=c_data['theta2'], y=s_res['sigma'], name="Gerilme (σ)", line=dict(color='#FF5252', width=2)))
            fig_stress.add_hline(y=s_res['Sy'], line_dash="dash", line_color="orange", annotation_text="Akma Sınırı (Sy)")
            fig_stress.add_hline(y=-s_res['Sy'], line_dash="dash", line_color="orange")
            
            fig_stress.update_layout(
                title="Çevrimsel Gerilme Analizi", 
                xaxis_title="Krank Açısı (°)", 
                yaxis_title="Normal Gerilme (MPa)", 
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig_stress, use_container_width=True)

            # 2. Fatigue Analysis
            st.subheader("3. Yorulma Ömrü Analizi")
            
            f_res = fatigue.calculate_fatigue_life(
                s_res['sigma_max'], 
                s_res['sigma_min'], 
                stress_params['material_key'], 
                stress_params['rpm']
            )
            
            f1, f2, f3 = st.columns(3)
            cycle_txt = "Sonsuz (>10⁷)" if f_res['life_cycles'] == float('inf') or f_res['life_cycles'] > 1e7 else f"{int(f_res['life_cycles']):,}"
            
            hours = f_res['life_hours']
            time_txt = "Sonsuz"
            if hours != float('inf'):
                 if hours > 24*365:
                     time_txt = f"{hours/(24*365):.1f} Yıl"
                 else:
                     time_txt = f"{hours:.1f} Saat"
            
            f1.metric("Tahmini Ömür", cycle_txt)
            f2.metric("Çalışma Süresi", time_txt)
            
            r_level = f_res['risk_level']
            r_delta = "off"
            if r_level == "Yüksek Risk": r_delta = "inverse"
            f3.metric("Risk Seviyesi", r_level, delta="Dikkat" if r_level!="Düşük Risk" else "Güvenli", delta_color=r_delta)
            
            # S-N Plot
            wp = f_res['wohler_params']
            fig_sn = plot_sn_curve(
                wp['Sut'], wp['Se'], wp['a'], wp['b'], 
                f_res['sigma_amp'], f_res['life_cycles'], 
                s_res['material']
            )
            st.plotly_chart(fig_sn, use_container_width=True)
            
        else:
            st.warning("Gerilme hesabı yapılamadı (Alan veya Kuvvet verisi eksik).")
            
    elif enable_stress and not enable_dynamics:
        st.info("ℹ️ Mukavemet analizini görmek için lütfen sol panelden **'Dinamik Analiz'** seçeneğini de aktif ediniz.")



# 4. STEP BY STEP
with tab_step:
    if hasattr(solver, 'steps') and solver.steps:
        # Helper to get category safely
        def get_cat(x): return getattr(x, 'category', 'Genel')
        
        # Itertools groupby expects sorted input for grouping
        for category, steps_iter in itertools.groupby(solver.steps, key=get_cat):
            # Create Section
            with st.expander(f"**{category}**", expanded=True):
                for s in steps_iter:
                    # Card Layout
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown(f"##### {s.title}")
                        if s.formula:
                            st.latex(s.formula)
                        if s.substitution:
                            st.caption("Yerine Koyma:")
                            st.latex(s.substitution)
                    with c2:
                        # Result Box
                        st.info(f"**Sonuç:**\n\n{s.result_str}")
                        if s.description:
                            st.markdown(f"*{s.description}*")
                    
                    st.divider()
    else:
        st.write("Veri yok veya analiz yapılmadı.")

# 5. DATA TABLES
# 5. DATA TABLES
with tab_data:
    # Use global enriched c_data
    
    # Improve Dataframe Columns for Display
    # Exclude 'joints' dictionary which causes DataFrame error
    safe_data = {k: v for k, v in c_data.items() if k != 'joints'}
    display_df = pd.DataFrame(safe_data)
    
    # Rename map based on mechanism type
    rename_map = {
        "theta2": "Giriş Açısı ($\\theta_2$)",
        "theta3": "Biyel Açısı ($\\theta_3$)",
        "omega3": "Biyel Hızı ($\\omega_3$)",
        "alpha3": "Biyel İvmesi ($\\alpha_3$)",
        "T_input": "Giriş Torku (Nm)",
        "F12": "F12 (N)", "F34": "F34 (N)"
    }
    
    if mech_type == "Dört Çubuk Mekanizması":
        rename_map.update({
            "theta4": "Çıkış Açısı ($\\theta_4$)",
            "output_vel": "Çıkış Hızı ($\\omega_4$)",
            "output_acc": "Çıkış İvmesi ($\\alpha_4$)",
            "mu": "Bağlama Açısı ($\\mu$)"
        })
    elif mech_type == "Krank-Biyel Mekanizması":
        rename_map.update({
            "theta3": "Biyel Açısı ($\\theta_3$)",
            "output_vel": "Piston Hızı ($V_P$)",
            "output_acc": "Piston İvmesi ($a_P$)"
        })
        if "theta4" in display_df.columns: display_df.drop(columns=["theta4"], inplace=True)

    else: # Kol-Kızak
        rename_map.update({
            "theta4": "Kızak Açısı ($\\theta_4$)",
            "output_vel": "Kızak Hızı ($\\omega_4$)",
            "output_acc": "Kızak İvmesi ($\\alpha_4$)"
        })

    display_df.rename(columns=rename_map, inplace=True)
    
    # Reorder
    # Note: Using generic names after rename
    preferred_order = [
        "Giriş Açısı ($\\theta_2$)", "Giriş Torku (Nm)",
        "Biyel Açısı ($\\theta_3$)", rename_map.get("theta4"), 
        "Biyel Hızı ($\\omega_3$)", rename_map.get("output_vel"),
        "F12 (N)", "F34 (N)"
    ]
    cols_to_show = [c for c in preferred_order if c and c in display_df.columns]
    display_df = display_df[cols_to_show]

    st.dataframe(display_df, use_container_width=True)

# 6. REPORT TAB
with tab_report:
    st.markdown("### Analiz Raporu Oluştur")
    st.write("Analiz sonuçlarını içeren PDF veya MATLAB dosyasını indirin.")
    
    col_pdf1, col_pdf2 = st.columns([1, 1])
    with col_pdf1:
        st.write("**PDF Raporu**")
        # Generate PDF logic
        if st.button("Raporu Oluştur (PDF)", key="btn_gen_rep"):
             try:
                pdf_path = create_pdf(L1, L2, L3, L4, omega2, alpha2, mech_type, theta2, {"w3": kin_state['omega3']} if kin_state else {}, cycle_stats if 'stats' in locals() else {})
                with open(pdf_path, "rb") as f:
                    st.download_button("İndir PDF", f, file_name="rapor.pdf", mime="application/pdf")
             except Exception as e:
                st.error(f"Hata: {e}")
    
    with col_pdf2:
         st.write("**MATLAB Script (R2015)**")
         # Params
         p_dict = {'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4, 'w2': omega2, 'alpha2': alpha2}
         try:
            p_dict.update({'m2':m2, 'J2':J2, 'm3':m3, 'J3':J3, 'm4':m4})
         except: pass 
         
         if 'c_data' in locals():
            ml_code = generate_matlab_code(c_data, p_dict, mech_type)
            st.download_button(
                label="💾 MATLAB Dosyasını İndir (.m)",
                data=ml_code,
                file_name="mekanizma_simulasyon.m",
                mime="text/plain"
            )
            
            st.divider()
            st.write("**Excel Dashboard**")
            # Generate Excel
            try:
                excel_data = generate_excel_report(c_data, p_dict)
                st.download_button(
                    label="📊 Excel Dashboard İndir (.xlsx)",
                    data=excel_data,
                    file_name="Analiz_Dashboard.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.write(f"Excel oluşturulamadı: {e}")

            st.divider()
            st.write("**Premium Export**")
            
            # DXF
            if st.button("📐 CAD Dosyası Oluştur (.dxf)", key="btn_dxf"):
                 # Use global current_joints
                 # And c_data for path
                 try:
                     dxf_bytes = exporters.export_to_dxf(current_joints, c_data)
                     st.download_button("İndir DXF", dxf_bytes, "mekanizma_cizimi.dxf", "application/dxf")
                 except Exception as e:
                     st.error(f"DXF Hatası: {e}")
                 
            # GIF
            if st.button("🎬 Animasyon Oluştur (.gif)", key="btn_gif"):
                 try:
                     with st.spinner("Animasyon işleniyor (bu işlem birkaç saniye sürebilir)..."):
                          gif_path = exporters.generate_animation_gif(mechanism, L1, L2, L3, L4, range(0, 360, 5))
                          with open(gif_path, "rb") as f:
                              st.download_button("İndir GIF", f, "mekanizma_animasyon.gif", "image/gif")
                 except Exception as e:
                     st.error(f"GIF Hatası: {e}")

            st.divider()
            st.write("**🤖 Mekatronik Entegrasyon (Arduino)**")
            
            with st.expander("Arduino Kodu Oluşturucu"):
                ard_col1, ard_col2 = st.columns(2)
                with ard_col1:
                    a_motor = st.radio("Motor Tipi", ["Standart Servo (0-180)", "Sürekli Servo", "Step Motor"])
                    a_pin = st.number_input("Pin No", value=9, step=1)
                
                with ard_col2:
                    a_mode = st.selectbox("Hareket Modu", ["Tek Tur (One-Shot)", "Döngüsel (Loop)", "Sarkaç (Sweep)"])
                    a_rpm = st.slider("Hedef Hız (RPM)", 1, 60, 10)
                
                # Data Source Logic
                target_data = c_data['theta2'] # Default
                data_name = "Giriş Mili"
                
                if "Step" in a_motor:
                    target_data = c_data['theta2'] # Drive Crank
                    data_name = "Giriş Mili (Krank)"
                elif 'theta4' in c_data:
                     # Check if theta4 is valid (not None list)
                     # c_data['theta4'] might contain Nones
                     valid_t4 = [x for x in c_data['theta4'] if x is not None and not np.isnan(x)]
                     if valid_t4:
                         target_data = valid_t4
                         data_name = "Çıkış Uzvu (Sarkaç)"
                
                if st.button("Kodu Üret"):
                    # Clean NaNs
                    clean_data = [d if d is not None and not np.isnan(d) else 0 for d in target_data]
                    
                    code_ino = arduino_gen.generate_arduino_code(clean_data, a_motor, int(a_pin), a_rpm, a_mode)
                    
                    st.success(f"Kod üretildi! ({data_name} verisi kullanıldı)")
                    st.code(code_ino, language='cpp')
                    
                    st.download_button("💾 .ino Dosyasını İndir", code_ino, "mekanizma_kontrol.ino", "text/plain")



