import numpy as np
from .base_solver import BaseSolver, SolutionStep, MechanismState

class FourBarSolver(BaseSolver):
    """
    Solver for Four-Bar Linkage Kinematics and Dynamics.
    Uses analytical and vector methods.
    """
    def __init__(self, L1, L2, L3, L4, assembly_mode=1):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4
        self.mode = assembly_mode
        self.mode = assembly_mode
        self.steps = []
        self.current_category = "Genel"

    def _add_step(self, title, formula, sub, res_val, unit, desc=""):
        if isinstance(res_val, (list, tuple)):
            val_str = ", ".join([f"{v:.2f}" for v in res_val])
            res_str = f"({val_str}) {unit}"
        elif isinstance(res_val, str):
             res_str = f"{res_val} {unit}".strip()
        elif res_val is None:
             res_str = "-"
        else:
            res_str = f"{res_val:.2f} {unit}"
        self.steps.append(SolutionStep(title, formula, sub, res_str, res_val, desc, self.current_category))

    def solve_kinematics(self, theta2_deg, omega2, alpha2):
        self.steps = [] # Reset steps
        self.current_category = "📐 Kinematik Hazırlık"
        
        # --- METU STANDARD NOTATION ---
        # a1 (Ground), a2 (Crank), a3 (Coupler), a4 (Rocker)
        a1, a2, a3, a4 = self.L1, self.L2, self.L3, self.L4
        theta12_deg = theta2_deg
        theta12 = np.radians(theta12_deg)
        w12 = omega2
        alp12 = alpha2

        # 0. Vector Loop Equation (Devre Kapalılık Denklemi)
        self._add_step(
            "Devre Kapalılık Denklemi (Vektör Döngüsü)",
            r"\vec{A_0A} + \vec{AB} = \vec{A_0B_0} + \vec{B_0B} \\ a_2 e^{i\theta_{12}} + a_3 e^{i\theta_{13}} = a_1 + a_4 e^{i\theta_{14}}",
            r"a_1=" + f"{a1}, a_2={a2}, a_3={a3}, a_4={a4}",
            "", "", "Temel vektör kapalılık denklemi. Hedef: Tüm konum değişkenlerini ($\\theta_{13}, \\theta_{14}$) bulmak."
        )

        # 1. Position Analysis (Raven Method / Geometric Intersection)
        Ax = a2 * np.cos(theta12)
        Ay = a2 * np.sin(theta12)
        
        self._add_step(
            "Giriş Uzvu (Krank) Konumu", 
            r"A_x = a_2 \cos \theta_{12}, \quad A_y = a_2 \sin \theta_{12}",
            f"A_x = {a2} \\cos {theta12_deg:.1f}^\\circ, \\quad A_y = {a2} \\sin {theta12_deg:.1f}^\\circ",
            (Ax, Ay), "mm", "A noktasının (Krank ucu) koordinatları."
        )

        d_sq = (a1 - Ax)**2 + (-Ay)**2
        d = np.sqrt(d_sq)
        dist_AO4 = d
        
        # Validity Check
        if dist_AO4 > a3 + a4 or dist_AO4 < abs(a3 - a4):
             self.steps.append(SolutionStep("Hata", "", "", "Mekanizma Kilitlendi", 0, "Uzuv boyları bu açı için uygun değil."))
             return None

        # Geometric solution (Circle Intersection)
        # s (diagonal) calculation
        val_cos_phi = (a3**2 - a4**2 + d**2) / (2 * a3 * d) 
        # Clamp for numerical safety
        val_cos_phi = max(-1.0, min(1.0, val_cos_phi))
        
        # Use simple geometric intersection logic for clarity in code, map to Raven method in UI
        a_len = (a3**2 - a4**2 + d**2) / (2 * d)
        h_len = np.sqrt(max(0, a3**2 - a_len**2))
        
        O4x, O4y = a1, 0
        x2 = Ax + a_len * (O4x - Ax) / d
        y2 = Ay + a_len * (O4y - Ay) / d
        
        Bx = x2 + (self.mode * h_len) * (O4y - Ay) / d
        By = y2 - (self.mode * h_len) * (O4x - Ax) / d
        
        # Calculate Angles
        t3 = np.arctan2(By - Ay, Bx - Ax)
        t4 = np.arctan2(By, Bx - a1)
        theta13_deg = np.degrees(t3)
        theta14_deg = np.degrees(t4)
        
        self._add_step(
            "Biyel Açısı Çözümü ($\\theta_{13}$)",
            r"\theta_{13} = \arctan\left(\frac{B_y - A_y}{B_x - A_x}\right)",
            f"\\theta_{13} = \\arctan({(By-Ay):.1f} / {(Bx-Ax):.1f})",
            theta13_deg, "°", "Devre kapalılık denkleminin geometrik/analitik yöntemle çözümü."
        )
        self._add_step(
            "Çıkış Kolu Açısı Çözümü ($\\theta_{14}$)",
            r"\theta_{14} = \arctan\left(\frac{B_y - B_{0y}}{B_x - B_{0x}}\right)",
            f"\\theta_{14} = \\arctan({By:.1f} / {(Bx-a1):.1f})",
            theta14_deg, "°"
        )
        
        # 2. Velocity Analysis (Derivative of Loop Equation)
        # V_A = w12 * a2
        Va_x = -a2 * w12 * np.sin(theta12)
        Va_y =  a2 * w12 * np.cos(theta12)
        
        denom = np.sin(t3 - t4)
        if abs(denom) < 1e-4:
            w13, w14 = 0, 0
            desc_vel = "Kritik konum (Determinant ~ 0)"
        else:
            w13 = w12 * (a2 * np.sin(t4 - theta12)) / (a3 * denom)
            w14 = w12 * (a2 * np.sin(t3 - theta12)) / (a4 * denom)
            desc_vel = "Hız devre denkleminin türevi ile çözüm."
            
        self._add_step(
            "Biyel Açısal Hızı ($\\omega_{13}$)",
            r"\omega_{13} = \frac{a_2 \omega_{12} \sin(\theta_{14} - \theta_{12})}{a_3 \sin(\theta_{13} - \theta_{14})}",
            f"\\omega_{13} = \\frac{{{a2} \\cdot {w12} \\cdot \\sin({theta14_deg:.1f}^\\circ - {theta12_deg:.1f}^\\circ)}}{{{a3} \\cdot \\sin({theta13_deg:.1f}^\\circ - {theta14_deg:.1f}^\\circ)}}",
            w13, "rad/s", desc_vel
        )
        
        # 3. Acceleration Analysis
        # K_x and K_y calculation (Intermediate terms)
        A_val = -a2 * alp12 * np.sin(theta12) - a2 * w12**2 * np.cos(theta12) - \
                a3 * w13**2 * np.cos(t3) + a4 * w14**2 * np.cos(t4)
        B_val =  a2 * alp12 * np.cos(theta12) - a2 * w12**2 * np.sin(theta12) - \
                a3 * w13**2 * np.sin(t3) + a4 * w14**2 * np.sin(t4)
        
        det = a3 * a4 * np.sin(t4 - t3)
        
        if abs(det) < 1e-4:
            alp13, alp14 = 0, 0
        else:
            alp13 = (A_val * (-a4 * np.cos(t4)) - B_val * (a4 * np.sin(t4))) / det
            alp14 = ((-a3 * np.sin(t3)) * B_val - (a3 * np.cos(t3)) * A_val) / det
            
        self._add_step(
            "İvme Analizi - Ara Terimler ($K_x, K_y$)",
            r"K_x = \text{Bilinen } X, \quad K_y = \text{Bilinen } Y",
            f"K_x = {A_val:.2f}, \quad K_y = {B_val:.2f}",
            (A_val, B_val), "mm/s^2", "İvme döngü denkleminden gelen sabit terimlerin toplamı."
        )

        self._add_step(
            "Biyel Açısal İvmesi ($\\alpha_{13}$)",
            r"\alpha_{13} = \frac{K_x \cos\theta_{14} + K_y \sin\theta_{14}}{a_3 \sin(\theta_{13} - \theta_{14})}",
            f"\\alpha_{13} = \\frac{{{A_val:.1f} \\cos {theta14_deg:.1f}^\\circ + {B_val:.1f} \\sin {theta14_deg:.1f}^\\circ}}{{{a3} \\sin({theta13_deg-theta14_deg:.1f}^\\circ)}}", 
            alp13, "rad/s^2", "Biyel açısal ivmesinin analitik çözümü."
        )
        
        return {
            "theta2": theta12_deg,
            # Map internal METU vars to App Standard vars
            "theta3": theta13_deg, "theta4": theta14_deg,
            "t2_rad": theta12, "t3_rad": t3, "t4_rad": t4,
            "omega2": w12, "omega3": w13, "omega4": w14,
            "alpha2": alp12, "alpha3": alp13, "alpha4": alp14,
            "A": (Ax, Ay), "B": (Bx, By), "O4": (a1, 0)
        }

    def solve_dynamics(self, kin_state, m2, J2, O2G2, m3, J3, BG3, m4, J4, O4G4):
        """
        Solves Inverse Dynamics using the Analytical Matrix Method (Example 7).
        """
        if not kin_state: return None
        
        def fmt_vec(fx, fy):
            mag = np.hypot(fx, fy)
            ang = np.degrees(np.arctan2(fy, fx)) % 360
            return f"{mag:.1f} N ∠ {ang:.1f}°"
            
        def fmt_rot(val, unit="Nm"):
            symbol = "↺" if val > 0 else "↻" # Counter-clockwise positive
            return f"{abs(val):.1f} {unit} {symbol}"

        # Unpack Kinematics (Angles in Radians for Calc, Degrees for display if needed)
        t2 = kin_state["t2_rad"]
        t3 = kin_state["t3_rad"]
        t4 = kin_state["t4_rad"]
        w2, a2 = kin_state["omega2"], kin_state["alpha2"]
        w3, a3 = kin_state["omega3"], kin_state["alpha3"]
        w4, a4 = kin_state["omega4"], kin_state["alpha4"]
        
        # Lengths (mm)
        l2, l3, l4 = self.L2, self.L3, self.L4
        
        self.current_category = "📐 Kinematik Hazırlık"
        
        # --- 1. Acceleration Analysis (CGs) ---
        # G2 (Crank)
        rG2x = O2G2 * np.cos(t2)
        rG2y = O2G2 * np.sin(t2)
        aG2x = -O2G2 * w2**2 * np.cos(t2) - O2G2 * a2 * np.sin(t2)
        aG2y = -O2G2 * w2**2 * np.sin(t2) + O2G2 * a2 * np.cos(t2)
        
        # G3 (Coupler) - Calculated relative to A?
        # Image Formula requires r3 (AG3).
        # We have BG3 input. Assuming Centroidal axis. 
        # r3 (AG3) = L3 - BG3 ??
        # Let's calculate G3 coords physically first.
        # A position:
        Ax = l2 * np.cos(t2)
        Ay = l2 * np.sin(t2)
        # B position:
        Bx = kin_state["B"][0]
        By = kin_state["B"][1]
        
        # Vector A->B
        ABx, ABy = Bx - Ax, By - Ay
        dist_AB = np.sqrt(ABx**2 + ABy**2) # Should be l3
        
        # G3 Position (Linear interpolation based on BG3)
        # If BG3 is distance from B, and we assume uniform rod G3 is on line AB.
        # Ag3_dist = L3 - BG3
        ag3_dist = self.L3 - BG3 if self.L3 > BG3 else 0.5 * self.L3
        # Use simple interpolation for Accel
        aAx = -l2 * w2**2 * np.cos(t2) - l2 * a2 * np.sin(t2)
        aAy = -l2 * w2**2 * np.sin(t2) + l2 * a2 * np.cos(t2)
        
        aBx = -l4 * w4**2 * np.cos(t4) - l4 * a4 * np.sin(t4)
        aBy = -l4 * w4**2 * np.sin(t4) + l4 * a4 * np.cos(t4)
        
        ratio = ag3_dist / self.L3
        aG3x = aAx + (aBx - aAx) * ratio
        aG3y = aAy + (aBy - aAy) * ratio
        
        # G4 (Rocker) - Relative to O4
        # r4 = O4G4
        r4 = O4G4
        aG4x = -r4 * w4**2 * np.cos(t4) - r4 * a4 * np.sin(t4)
        aG4y = -r4 * w4**2 * np.sin(t4) + r4 * a4 * np.cos(t4)
        
        # Display Accelerations
        # Display Accelerations with detailed formulas
        mag2 = np.hypot(aG2x, aG2y)
        self._add_step(
             "G2 İvmesi ($\\vec{a}_{G_2}$)",
             r"\vec{a}_{G_2} = \vec{\alpha}_2 \times \vec{r}_{G_2} - \omega_2^2 \vec{r}_{G_2}",
             f"a_{{G_2x}}={aG2x:.1f}, \\quad a_{{G_2y}}={aG2y:.1f}",
             f"{mag2:.1f}", "mm/s^2",
             "Krank üzerindeki $G_2$ noktasının ivmesi (Sabit eksenli dönme)."
        )

        mag3 = np.hypot(aG3x, aG3y)
        self._add_step(
             "G3 İvmesi ($\\vec{a}_{G_3}$)",
             r"\vec{a}_{G_3} = \vec{a}_A + \frac{AG_3}{AB}(\vec{a}_B - \vec{a}_A)",
             f"a_{{G_3x}}={aG3x:.1f}, \\quad a_{{G_3y}}={aG3y:.1f}",
             f"{mag3:.1f}", "mm/s^2",
             "$G_3$ ivmesi, $A$ ve $B$ ivmeleri kullanılarak interpolasyonla bulunur."
        )

        mag4 = np.hypot(aG4x, aG4y)
        self._add_step(
             "G4 İvmesi ($\\vec{a}_{G_4}$)",
             r"\vec{a}_{G_4} = \vec{\alpha}_4 \times \vec{r}_{G_4} - \omega_4^2 \vec{r}_{G_4}",
             f"a_{{G_4x}}={aG4x:.1f}, \\quad a_{{G_4y}}={aG4y:.1f}",
             f"{mag4:.1f}", "mm/s^2",
             "Sarkaç üzerindeki $G_4$ noktasının ivmesi."
        )

        # --- 2. Inertia Forces & Torques ---
        # Forces (N) - Mass in kg, Acc in mm/s^2 -> * 1e-3
        F2x, F2y = -m2*aG2x*1e-3, -m2*aG2y*1e-3
        F3x, F3y = -m3*aG3x*1e-3, -m3*aG3y*1e-3
        F4x, F4y = -m4*aG4x*1e-3, -m4*aG4y*1e-3
        
        # Torques (Nm) -> J in kgm2, alpha in rad/s^2 -> Nm
        T2_inert = -J2 * a2 
        T3_inert = -J3 * a3
        T4_inert = -J4 * a4
        
        self.current_category = "⚖️ Atalet Kuvvetleri"
        
        self._add_step(
            "Atalet Kuvvetleri", 
            r"F = -m a_G, \quad T = -J \alpha",
            f"F3={fmt_vec(F3x, F3y)}, T3={fmt_rot(T3_inert, 'Nm')} | F4={fmt_vec(F4x, F4y)}, T4={fmt_rot(T4_inert, 'Nm')}",
            "", ""
        )
        
        # --- 3. Matrix System Setup (Example 7 Steps) ---
        
        self.current_category = "🧩 Dinamik Denge (Matris Yöntemi)"
        
        # 3.1 Coefficients a_ij
        s3, c3 = np.sin(t3), np.cos(t3)
        s4, c4 = np.sin(t4), np.cos(t4)
        
        a11 = -l4 * s4
        a12 =  l4 * c4
        a21 =  l3 * s3
        a22 = -l3 * c3
        
        self._add_step(
            "Katsayılar (a_ij)",
            r"a_{11}=-l_4 \sin\theta_4, \quad a_{12}=l_4 \cos\theta_4 \\ a_{21}=l_3 \sin\theta_3, \quad a_{22}=-l_3 \cos\theta_3",
            f"a_{{11}} = -{l4} \\sin({np.degrees(t4):.0f}^\\circ) = {a11:.1f} \\\\ a_{{12}} = {l4} \\cos({np.degrees(t4):.0f}^\\circ) = {a12:.1f} \\\\ a_{{21}} = {l3} \\sin({np.degrees(t3):.0f}^\\circ) = {a21:.1f} \\\\ a_{{22}} = -{l3} \\cos({np.degrees(t3):.0f}^\\circ) = {a22:.1f}",
            "", "mm"
        )
        
        # 3.2 Constants b_i
        T3_Nmm = T3_inert * 1000.0
        T4_Nmm = T4_inert * 1000.0
        r3 = ag3_dist # AG3
        
        b1 = (F4x * r4 * s4) - (F4y * r4 * c4) - T4_Nmm
        b2 = (F3x * r3 * s3) - (F3y * r3 * c3) - T3_Nmm
        
        self._add_step(
            "Sabitler (b_i)",
            r"b_1 = F_{4x}r_4 \sin\theta_4 - F_{4y}r_4 \cos\theta_4 - T_4 \\ b_2 = F_{3x}r_3 \sin\theta_3 - F_{3y}r_3 \cos\theta_3 - T_3",
            f"b_1 = ({F4x:.1f})({r4})\\sin({np.degrees(t4):.0f}^\\circ) - ({F4y:.1f})({r4})\\cos({np.degrees(t4):.0f}^\\circ) - ({T4_Nmm:.1f}) = {b1:.1f} \\\\ b_2 = ({F3x:.1f})({r3})\\sin({np.degrees(t3):.0f}^\\circ) - ({F3y:.1f})({r3})\\cos({np.degrees(t3):.0f}^\\circ) - ({T3_Nmm:.1f}) = {b2:.1f}",
            "", "Nmm"
        )
        
        # --- 4. Solve System ---
        det = a11 * a22 - a12 * a21
        
        if abs(det) < 1e-4:
            F34x, F34y = 0, 0
            res_str = "Tekillik"
        else:
            F34x = (a22 * b1 - a12 * b2) / det
            F34y = (a11 * b2 - a21 * b1) / det
            res_str = f"F_{{34x}}={F34x:.1f}, F_{{34y}}={F34y:.1f}"
            
        self._add_step(
            "Bilinmeyen Kuvvetler (F34)",
            r"F_{34x} = \frac{a_{22}b_1 - a_{12}b_2}{a_{11}a_{22} - a_{12}a_{21}}, \quad F_{34y} = \frac{a_{11}b_2 - a_{21}b_1}{a_{11}a_{22} - a_{12}a_{21}}",
            res_str + f" (Det={det:.0f})",
            fmt_vec(F34x, F34y), "N"
        )
        
        # --- 5. Calculate Other Forces ---
        # F14
        F14x = -F34x - F4x
        F14y = -F34y - F4y
        
        self.current_category = "🏁 Sonuçlar ve Tork"

        self._add_step(
            "F14 Kuvveti",
            r"\vec{F}_{14} = -\vec{F}_{34} - \vec{F}_4",
            f"F_{{14x}} = -({F34x:.1f}) - ({F4x:.1f}) = {F14x:.1f} \\\\ F_{{14y}} = -({F34y:.1f}) - ({F4y:.1f}) = {F14y:.1f}",
            fmt_vec(F14x, F14y), "N"
        )
        
        # F23
        F23x = F34x - F3x
        F23y = F34y - F3y
        self._add_step(
            "F23 Kuvveti",
            r"\vec{F}_{23} = \vec{F}_{34} - \vec{F}_3",
            f"F_{{23x}} = {F34x:.1f} - ({F3x:.1f}) = {F23x:.1f} \\\\ F_{{23y}} = {F34y:.1f} - ({F3y:.1f}) = {F23y:.1f}",
            fmt_vec(F23x, F23y), "N"
        )
        
        # F12
        F12x = F23x - F2x
        F12y = F23y - F2y
        self._add_step(
            "F12 Kuvveti", 
            r"\vec{F}_{12} = \vec{F}_{23} - \vec{F}_2",
            f"F_{{12x}} = {F23x:.1f} - {F2x:.1f} = {F12x:.1f} \\\\ F_{{12y}} = {F23y:.1f} - {F2y:.1f} = {F12y:.1f}",
            fmt_vec(F12x, F12y), "N"
        )
        
        # --- 6. Input Torque ---
        # T = F23y l2 c2 - F23x l2 s2 - F2y r2 c2 + F2x r2 s2 - T2  (Form from Image)
        # Note: Image uses F2y terms if G2 is not at O2.
        # r2 = O2G2
        r2 = O2G2
        s2, c2 = np.sin(t2), np.cos(t2)
        
        term_F23 = (F23y * l2 * c2) - (F23x * l2 * s2)
        term_F2 =  -(F2y * r2 * c2) + (F2x * r2 * s2) # Note signs in image: -F2y... + F2x...
        
        # T_input equation result
        T_val_Nmm = term_F23 + term_F2 - (T2_inert * 1000.0)
        
        self._add_step(
            "Giriş Torku (T_denge)",
            r"T = F_{23y}l_2 c_2 - F_{23x}l_2 s_2 - F_{2y}r_2 c_2 + F_{2x}r_2 s_2 - T_2",
            f"T = {fmt_rot(T_val_Nmm, 'Nmm')}",
            T_val_Nmm, "Nmm"
        )
        
        # Return State
        return {
            "T_input": T_val_Nmm,
            "F12": (F12x, F12y), "F23": (F23x, F23y), "F34": (F34x, F34y), "F14": (F14x, F14y),
            "F_inertia_2": (F2x, F2y), "F_inertia_3": (F3x, F3y), "F_inertia_4": (F4x, F4y),
            "aG2": (aG2x, aG2y), "aG3": (aG3x, aG3y), "aG4": (aG4x, aG4y)
        }


