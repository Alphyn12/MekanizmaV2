import numpy as np
from .base_solver import BaseSolver, SolutionStep

class SliderCrankSolver(BaseSolver):
    """
    Solver for Slider-Crank Kinematics and Dynamics.
    """
    def __init__(self, r, l, e, assembly_mode=1):
        self.r = r
        self.l = l
        self.e = e
        self.mode = assembly_mode
        self.steps = []

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
        self.steps.append(SolutionStep(title, formula, sub, res_str, res_val, desc))

    def solve_kinematics(self, theta2_deg, omega2, alpha2):
        self.steps = []
        
        t2 = np.radians(theta2_deg)
        Bx = self.r * np.cos(t2)
        By = self.r * np.sin(t2)
        
        self._add_step(
            "B Noktası (Krank Pimi)",
            r"B_x = r \cos \theta_2, \quad B_y = r \sin \theta_2",
            f"B = ({Bx:.1f}, {By:.1f})",
            0, "mm"
        )
        
        # Piston Position C
        # (Cx - Bx)^2 + (Cy - By)^2 = l^2
        # Cy = e
        Cy = self.e
        term = self.l**2 - (self.e - By)**2
        
        if term < 0:
             self.steps.append(SolutionStep("Hata", "", "", "Kilitlendi", 0, "Konum geometrik sınırların dışında."))
             return None
             
        Cx = Bx + self.mode * np.sqrt(term)
        
        self._add_step(
            "Piston Konumu (C)",
            r"C_x = B_x \pm \sqrt{l^2 - (e - B_y)^2}",
            f"C_x = {Cx:.1f}",
            Cx, "mm"
        )
        
        t3 = np.arctan2(Cy - By, Cx - Bx)
        self._add_step(
            "Biyel Açısı (theta3)",
            r"\theta_3 = \arctan(\frac{C_y - B_y}{C_x - B_x})",
            f"\\theta_3 = {np.degrees(t3):.1f}^\\circ",
            np.degrees(t3), "°"
        )
        
        # Velocity
        # Standard analytical:
        # V_P = -r w2 sin(t2) - l w3 sin(t3)
        # w3 = -(r w2 cos(t2)) / (l cos(t3))  (assuming e=0 derivative simplication, but let's use full derivative if e!=0)
        # Full deriv:
        # r cos(t2) + l cos(t3) w3/w2 = V_Cx / w2 ...
        # Loop: r cos(t2) + l cos(t3) = C_x
        # Diff: -r s2 w2 - l s3 w3 = V_P
        #       r s2 + l s3 = C_y = e (const)
        # Diff Y: r c2 w2 + l c3 w3 = 0  => w3 = -(r c2 w2)/(l c3)
        # Valid even for offset e, provided Cy is const.
        
        if abs(np.cos(t3)) < 1e-4:
            w3 = 0
            v_piston = 0
        else:
            w3 = -(self.r * omega2 * np.cos(t2)) / (self.l * np.cos(t3))
            v_piston = -self.r * omega2 * np.sin(t2) - self.l * w3 * np.sin(t3)
            
        self._add_step(
            "Biyel Açısal Hızı (omega3)",
            r"\omega_3 = -\frac{r \omega_2 \cos \theta_2}{l \cos \theta_3}",
            f"{w3:.2f}", w3, "rad/s"
        )
        
        self._add_step(
            "Piston Hızı (V_P)",
            r"V_P = -r \omega_2 \sin \theta_2 - l \omega_3 \sin \theta_3",
            f"{v_piston:.1f}", v_piston, "mm/s"
        )
        
        # Acceleration
        # Diff Y again: r(-s2 w2^2 + c2 a2) + l(-s3 w3^2 + c3 a3) = 0
        # a3 = (r(s2 w2^2 - c2 a2) + l s3 w3^2) / (l c3)
        
        term1 = self.r * (np.sin(t2)*omega2**2 - np.cos(t2)*alpha2)
        term2 = self.l * np.sin(t3)*w3**2
        
        if abs(np.cos(t3)) < 1e-4:
             alpha3 = 0
             a_piston = 0
        else:
             alpha3 = (term1 + term2) / (self.l * np.cos(t3))
             
             # a_P = -r a2 s2 - r w2^2 c2 - l a3 s3 - l w3^2 c3
             a_piston = -self.r * alpha2 * np.sin(t2) - self.r * omega2**2 * np.cos(t2) \
                        - self.l * alpha3 * np.sin(t3) - self.l * w3**2 * np.cos(t3)

        self._add_step(
            "Biyel Açısal İvmesi (A_3)",
            r"\alpha_3 = ...",
            f"{alpha3:.1f}", alpha3, "rad/s^2"
        )
        
        return {
            "theta2": theta2_deg,
            "theta3": np.degrees(t3),
            "t2_rad": t2, "t3_rad": t3,
            "omega2": omega2, "omega3": w3,
            "alpha2": alpha2, "alpha3": alpha3,
            "C": (Cx, Cy), "B": (Bx, By),
            "v_piston": v_piston, "a_piston": a_piston
        }

    def solve_dynamics(self, kin_state, m2, J2, O2G2, m3, J3, BG3, m4, P_gas):
        """
        Solves Slider-Crank Dynamics including Crank Inertia.
        """
        if not kin_state: return None
        
        t3 = kin_state["t3_rad"]
        t2 = kin_state["t2_rad"]
        w3, a3 = kin_state["omega3"], kin_state["alpha3"]
        w2, a2 = kin_state["omega2"], kin_state["alpha2"]
        a_piston = kin_state["a_piston"]
        
        # --- 1. CG Kinematics ---
        
        # G2 (Crank)
        rG2x = O2G2 * np.cos(t2)
        rG2y = O2G2 * np.sin(t2)
        aG2x = -O2G2 * w2**2 * np.cos(t2) - O2G2 * a2 * np.sin(t2)
        aG2y = -O2G2 * w2**2 * np.sin(t2) + O2G2 * a2 * np.cos(t2)
        
        self._add_step(
            "G2 İvmesi", r"\vec{a}_{G2}", f"{np.hypot(aG2x, aG2y):.1f}", np.hypot(aG2x, aG2y), "mm/s^2"
        )
        
        # G3 (Coupler) - Relative to B
        # a_G3 = a_B + a_G3/B
        aBx = -self.r*a2*np.sin(t2) - self.r*w2**2*np.cos(t2)
        aBy =  self.r*a2*np.cos(t2) - self.r*w2**2*np.sin(t2)
        
        rGx = BG3 * np.cos(t3)
        rGy = BG3 * np.sin(t3)
        
        aG3_rel_x = -w3**2 * rGx - a3 * rGy
        aG3_rel_y = -w3**2 * rGy + a3 * rGx
        
        aG3x = aBx + aG3_rel_x
        aG3y = aBy + aG3_rel_y
        
        self._add_step("G3 İvmesi", "", f"{np.hypot(aG3x, aG3y):.1f}", np.sqrt(aG3x**2+aG3y**2), "mm/s^2")
        
        # G4 (Piston)
        aG4x = a_piston
        aG4y = 0
        self._add_step("Piston İvmesi", "", f"{abs(a_piston):.1f}", abs(a_piston), "mm/s^2")

        # --- 2. Inertia Forces ---
        # Link 2
        Fx2 = -m2 * aG2x * 1e-3
        Fy2 = -m2 * aG2y * 1e-3
        Tg2 = -J2 * a2 * 1e-3
        Tg2_Nmm = Tg2 * 1000.0
        
        # Link 3
        Fx3 = -m3 * aG3x * 1e-3
        Fy3 = -m3 * aG3y * 1e-3
        Tg3 = -J3 * a3 * 1e-3 # Nm
        Tg3_Nmm = Tg3 * 1000.0

        # Link 4 (Piston)
        Fx4 = -m4 * aG4x * 1e-3
        Fy4 = -m4 * aG4y * 1e-3 # Should be 0
        
        # --- 3. Force Analysis ---
        
        # Piston Equilibrium (Sum Fx = 0)
        # Forces on Piston: P_gas (External), F34 (Rod Force), F14 (Wall Normal), F_inertia_4
        # F34x + F14x(0) + P_gas + Fx4 = 0
        # F34x = -P_gas - Fx4
        
        F34x = -Fx4 - P_gas 
        
        # Link 3 Equilibrium
        # F34 = -F43 => F43x = -F34x
        F43x = -F34x
        
        # Sum M_B = 0 (about B)
        # (r_C/B x F43) + M_inert_3 (about B) = 0
        # r_C/B = (L cos t3, L sin t3)
        # Cross: L c3 * F43y - L s3 * F43x
        
        # Moment of Inertia Force about B:
        # M_inert_B = Tg3_Nmm + (rGx * Fy3 - rGy * Fx3)
        
        M_inert_3 = Tg3_Nmm + (rGx * Fy3 - rGy * Fx3)
        
        # L c3 * F43y - L s3 * F43x + M_inert_3 = 0
        # F43y = (L s3 * F43x - M_inert_3) / (L c3)
        
        denom = self.l * np.cos(t3)
        if abs(denom) < 1e-4:
            F43y = 0 
        else:
            F43y = (self.l * np.sin(t3) * F43x - M_inert_3) / denom
            
        # Crank Forces
        # F23 + F43 + Fin3 = 0
        F23x = -F43x - Fx3
        F23y = -F43y - Fy3
        
        F32x, F32y = -F23x, -F23y
        
        # Input Torque
        # T + (r_B x F32) + (r_G2 x Fi2) + Tg2 = 0
        M_F32 = (self.r * np.cos(t2) * F32y - self.r * np.sin(t2) * F32x)
        M_Fi2 = (rG2x * Fy2 - rG2y * Fx2)
        
        T_input_Nmm = -(M_F32 + M_Fi2 + Tg2_Nmm)
        
        self._add_step(
            "Giriş Torku", 
            "T = ...",
            f"{T_input_Nmm/1000:.2f}", T_input_Nmm, "Nmm"
        )
        
        # Wall Reaction F14
        # Sum Fy on Piston: F34y + F14y + Fy4 + Weight? (ignoring weight unless m4g added) = 0
        # F14y = -F34y - Fy4
        # F34y = -F43y
        F34y = -F43y
        F14y = -F34y - Fy4
        
        return {
            "T_input": T_input_Nmm,
            "F34": (F34x, F34y), 
            "F23": (F23x, F23y),
            "F12": (-F32x, -F32y),
            "F14": (0, F14y), 
            "aG2": (aG2x, aG2y), "aG3": (aG3x, aG3y), "aG4": (aG4x, aG4y),
            "F_inertia_2": (Fx2, Fy2), "F_inertia_3": (Fx3, Fy3), "F_inertia_4": (Fx4, Fy4)
        }
