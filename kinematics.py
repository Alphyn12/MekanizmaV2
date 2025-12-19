import numpy as np
import streamlit as st
from solvers.fourbar_solver import FourBarSolver
from solvers.slidercrank_solver import SliderCrankSolver

class FourBarLinkage:
    def __init__(self, L1, L2, L3, L4):
        self.L1 = L1 # Fixed, Ground
        self.L2 = L2 # Crank
        self.L3 = L3 # Coupler
        self.L4 = L4 # Rocker
        
    def check_grashof(self):
        links = sorted([self.L1, self.L2, self.L3, self.L4])
        s, l, p, q = links[0], links[3], links[1], links[2]
        if s + l <= p + q:
            return "Grashof (Döner)"
        else:
             return "Non-Grashof (Dönmez/Sallanır)"

    def get_position(self, theta2_deg, assembly_mode=1):
        """
        Calculates joint positions A and B.
        assembly_mode: +1 for Standard/Up, -1 for Crossed/Down
        Returns dict with O2, A, B, O4, theta3, theta4.
        """
        theta2 = np.radians(theta2_deg)
        
        # O2 at origin
        O2 = (0, 0)
        O4 = (self.L1, 0)
        
        # Point A (Crank Tip)
        Ax = self.L2 * np.cos(theta2)
        Ay = self.L2 * np.sin(theta2)
        A = (Ax, Ay)
        
        # Point B
        dx = O4[0] - Ax
        dy = O4[1] - Ay
        d_sq = dx**2 + dy**2
        d = np.sqrt(d_sq)
        
        if d > self.L3 + self.L4 or d < abs(self.L3 - self.L4) or d == 0:
            return None
            
        a = (self.L3**2 - self.L4**2 + d**2) / (2 * d)
        h = np.sqrt(max(0, self.L3**2 - a**2))
        
        x2 = Ax + a * (O4[0] - Ax) / d
        y2 = Ay + a * (O4[1] - Ay) / d
        
        # Use assembly_mode to flip the triangle height direction
        # h needs to be multiplied by assembly_mode
        # B calculation involves +/- linear combination perpendicular to d line
        
        Bx = x2 + (assembly_mode * h) * (O4[1] - Ay) / d
        By = y2 - (assembly_mode * h) * (O4[0] - Ax) / d
        B = (Bx, By)
        
        theta3 = np.degrees(np.arctan2(By - Ay, Bx - Ax))
        theta4 = np.degrees(np.arctan2(By, Bx - self.L1))

        return {
            "O2": O2, "A": A, "B": B, "O4": O4,
            "theta3": theta3, "theta4": theta4
        }
        
    def get_velocity(self, theta2_deg, omega2, assembly_mode=1):
        pos = self.get_position(theta2_deg, assembly_mode)
        if not pos: return None, None
        
        t2 = np.radians(theta2_deg)
        t3 = np.radians(pos['theta3'])
        t4 = np.radians(pos['theta4'])
        
        denom = np.sin(t3 - t4)
        if abs(denom) < 1e-5: return 0, 0
        
        w3 = omega2 * (self.L2 * np.sin(t4 - t2)) / (self.L3 * denom)
        w4 = omega2 * (self.L2 * np.sin(t3 - t2)) / (self.L4 * denom)
        return w3, w4

    def get_acceleration(self, theta2_deg, omega2, alpha2, assembly_mode=1):
        pos = self.get_position(theta2_deg, assembly_mode)
        if not pos: return None, None
        
        w3, w4 = self.get_velocity(theta2_deg, omega2, assembly_mode)
        if w3 is None: return None, None
        
        t2 = np.radians(theta2_deg)
        t3 = np.radians(pos['theta3'])
        t4 = np.radians(pos['theta4'])
        
        A = -self.L2 * alpha2 * np.sin(t2) - self.L2 * omega2**2 * np.cos(t2) - self.L3 * w3**2 * np.cos(t3) + self.L4 * w4**2 * np.cos(t4)
        B = self.L2 * alpha2 * np.cos(t2) - self.L2 * omega2**2 * np.sin(t2) - self.L3 * w3**2 * np.sin(t3) + self.L4 * w4**2 * np.sin(t4)
        
        det = self.L3 * self.L4 * np.sin(t4 - t3)
        if abs(det) < 1e-5: return 0, 0
        
        alpha3 = (A * (-self.L4 * np.cos(t4)) - B * (self.L4 * np.sin(t4))) / det
        alpha4 = ((-self.L3 * np.sin(t3)) * B - (self.L3 * np.cos(t3)) * A) / det
        
        return alpha3, alpha4

    def calculate_transmission_angle(self, theta2, assembly_mode=1):
        pos = self.get_position(theta2, assembly_mode)
        if not pos: return None
        t3, t4 = pos['theta3'], pos['theta4']
        angle = abs(t3 - t4)
        while angle > 180: angle -= 360
        while angle < -180: angle += 360
        return abs(angle)

    def get_coupler_point(self, theta2, r_p, delta_deg, assembly_mode=1):
        pos = self.get_position(theta2, assembly_mode)
        if not pos: return None
        
        A = pos["A"]
        # theta3 calculated in get_position
        theta3_rad = np.radians(pos["theta3"])
        delta_rad = np.radians(delta_deg)
        
        Px = A[0] + r_p * np.cos(theta3_rad + delta_rad)
        Py = A[1] + r_p * np.sin(theta3_rad + delta_rad)
        return (Px, Py)

    def calculate_instant_centers(self, theta2, assembly_mode=1):
        """
        Calculates coordinates of primary Instant Centers I13 and I24 using Kennedy's Theorem.
        I12=(0,0), I14=(L1,0), I23=A, I34=B
        I13: Intersection of Line(O2, A) and Line(O4, B)
        I24: Intersection of Line(O2, O4) [X-axis] and Line(A, B) [Coupler line]
        """
        pos = self.get_position(theta2, assembly_mode)
        if not pos: return None
        
        A = pos["A"]
        B = pos["B"]
        O2 = pos["O2"]
        O4 = pos["O4"]
        
        # --- I13 Calculation ---
        # Line 1 (O2-A): y = m2*x (passes through 0,0)
        # Line 2 (O4-B): y - 0 = m4*(x - L1) => y = m4*x - m4*L1
        
        # Check for vertical lines to avoid infinite slope
        I13 = None
        m2_inf = abs(A[0]) < 1e-5
        m4_inf = abs(B[0] - self.L1) < 1e-5
        
        if m2_inf and m4_inf:
            I13 = None # Parallel vertical lines (at infinity)
        elif m2_inf:
            # Line 1 is x=0. Intersection with Line 2 at x=0.
            # y = m4*0 - m4*L1 = -m4*L1
            m4 = (B[1]) / (B[0] - self.L1)
            I13 = (0, -m4 * self.L1)
        elif m4_inf:
            # Line 2 is x = L1. Intersection with Line 1 at x=L1.
            m2 = A[1] / A[0]
            I13 = (self.L1, m2 * self.L1)
        else:
            m2 = A[1] / A[0]
            m4 = B[1] / (B[0] - self.L1)
            
            if abs(m2 - m4) < 1e-5:
                I13 = None # Parallel lines (at infinity)
            else:
                # m2*x = m4*x - m4*L1  => x(m2-m4) = -m4*L1 => x = m4*L1 / (m4-m2)
                I13_x = (m4 * self.L1) / (m4 - m2)
                I13_y = m2 * I13_x
                I13 = (I13_x, I13_y)
                
        # --- I24 Calculation ---
        # Line 1 (O2-O4): y = 0
        # Line 2 (A-B): y - Ay = m3*(x - Ax)
        # Intersection y=0 => -Ay = m3*(x - Ax) => x - Ax = -Ay/m3 => x = Ax - Ay/m3
        
        I24 = None
        if abs(B[0] - A[0]) < 1e-5:
            # Vertical coupler. x = Ax. y=0 intersection is (Ax, 0)
            I24 = (A[0], 0)
        else:
            m3 = (B[1] - A[1]) / (B[0] - A[0])
            if abs(m3) < 1e-5:
                # Horizontal coupler. Parallel to Ground (y=0). I24 at infinity.
                I24 = None
            else:
                I24_x = A[0] - A[1]/m3
                I24 = (I24_x, 0)
                
        return {"I13": I13, "I24": I24}

    def calculate_mechanical_advantage(self, theta2, assembly_mode=1):
        """
        Calculates Mechanical Advantage (MA) based on Instant Centers (Ch 5.3).
        MA = T_out / T_in = |I14-I24| / |I12-I24|
        """
        ics = self.calculate_instant_centers(theta2, assembly_mode)
        if not ics or not ics["I24"]: return 0.0 # Undefined or zero
        
        I24 = ics["I24"]
        I12 = (0, 0)
        I14 = (self.L1, 0)
        
        # These are all on the x-axis (y=0) since I24 is intersection with ground line.
        # Distance is just abs difference in x coordinates.
        dist_I14_I24 = abs(I14[0] - I24[0])
        dist_I12_I24 = abs(I12[0] - I24[0])
        
        if dist_I12_I24 < 1e-4: return 999.0 # Infinite MA (Toggle position)
        
        ma = dist_I14_I24 / dist_I12_I24
        return ma


class SliderCrankMechanism:
    def __init__(self, r, l, e):
        self.r = r 
        self.l = l 
        self.e = e 
        
    def get_position(self, theta2_deg, assembly_mode=1):
        # Slider crank usually has 2 solutions for theta3 if offset is used
        # Standard analytical method:
        # B = (r cos t2, r sin t2)
        # (Cx - Bx)^2 + (Cy - By)^2 = l^2
        # Cy = e
        # (Cx - Bx)^2 = l^2 - (e - By)^2
        # Cx = Bx +/- sqrt(...)
        # The +/- corresponds to assembly mode!
        # Standard is usually + (Slider to the right)
        
        theta2 = np.radians(theta2_deg)
        O2 = (0, 0)
        
        Bx = self.r * np.cos(theta2)
        By = self.r * np.sin(theta2)
        B = (Bx, By)
        
        term = self.l**2 - (self.e - By)**2
        if term < 0: return None
        
        # Apply assembly mode here for Cx
        Cx = Bx + assembly_mode * np.sqrt(term)
        Cy = self.e
        C = (Cx, Cy)

        theta3 = np.degrees(np.arctan2(Cy - By, Cx - Bx))
        return {
            "O2": O2, "B": B, "C": C,
            "theta3": theta3, "v_piston": 0, "a_piston": 0
        }

    def get_velocity(self, theta2_deg, omega2, assembly_mode=1):
        pos = self.get_position(theta2_deg, assembly_mode)
        if not pos: return None, None
        
        theta2 = np.radians(theta2_deg)
        theta3 = np.radians(pos['theta3'])
        
        if abs(np.cos(theta3)) < 1e-4: return 0, 0
        
        w3 = -(self.r * omega2 * np.cos(theta2)) / (self.l * np.cos(theta3))
        v_piston = -self.r * omega2 * np.sin(theta2) - self.l * w3 * np.sin(theta3)
        return w3, v_piston
        
    def get_acceleration(self, theta2_deg, omega2, alpha2, assembly_mode=1):
        pos = self.get_position(theta2_deg, assembly_mode)
        if not pos: return None, None
        
        theta2 = np.radians(theta2_deg)
        theta3 = np.radians(pos['theta3'])
        w3, vp = self.get_velocity(theta2_deg, omega2, assembly_mode)
        
        if abs(np.cos(theta3)) < 1e-4: return 0, 0

        term1 = self.r * omega2**2 * np.sin(theta2)
        term2 = self.l * w3**2 * np.sin(theta3)
        term3 = self.r * alpha2 * np.cos(theta2)
        
        alpha3 = (term1 + term2 - term3) / (self.l * np.cos(theta3))
        
        a_piston = -self.r * alpha2 * np.sin(theta2) - \
                   self.r * omega2**2 * np.cos(theta2) - \
                   self.l * alpha3 * np.sin(theta3) - \
                   self.l * w3**2 * np.cos(theta3)
                   
        return alpha3, a_piston

    def get_coupler_point(self, theta2, r_p, delta_deg, assembly_mode=1):
        pos = self.get_position(theta2, assembly_mode)
        if not pos: return None
        
        B = pos["B"] # Crank Tip
        theta3_rad = np.radians(pos["theta3"])
        delta_rad = np.radians(delta_deg)
        
        Px = B[0] + r_p * np.cos(theta3_rad + delta_rad)
        Py = B[1] + r_p * np.sin(theta3_rad + delta_rad)
        return (Px, Py)

    def calculate_kinematics(self, theta2, omega2, alpha2, assembly_mode=1):
        p = self.get_position(theta2, assembly_mode)
        if not p: return None
        w3, vp = self.get_velocity(theta2, omega2, assembly_mode)
        a3, ap = self.get_acceleration(theta2, omega2, alpha2, assembly_mode)
        return {
            "theta3": p['theta3'],
            "omega3": w3, "v_piston": vp,
            "alpha3": a3, "a_piston": ap
        }

class InvertedSliderCrankMechanism:
    def __init__(self, r1, r2, r4_init=None): 
        # r1=Ground (O2-O4 distance), r2=Crank, r4_init=Optional visual length
        self.r1 = r1 
        self.r2 = r2
        
    def get_position(self, theta2_deg, assembly_mode=1):
        # Vertical alignment assumption for standard Whitworth/Quick Return? 
        # Or Horizontal? Let's use Horizontal (O2 at 0,0; O4 at r1,0) to start.
        # This matches standard horizontal slider crank coordinate systems usually.
        # O2 = (0,0), O4 = (r1, 0)
        
        theta2 = np.radians(theta2_deg)
        O2 = (0, 0)
        O4 = (self.r1, 0) 
        
        # Point A (Crank Tip)
        Ax = self.r2 * np.cos(theta2)
        Ay = self.r2 * np.sin(theta2)
        A = (Ax, Ay)
        
        # Point B (Rocker Tip - visualization only)
        # B is on the line O4-A
        dx = Ax - O4[0]
        dy = Ay - O4[1]
        
        r4_current = np.sqrt(dx**2 + dy**2)
        theta4 = np.arctan2(dy, dx)
        
        # Visual length of rocker
        L_vis = self.r2 + self.r1 + 50
        Bx = O4[0] + L_vis * np.cos(theta4)
        By = O4[1] + L_vis * np.sin(theta4)
        B = (Bx, By)
        
        return {
            "O2": O2, "O4": O4, "A": A, "B": B,
            "theta3": 0,
            "theta4": np.degrees(theta4),
            "r4_val": r4_current
        }

    def get_velocity(self, theta2_deg, omega2, assembly_mode=1):
        pos = self.get_position(theta2_deg)
        t2 = np.radians(theta2_deg)
        t4 = np.radians(pos['theta4'])
        r4 = pos['r4_val']
        
        # Analytical Derivation:
        # V_A (crank) = V_A (rocker point) + V_slip
        # w4 = (w2 * r2 * cos(t2 - t4)) / r4
        if r4 == 0: return 0, 0
        w4 = (omega2 * self.r2 * np.cos(t2 - t4)) / r4
        
        # r4_dot = w2 * r2 * sin(t4 - t2)
        r4_dot = -omega2 * self.r2 * np.sin(t2 - t4)  # equivalent sin(t4-t2)
        
        return 0, w4 # omega3 is 0 or undefined, return w4 as secondary

    def get_acceleration(self, theta2_deg, omega2, alpha2, assembly_mode=1):
        pos = self.get_position(theta2_deg)
        t2 = np.radians(theta2_deg)
        t4 = np.radians(pos['theta4'])
        r4 = pos['r4_val']
        
        if r4 == 0: return 0, 0
            
        _, w4 = self.get_velocity(theta2_deg, omega2)
        r4_dot = -omega2 * self.r2 * np.sin(t2 - t4)
        
        # alpha4 formula derived from Coriolis equation
        term1 = - (omega2**2) * self.r2 * np.sin(t2 - t4)
        term2 = alpha2 * self.r2 * np.cos(t2 - t4)
        coriolis = 2 * w4 * r4_dot
        
        alpha4 = (term1 + term2 - coriolis) / r4
        
        return 0, alpha4

    def calculate_transmission_angle(self, theta2, assembly_mode=1):
        # Angle between Crank (Force) and Rocker (Motion perp)
        # Optimal is 90 deg difference implies 0 force transmission? No.
        # "Force applied at A perpendicular to Crank OA" -> Torque T2
        # Force F transmitted to slot wall is perpendicular to slot wall (Rocker).
        # We want component of F perpendicular to Rocker (to produce torque T4) to be max.
        # Actually, slider force N is perp to Rocker.
        # Driving force F is perp to Crank.
        # N = F / cos(mu)? 
        # Let's stick to geometric angle |theta2 - theta4|.
        # If theta2 = theta4 (aligned), force on slider is perp to rocker ==> BEST torque T4.
        # If theta2 perp theta4, force is along rocker ==> ZERO torque T4.
        # So "Quality" of transmission: 100% at 0 deg diff, 0% at 90 deg diff.
        # "Transmission Angle" mu is usually defined such that 90 is ideal.
        # So let mu = 90 - |angle_diff| ??
        # Or just return the angle difference and interpret later.
        # Let's return the "Deviation from ideal 90" equivalent.
        # Ideal: 0 deg diff. Worst: 90 deg diff.
        # Standard mu: Ideal 90, Worst 0.
        # So mu_equiv = 90 - abs( |t2-t4| - 0 ) ? No.
        # diff = abs(t2-t4). 
        # If diff=0, we want 90. If diff=90, we want 0.
        # mu = 90 - (diff % 90) ? No.
        # mu = |cos(diff)| * 90 ? No.
        # mu = 90 - abs(sin(diff)*90) ??
        # Let's simply return abs(theta2 - theta4) and let UI interpret "0 is best".
        # But `analyze_cycle` expects mu for min/max stats.
        
        pos = self.get_position(theta2)
        t2 = theta2 % 360
        t4 = pos["theta4"] % 360
        diff = abs(t2 - t4)
        if diff > 180: diff = 360 - diff
        return diff

    def calculate_kinematics(self, theta2, omega2, alpha2, assembly_mode=1):
        # wrapper
        p = self.get_position(theta2)
        _, w4 = self.get_velocity(theta2, omega2)
        _, a4 = self.get_acceleration(theta2, omega2, alpha2)
        return {
            "theta3": 0, "omega3": 0, "alpha3": 0,
            "theta4": p['theta4'], "omega4": w4, "alpha4": a4,
            "O2": p['O2'], "O4": p['O4'], "A": p['A'], "B": p['B'] # Add cords for solver compat
        }

    solve_kinematics = calculate_kinematics

def analyze_cycle(mechanism, mech_type, omega2, alpha2, assembly_mode=1):
    theta_range = np.linspace(0, 360, 361) # 0 to 360
    
    # Lists to store timeseries
    w3_l, out_v_l = [], []
    a3_l, out_a_l = [], []
    mu_l = []
    ma_l = []
    
    # Store position data too (Fixed KeyError)
    t3_l, t4_l = [], []
    j_A, j_B, j_C = [], [], []

    for t in theta_range:
        # 1. Position Analysis (Common)
        pos = mechanism.get_position(t, assembly_mode)
        if pos:
             # Extract
             t3 = pos.get('theta3', 0)
             t4 = pos.get('theta4', 0)
             A = pos.get('A')
             B = pos.get('B')
             C = pos.get('C')
             
             t3_l.append(t3)
             t4_l.append(t4)
             j_A.append(A if A else (0,0))
             j_B.append(B if B else (0,0))
             j_C.append(C if C else (0,0))
        else:
             t3_l.append(None)
             t4_l.append(None)
             j_A.append(None); j_B.append(None); j_C.append(None)

        # 2. Kinematic Analysis
        if mech_type == "Dört Çubuk Mekanizması":
            w3, w4 = mechanism.get_velocity(t, omega2, assembly_mode)
            a3, a4 = mechanism.get_acceleration(t, omega2, alpha2, assembly_mode)
            mu = mechanism.calculate_transmission_angle(t, assembly_mode)
            ma = mechanism.calculate_mechanical_advantage(t, assembly_mode)
            
            # Helper to append None or val
            w3_l.append(w3 if w3 is not None else np.nan)
            out_v_l.append(w4 if w4 is not None else np.nan)
            a3_l.append(a3 if a3 is not None else np.nan)
            out_a_l.append(a4 if a4 is not None else np.nan)
            mu_l.append(mu if mu is not None else np.nan)
            ma_l.append(ma if ma is not None else np.nan)
                 
        elif mech_type == "Kol-Kızak (Whitworth) Mekanizması":
            _, w4 = mechanism.get_velocity(t, omega2, assembly_mode)
            _, a4 = mechanism.get_acceleration(t, omega2, alpha2, assembly_mode)
            mu = mechanism.calculate_transmission_angle(t, assembly_mode)
            
            w3_l.append(np.nan) # No Coupler Angular Velocity relevant
            out_v_l.append(w4 if w4 is not None else np.nan)
            a3_l.append(np.nan)
            out_a_l.append(a4 if a4 is not None else np.nan)
            mu_l.append(mu if mu is not None else np.nan)
            ma_l.append(np.nan)
            
        else: # Slider Crank
            w3, vp = mechanism.get_velocity(t, omega2, assembly_mode)
            a3, ap = mechanism.get_acceleration(t, omega2, alpha2, assembly_mode)
            
            w3_l.append(w3 if w3 is not None else np.nan)
            out_v_l.append(vp if vp is not None else np.nan)
            a3_l.append(a3 if a3 is not None else np.nan)
            out_a_l.append(ap if ap is not None else np.nan)
            mu_l.append(np.nan) 
            ma_l.append(np.nan)
                 
    # Convert to arrays for easy stats
    w3_arr = np.array(w3_l, dtype=float)
    out_v_arr = np.array(out_v_l, dtype=float)
    a3_arr = np.array(a3_l, dtype=float)
    out_a_arr = np.array(out_a_l, dtype=float)
    mu_arr = np.array(mu_l, dtype=float)
    ma_arr = np.array(ma_l, dtype=float)
    
    # Helper index finder
    def get_max_info(arr):
        if np.all(np.isnan(arr)): return 0, 0
        idx = np.nanargmax(np.abs(arr)) # Index of max absolute interaction
        return arr[idx], theta_range[idx]

    def get_min_info(arr): # For Mu min
        if np.all(np.isnan(arr)): return 0, 0
        idx = np.nanargmin(arr)
        return arr[idx], theta_range[idx]
        
    def get_abs_max_info(arr): 
        if np.all(np.isnan(arr)): return 0, 0
        idx = np.nanargmax(arr)
        return arr[idx], theta_range[idx]

    # Calculate Stats
    max_w3, max_w3_ang = get_max_info(w3_arr)
    max_out_v, max_out_v_ang = get_max_info(out_v_arr)
    max_a3, max_a3_ang = get_max_info(a3_arr)
    max_out_a, max_out_a_ang = get_max_info(out_a_arr)
    
    min_mu, min_mu_ang = get_min_info(mu_arr)
    max_mu, max_mu_ang = get_abs_max_info(mu_arr)
    
    time_ratio = None
    if mech_type == "Kol-Kızak (Whitworth) Mekanizması":
         # Count frames with positive vs negative velocity
         # Only valid if it oscillates (Quick Return). If Whitworth (Rotary), velocity might not change sign (always +).
         # Whitworth: usually varying speed but single direction.
         # Quick Return (Crank-Rocker inversion): Oscillates.
         # Let's check sign changes.
         signs = np.sign(out_v_arr)
         pos_count = np.sum(signs > 0)
         neg_count = np.sum(signs < 0)
         if pos_count > 0 and neg_count > 0:
             # Ratio of larger to smaller
             t1 = max(pos_count, neg_count)
             t2 = min(pos_count, neg_count)
             time_ratio = t1 / t2
         else:
             time_ratio = "Sürekli Dönüş" # Rotary output

    stats = {
        # Common
        "max_w3": max_w3,
        "max_w3_angle": max_w3_ang,
        "max_alpha3": max_a3,
        "max_alpha3_angle": max_a3_ang,
        
        # Output V/A (Aliased)
        "max_w4": max_out_v, 
        "max_w4_angle": max_out_v_ang,
        "max_v_piston": max_out_v,
        "max_v_piston_angle": max_out_v_ang,
        
        "max_alpha4": max_out_a, 
        "max_alpha4_angle": max_out_a_ang,
        "max_a4": max_out_a, 
        "max_a4_angle": max_out_a_ang,
        "max_a_piston": max_out_a,
        "max_a_piston_angle": max_out_a_ang,
        
        # Transmission Angle
        "min_mu": min_mu,
        "min_mu_angle": min_mu_ang,
        "max_mu": max_mu,
        "max_mu_angle": max_mu_ang,
        
        # Special
        "time_ratio": time_ratio
    }
    
    # Replace NaNs with None for JSON/Dict compatibility if desired, or keep NaNs
    # But for display, maybe clean?
    # Keeping logic simple
    w3_l = [None if np.isnan(x) else x for x in w3_l]
    out_v_l = [None if np.isnan(x) else x for x in out_v_l]
    a3_l = [None if np.isnan(x) else x for x in a3_l]
    out_a_l = [None if np.isnan(x) else x for x in out_a_l]
    mu_l = [None if np.isnan(x) else x for x in mu_l]
    ma_l = [None if np.isnan(x) else x for x in ma_l]

    data = {
        "theta2": theta_range,
        "theta3": t3_l,
        "theta4": t4_l,
        "omega3": w3_l,
        "output_vel": out_v_l,
        "alpha3": a3_l,
        "output_acc": out_a_l,
        "mu": mu_l,
        "ma": ma_l,
        "joints": {
             "A": j_A,
             "B": j_B,
             "C": j_C
        }
    }
    
    return stats, data

@st.cache_data
def calculate_full_cycle(mech_type, omega2, alpha2, assembly_mode, dyn_params, L1, L2, L3, L4, r, l, e, r1, r2):
    """
    Cached function to perform full cycle analysis (Kinematics + Dynamics).
    Optimized to run only when inputs change.
    """
    # 1. Setup Mechanism & Solver
    mechanism = None
    solver = None
    
    if mech_type == "Dört Çubuk Mekanizması":
        mechanism = FourBarLinkage(L1, L2, L3, L4)
        solver = FourBarSolver(L1, L2, L3, L4, assembly_mode=assembly_mode)
    elif mech_type == "Krank-Biyel Mekanizması":
        mechanism = SliderCrankMechanism(r, l, e)
        solver = SliderCrankSolver(r, l, e, assembly_mode=assembly_mode)
    else: # Whitworth / Kol-Kızak
        mechanism = InvertedSliderCrankMechanism(r1, r2)
        solver = mechanism # Self-solving for kinematics

    # 2. Run Kinematic Cycle
    stats, c_data = analyze_cycle(mechanism, mech_type, omega2, alpha2, assembly_mode)

    # 3. Run Dynamic Cycle (If applicable)
    if dyn_params and solver and mech_type in ["Dört Çubuk Mekanizması", "Krank-Biyel Mekanizması"]:
        t_list, f12_l, f23_l, f34_l, f14_l = [], [], [], [], []
        
        # Pre-calculation loop
        for i in range(len(c_data['theta2'])):
            t2r = np.radians(c_data['theta2'][i])
            
            # Construct State Dict
            ks = {
                "t2_rad": t2r,
                "t3_rad": np.radians(c_data['theta3'][i]),
                # Handle None/NaN in theta4
                "t4_rad": np.radians(c_data.get('theta4', [0]*len(c_data['theta3']))[i] if c_data.get('theta4') and c_data['theta4'][i] is not None else 0),
                "omega2": omega2, "alpha2": alpha2,
                "omega3": c_data['omega3'][i], 
                "alpha3": c_data['alpha3'][i],
                "omega4": c_data.get('output_vel', [0]*len(c_data['theta3']))[i], 
                "alpha4": c_data.get('output_acc', [0]*len(c_data['theta3']))[i],
                "B": c_data['joints']['B'][i] if 'B' in c_data['joints'] else (0,0)
            }
            if mech_type == "Krank-Biyel Mekanizması" and 'C' in c_data['joints']:
                 ks["C"] = c_data['joints']['C'][i]

            try:
                dyn = solver.solve_dynamics(ks, **dyn_params)
                
                if dyn:
                    t_list.append(dyn['T_input']/1000.0) # Nmm -> Nm ? No, kept as is usually, but UI expects Nm?
                    # App.py original code divided by 1000.0 for T_input. 
                    # Let's check App.py line 188: t_list.append(dyn['T_input']/1000.0)
                    
                    f12_l.append(np.hypot(*dyn['F12'])); f23_l.append(np.hypot(*dyn['F23']))
                    f34_l.append(np.hypot(*dyn['F34'])); f14_l.append(np.hypot(*dyn['F14']))
                else:
                    t_list.append(0); f12_l.append(0); f23_l.append(0); f34_l.append(0); f14_l.append(0)
            except Exception:
                t_list.append(0); f12_l.append(0); f23_l.append(0); f34_l.append(0); f14_l.append(0)

        c_data['T_input'] = t_list
        c_data['F12'] = f12_l; c_data['F23'] = f23_l; c_data['F34'] = f34_l; c_data['F14'] = f14_l
        
    return stats, c_data
