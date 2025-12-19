import unittest
import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.fourbar_solver import FourBarSolver
from solvers.slidercrank_solver import SliderCrankSolver

class TestPDFScenarios(unittest.TestCase):
    
    def test_example_7_fourbar_dynamics(self):
        print("\n--- Testing Example 7 (Four-Bar) ---")
        solver = FourBarSolver(L1=100, L2=50, L3=151, L4=111, assembly_mode=-1) 
        
        kin = solver.solve_kinematics(theta2_deg=120, omega2=95, alpha2=0)
        if kin is None:
             self.fail("Kinematics failed to solve.")
        
        # Verify Kinematics First (PDF Page 32)
        # w3 = 20.86, w4=49.55
        print(f"w3: {kin['omega3']:.2f} (Target: 20.86)")
        print(f"w4: {kin['omega4']:.2f} (Target: 49.55)")
        print(f"a3: {kin['alpha3']:.2f} (Target: ~1001)")
        print(f"a4: {kin['alpha4']:.2f} (Target: ~1121)")
        
        # Dynamics
        # m3=0.5, J3=1.21e-3, BG3=75.5
        # m4=0.4, J4=0.91e-3, O4G4=55.5
        res = solver.solve_dynamics(
            kin, 
            m3=0.5, J3=1.21e-3, BG3=75.5,
            m4=0.4, J4=0.91e-3, O4G4=55.5
        )
        
        T_target = 3481.5 # Nmm
        error = abs(res['T_input'] - T_target)
        print(f"Torque: {res['T_input']:.2f} Nmm (Target: {T_target})")
        
        # Approx 1% tolerance
        self.assertTrue(error < 50, f"Torque calc failed. Got {res['T_input']}, expected {T_target}")

    def test_example_9_slider_dynamics(self):
        print("\n--- Testing Example 9 (Slider-Crank) ---")
        # r=51, l=200, e=0
        solver = SliderCrankSolver(r=51, l=200, e=0, assembly_mode=1)
        
        # Kinematics
        # w2=314
        kin = solver.solve_kinematics(theta2_deg=60, omega2=314, alpha2=0)
        
        # PDF Page 34: w3 = 41.5
        print(f"w3: {kin['omega3']:.2f} (Target: 41.5)")
        
        # Dynamics
        # m3=1.36, J3=0.0102, BG3 = 51 (from 'Gl=51' ?)
        # m4=0.91
        # P = 6300
        
        res = solver.solve_dynamics(
            kin,
            m3=1.36, J3=0.0102, BG3=51,
            m4=0.91, P_gas=-6300
        )
        
        # Target T = 196.7 Nm = 196700 Nmm
        # Note: My solver returns Nmm
        T_target_Nmm = 196700
        T_calc = res['T_input']  
        
        print(f"Torque: {T_calc:.2f} Nmm (Target: {T_target_Nmm})")
        
        # Allow some deviation due to P definition ambiguity or 'Gl' interpretation
        self.assertTrue(abs(T_calc - T_target_Nmm) < 3000, f"Torque mismatch. Got {T_calc}")

if __name__ == '__main__':
    unittest.main()
