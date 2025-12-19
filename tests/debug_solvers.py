
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from solvers.fourbar_solver import FourBarSolver
from solvers.slidercrank_solver import SliderCrankSolver

def debug_ex7():
    print("--- DEBUG EX 7 ---")
    s = FourBarSolver(100, 50, 151, 111, assembly_mode=1)
    kin = s.solve_kinematics(120, 95, 0)
    if not kin:
        print("Kinematics failed!")
        return

    print(f"w3: {kin['omega3']:.2f}, w4: {kin['omega4']:.2f}")
    print(f"a3: {kin['alpha3']:.2f}, a4: {kin['alpha4']:.2f}")
    
    dyn = s.solve_dynamics(kin, 0.5, 1.21e-3, 75.5, 0.4, 0.91e-3, 55.5)
    print(f"T_input: {dyn['T_input']:.2f}")
    print(f"F34: {dyn['F34']}")
    print(f"F23: {dyn['F23']}")

def debug_ex9():
    print("\n--- DEBUG EX 9 ---")
    s = SliderCrankSolver(51, 200, 0, assembly_mode=1)
    # Ex 9 Inputs
    t2 = 60
    w2 = 314
    a2 = 0
    kin = s.solve_kinematics(t2, w2, a2)
    
    print(f"w3: {kin['omega3']:.2f}")
    print(f"a_piston: {kin['a_piston']:.2f}")
    
    # Dynamics Inputs
    m3 = 1.36
    J3 = 0.0102
    BG3 = 51
    m4 = 0.91
    P = 6300
    
    dyn = s.solve_dynamics(kin, m3, J3, BG3, m4, P)
    print(f"T_input: {dyn['T_input']:.2f}")

if __name__ == "__main__":
    debug_ex7()
    debug_ex9()
