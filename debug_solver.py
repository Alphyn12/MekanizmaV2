from kinematics import FourBarLinkage
import numpy as np
from scipy.optimize import fsolve

def debug_case():
    linkage = FourBarLinkage(L1=400, L2=100, L3=300, L4=300)
    theta2 = 67
    theta2_rad = np.radians(theta2)
    
    print(f"Testing L={linkage.L1, linkage.L2, linkage.L3, linkage.L4} at theta2={theta2}")
    
    guesses = [
        [np.radians(0), np.radians(90)],
        [np.radians(45), np.radians(135)],
        [0,0]
    ]
    
    for i, guess in enumerate(guesses):
        print(f"\n--- Guess {i}: {guess} ---")
        try:
            # Replicating the call inside kinematics.py
            solution, info, ier, msg = fsolve(linkage.equations, guess, args=(theta2_rad,), full_output=True)
            print(f"Result ier={ier}")
            print(f"Msg: {msg}")
            print(f"Solution: {solution}")
            
            res = linkage.equations(solution, theta2_rad)
            print(f"Residuals: {res}")
            print(f"Sum Abs Res: {np.sum(np.abs(res))}")
            
            if np.sum(np.abs(res)) < 1e-4:
                print(">>> SUCCESS <<<")
            else:
                print(">>> FAIL <<<")
                
        except Exception as e:
            print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    debug_case()
