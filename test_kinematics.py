import unittest
import numpy as np
from kinematics import FourBarLinkage

class TestKinematics(unittest.TestCase):
    def test_parallelogram(self):
        # Parallelogram linkage: L1=L3, L2=L4
        # Should always be able to rotate fully.
        linkage = FourBarLinkage(L1=10, L2=5, L3=10, L4=5)
        
        # Test theta2 = 90 degrees
        # A should be at (0, 5)
        # O4 is at (10, 0)
        # B should be at (10, 5) -> theta3 = 0, theta4 = 90
        
        joints = linkage.get_joint_positions(90)
        
        self.assertIsNotNone(joints)
        self.assertAlmostEqual(joints['A'][0], 0, places=4)
        self.assertAlmostEqual(joints['A'][1], 5, places=4)
        
        # Check B position
        self.assertAlmostEqual(joints['B'][0], 10, places=4)
        self.assertAlmostEqual(joints['B'][1], 5, places=4)
        
        # Check angles
        self.assertAlmostEqual(joints['theta3'], 0, places=1)
        self.assertAlmostEqual(joints['theta4'], 90, places=1)

    def test_grashof_condition(self):
        # Grashof Crank-Rocker
        # s=2, l=5, p=4, q=4. s+l = 7 < p+q = 8. OK.
        # Fixed L1=4 (p or q) or 5 (l). s=2 must be crank.
        # Let's say L1=5, L2=2, L3=4, L4=4.
        linkage = FourBarLinkage(L1=5, L2=2, L3=4, L4=4)
        linkage.check_grashof() # Just ensure it runs

    
    def test_velocity_acceleration(self):
        # Parallelogram linkage: L1=L3, L2=L4
        # If omega2 = 10, then omega4 should be 10, omega3 should be 0 (pure translation).
        # Alpha2 = 0 implies alpha4 = 0, alpha3 = 0.
        
        linkage = FourBarLinkage(L1=10, L2=5, L3=10, L4=5)
        
        # Test at theta2 = 90
        theta2 = 90
        omega2 = 10.0
        alpha2 = 0.0
        
        joints = linkage.get_joint_positions(theta2)
        theta3 = joints['theta3']
        theta4 = joints['theta4']
        
        w3, w4 = linkage.calculate_velocities(theta2, theta3, theta4, omega2)
        
        self.assertAlmostEqual(w3, 0.0, places=4)
        self.assertAlmostEqual(w4, omega2, places=4)
        
        a3, a4 = linkage.calculate_accelerations(theta2, theta3, theta4, omega2, w3, w4, alpha2)
        
        self.assertAlmostEqual(a3, 0.0, places=4)
        self.assertAlmostEqual(a4, 0.0, places=4)

    
    def test_difficult_configuration(self):
        # User reported case: 400-100-300-300 at 67 degrees failed before.
        linkage = FourBarLinkage(L1=400, L2=100, L3=300, L4=300)
        
        theta2 = 67
        joints = linkage.get_joint_positions(theta2)
        
        self.assertIsNotNone(joints, "Solver failed for difficult configuration")
        
        # Verify closure
        t3 = np.radians(joints['theta3'])
        t4 = np.radians(joints['theta4'])
        t2 = np.radians(theta2)
        
        # Vector loop check
        # L2 + L3 = L1 + L4 (vectors)
        x_closure = 100 * np.cos(t2) + 300 * np.cos(t3) - 400 - 300 * np.cos(t4)
        y_closure = 100 * np.sin(t2) + 300 * np.sin(t3) - 300 * np.sin(t4)
        
        self.assertAlmostEqual(x_closure, 0, places=3)
        self.assertAlmostEqual(y_closure, 0, places=3)

if __name__ == '__main__':
    unittest.main()


