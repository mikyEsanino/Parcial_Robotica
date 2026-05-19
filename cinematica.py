import sympy as sp
import numpy as np
from sympy import symbols, diff, sin, cos
from sympy.physics.mechanics import dynamicsymbols

sp.init_printing(use_unicode=True)

# definimos rotacion y traslacion
def trans(x,y,z):
  M = sp.Matrix([[1, 0, 0, x],[0, 1, 0, y],[0, 0, 1, z],[0, 0, 0, 1]])
  return M

def rotx(ang):
  M = sp.Matrix([[1,0,0,0],[0, sp.cos(ang), -sp.sin(ang) , 0],[0, sp.sin(ang), sp.cos(ang),0 ],[0,0,0,1]])
  return M

def roty(ang):
  M = sp.Matrix([[ sp.cos(ang), 0, sp.sin(ang), 0],[0,1,0,0],[-sp.sin(ang), 0, sp.cos(ang),0],[0,0,0,1]])
  return M

def rotz(ang):
  M = sp.Matrix([[ sp.cos(ang), -sp.sin(ang), 0, 0],[sp.sin(ang), sp.cos(ang),0, 0],[0,0,1,0], [0,0,0,1]])
  return M

# definimos DH
def DH(theta,d,a,alpha):
  tr = rotz(theta)*trans(0,0,d)*trans(a,0,0)*rotx(alpha)
  return tr

# definimos las variables simbolicas
θ1, θ2, θ3, θ4, θ5, θ6 = sp.symbols('θ1 θ2 θ3 θ4 θ5 θ6')

# armamos las 6 matrices de transformación con los parametros DH
A1 = DH(θ1, 131.56, 0, 90) #base
sp.pprint(A1)

A2 = DH(θ2, 0, 110.4, 0) #hombro
sp.pprint(A2)

A3 = DH(θ3, 0, 96, 0) #codo
sp.pprint(A3)

A4 = DH(θ4, 66.39, 0, -90) #muñeca 1
sp.pprint(A4)

A5 = DH(θ5, 73.18, 0, 90) #muñeca 2
sp.pprint(A5)

A6 = DH(θ6, 48.6, 0, 0) #gripper
sp.pprint(A6)

class ForwardKinematics:
    def __init__(self):
        """Inicializa las longitudes físicas de los eslabones (en mm)"""
        self.d1 = 131.56
        self.a2 = 110.4
        self.a3 = 96.0
        self.d4 = 66.39
        self.d5 = 73.18
        self.d6 = 48.6

    def _dh_matrix(self, theta, d, a, alpha):
        """Función auxiliar numérica usando NumPy"""
        rad_theta = np.radians(theta)
        rad_alpha = np.radians(alpha)
        
        ct = np.cos(rad_theta)
        st = np.sin(rad_theta)
        ca = np.cos(rad_alpha)
        sa = np.sin(rad_alpha)
        
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d     ],
            [0,   0,        0,       1     ]
        ])

    def compute_fk(self, joints):
        """P2: Calcula la matriz de transformación homogénea T_0_6 final"""
        q1, q2, q3, q4, q5, q6 = joints
        
        A1_num = self._dh_matrix(q1, self.d1, 0, 90)
        A2_num = self._dh_matrix(q2, 0, self.a2, 0)
        A3_num = self._dh_matrix(q3, 0, self.a3, 0)
        A4_num = self._dh_matrix(q4, self.d4, 0, -90)
        A5_num = self._dh_matrix(q5, self.d5, 0, 90)
        A6_num = self._dh_matrix(q6, self.d6, 0, 0)
        
        T_0_6 = A1_num @ A2_num @ A3_num @ A4_num @ A5_num @ A6_num
        return T_0_6