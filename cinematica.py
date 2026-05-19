import sympy as sp
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