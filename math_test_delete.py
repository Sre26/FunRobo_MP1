import numpy as np
PI = 3.1415

deg_angles = [0, 90, 180, 360]
rad_ang1 = deg_angles * PI/180
rad_ang2 = np.radians(deg_angles) 
rad_ang3 = [np.radians(ang) for ang in deg_angles]
rad_ang4 = np.deg2rad(deg_angles) 
rad_ang5 = [np.deg2rad(ang) for ang in deg_angles]

ref = np.array([0, PI/2, PI, 2*PI])

print("*PI/180 ", rad_ang1)
print("rad() ", rad_ang2)
print("rad() in [] ", rad_ang3)
print("d2r() ", rad_ang4)
print("d2r() in [] ", rad_ang5)
print("correct ans: ", ref)
