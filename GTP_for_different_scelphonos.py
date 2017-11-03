import numpy as np
from numpy import sqrt as sqrt
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sympy as sym
from sympy import lambdify
from matplotlib import cm
from sympy.solvers import solve
import subprocess
from itertools import chain

# Function to fit:
def func(X, a0, a1, a2, a3, a4, a5):
     x, y = X
     return a0 + a1*y + a2*x + a3*y**2 + a4*x**2  + a5*x*y 

# Load the raw T, P, G data: 17 volumes calcite I and 4 volumes calcite II, for instance:

# Calcite I: SCELPHONO 2x2x2
y_data_small, z_data_small, x_data_small  = np.loadtxt('/home/david/Trabajo/structures/SCRIPT_ON_ALL_THE_PHASES/Calcite_I_over_17_volumes/SHRINK_3_3/crystal17_Pcrystal/volumes/grabbing_exact_value_of_freqs/solid_1__xyz_sorted_as_P_wise.dat').T

# Calcite II: SCELPHONO 121_1-21_210
y_data_2_small, z_data_2_small, x_data_2_small  = np.loadtxt('/home/david/Trabajo/structures/SCRIPT_ON_ALL_THE_PHASES/Calcite_II_correct_description/scelphono_121_1-21_210__SHRINK_3_3/new/solid_1__xyz_sorted_as_P_wise.dat').T

# Calcite I: SCELPHONO 4x4x4
y_data, z_data, x_data  = np.loadtxt('/home/david/Trabajo/structures/SCRIPT_ON_ALL_THE_PHASES/Calcite_I_over_17_volumes/solid_1__xyz_sorted_as_P_wise.dat').T

# Calcite II: SCELPHONO 4x5x3
y_data_2, z_data_2, x_data_2  = np.loadtxt('/home/david/Trabajo/structures/SCRIPT_ON_ALL_THE_PHASES/Calcite_II_correct_description/solid_1__xyz_sorted_as_P_wise.dat').T

# Obtain the values of G and P for a three constant Temperatures:
T_one = 10.0000000000000
T_two = 271.3100000000000
T_three = 2000.0000000000000

# 1) Calcite I SCELPHONO 444:
XS_T_one = []
YS_T_one = []
ZS_T_one = []

XS_T_two = []
YS_T_two = []
ZS_T_two = []

XS_T_three = []
YS_T_three = []
ZS_T_three = []


for ys, zs, xs in zip(y_data, z_data, x_data):
  if xs == T_one:
     XS_T_one.append(xs)
     YS_T_one.append(ys)
     ZS_T_one.append(zs)

  if xs == T_two:
     XS_T_two.append(xs)
     YS_T_two.append(ys)
     ZS_T_two.append(zs)

  if xs == T_three:
     XS_T_three.append(xs)
     YS_T_three.append(ys)
     ZS_T_three.append(zs)


output_array = np.vstack((XS_T_one, YS_T_one, ZS_T_one)).T
np.savetxt('Calcite_I_scelphono_444__G_P_at_10K.dat', output_array, header="T (K)  \t P (GPa) \t G / F.unit (a.u.)", fmt="%s")

output_array = np.vstack((XS_T_two, YS_T_two, ZS_T_two)).T
np.savetxt('Calcite_I_scelphono_444__G_P_at_271.31K.dat', output_array, header="T (K)  \t P (GPa) \t G / F.unit (a.u.)", fmt="%s")

output_array = np.vstack((XS_T_three, YS_T_three, ZS_T_three)).T
np.savetxt('Calcite_I_scelphono_444__G_P_at_2000K.dat', output_array, header="T (K)  \t P (GPa) \t G / F.unit (a.u.)", fmt="%s")


# 2) Calcite I SCELPHONO 222:
XS_SMALL_T_one = []
YS_SMALL_T_one = []
ZS_SMALL_T_one = []

XS_SMALL_T_two = []
YS_SMALL_T_two = []
ZS_SMALL_T_two = []

XS_SMALL_T_three = []
YS_SMALL_T_three = []
ZS_SMALL_T_three = []


for ys_small, zs_small, xs_small in zip(y_data_small, z_data_small, x_data_small):
  if xs_small == T_one:
     print xs_small, ys_small, zs_small 
     XS_SMALL_T_one.append(xs_small)
     YS_SMALL_T_one.append(ys_small)
     ZS_SMALL_T_one.append(zs_small)

  if xs_small == T_two:
     print xs_small, ys_small, zs_small 
     XS_SMALL_T_two.append(xs_small)
     YS_SMALL_T_two.append(ys_small)
     ZS_SMALL_T_two.append(zs_small)

  if xs_small == T_three:
     print xs_small, ys_small, zs_small 
     XS_SMALL_T_three.append(xs_small)
     YS_SMALL_T_three.append(ys_small)
     ZS_SMALL_T_three.append(zs_small)


output_array = np.vstack((XS_SMALL_T_one, YS_SMALL_T_one, ZS_SMALL_T_one)).T
np.savetxt('Calcite_I_scelphono_222__G_P_at_10K.dat', output_array, header="T (K)  \t P (GPa) \t G / F.unit (a.u.)", fmt="%s")

output_array = np.vstack((XS_SMALL_T_two, YS_SMALL_T_two, ZS_SMALL_T_two)).T
np.savetxt('Calcite_I_scelphono_222__G_P_at_271.31K.dat', output_array, header="T (K)  \t P (GPa) \t G / F.unit (a.u.)", fmt="%s")

output_array = np.vstack((XS_SMALL_T_three, YS_SMALL_T_three, ZS_SMALL_T_three)).T
np.savetxt('Calcite_I_scelphono_222__G_P_at_2000K.dat', output_array, header="T (K)  \t P (GPa) \t G / F.unit (a.u.)", fmt="%s")


# 1) Calcite II SCELPHONO 453:
XS_2_T_one = []
YS_2_T_one = []
ZS_2_T_one = []

XS_2_T_two = []
YS_2_T_two = []
ZS_2_T_two = []

XS_2_T_three = []
YS_2_T_three = []
ZS_2_T_three = []


for ys_2, zs_2, xs_2 in zip(y_data_2, z_data_2, x_data_2):
  if xs_2 == T_one:
     XS_2_T_one.append(xs_2)
     YS_2_T_one.append(ys_2)
     ZS_2_T_one.append(zs_2)

  if xs_2 == T_two:
     XS_2_T_two.append(xs_2)
     YS_2_T_two.append(ys_2)
     ZS_2_T_two.append(zs_2)

  if xs_2 == T_three:
     XS_2_T_three.append(xs_2)
     YS_2_T_three.append(ys_2)
     ZS_2_T_three.append(zs_2)


output_array = np.vstack((XS_2_T_one, YS_2_T_one, ZS_2_T_one)).T
np.savetxt('Calcite_II_scelphono_453__G_P_at_10K.dat', output_array, header="T (K)  \t P (GPa) \t G / F.unit (a.u.)", fmt="%s")

output_array = np.vstack((XS_2_T_two, YS_2_T_two, ZS_2_T_two)).T
np.savetxt('Calcite_II_scelphono_453__G_P_at_271.31K.dat', output_array, header="T (K)  \t P (GPa) \t G / F.unit (a.u.)", fmt="%s")

output_array = np.vstack((XS_2_T_three, YS_2_T_three, ZS_2_T_three)).T
np.savetxt('Calcite_I_scelphono_453__G_P_at_2000K.dat', output_array, header="T (K)  \t P (GPa) \t G / F.unit (a.u.)", fmt="%s")


# 2) Calcite II SCELPHONO 121_1-21_210
XS_2_SMALL_T_one = []
YS_2_SMALL_T_one = []
ZS_2_SMALL_T_one = []

XS_2_SMALL_T_two = []
YS_2_SMALL_T_two = []
ZS_2_SMALL_T_two = []

XS_2_SMALL_T_three = []
YS_2_SMALL_T_three = []
ZS_2_SMALL_T_three = []


for ys_2_small, zs_2_small, xs_2_small in zip(y_data_2_small, z_data_2_small, x_data_2_small):
  if xs_2_small == T_one:
     print xs_2_small, ys_2_small, zs_2_small 
     XS_2_SMALL_T_one.append(xs_2_small)
     YS_2_SMALL_T_one.append(ys_2_small)
     ZS_2_SMALL_T_one.append(zs_2_small)

  if xs_2_small == T_two:
     print xs_2_small, ys_2_small, zs_2_small 
     XS_2_SMALL_T_two.append(xs_2_small)
     YS_2_SMALL_T_two.append(ys_2_small)
     ZS_2_SMALL_T_two.append(zs_2_small)

  if xs_2_small == T_three:
     print xs_2_small, ys_2_small, zs_2_small 
     XS_2_SMALL_T_three.append(xs_2_small)
     YS_2_SMALL_T_three.append(ys_2_small)
     ZS_2_SMALL_T_three.append(zs_2_small)


output_array = np.vstack((XS_2_SMALL_T_one, YS_2_SMALL_T_one, ZS_2_SMALL_T_one)).T
np.savetxt('Calcite_II_scelphono_121_1-21_210__G_P_at_10K.dat', output_array, header="T (K)  \t P (GPa) \t G / F.unit (a.u.)", fmt="%s")

output_array = np.vstack((XS_2_SMALL_T_two, YS_2_SMALL_T_two, ZS_2_SMALL_T_two)).T
np.savetxt('Calcite_II_scelphono_121_1-21_210__G_P_at_271.31K.dat', output_array, header="T (K)  \t P (GPa) \t G / F.unit (a.u.)", fmt="%s")

output_array = np.vstack((XS_2_SMALL_T_three, YS_2_SMALL_T_three, ZS_2_SMALL_T_three)).T
np.savetxt('Calcite_II_scelphono_121_1-21_210__G_P_at_2000K.dat', output_array, header="T (K)  \t P (GPa) \t G / F.unit (a.u.)", fmt="%s")


#sys.exit()

####### Calcite I scattered:
# In a new figure, each surface separately:
# set "fig" and "ax" varaibles
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the initial scattered points
ax.scatter(x_data, y_data, z_data, color='k', marker='o', label='SCELPHONO 4x4x4')
ax.scatter(x_data_small, y_data_small, z_data_small, color='m', marker='o', label='SCELPHONO 2x2x2') 

ax.set_xlabel('\nT (K)')
ax.set_ylabel('P (GPa)')
ax.set_zlabel('\nGibbs free energy / F.unit (a.u.)', linespacing=3)
ax.set_title('\n\nCalcite I', linespacing=3)
xlabels=[0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
ax.set_xticklabels(xlabels,rotation=90,
                  verticalalignment='baseline',#)#,
                  horizontalalignment='left')

ylabels = [-4, -2, 0, 2, 4, 6, 8, 10] 
ax.set_yticklabels(ylabels,rotation=0,
                  verticalalignment='baseline')#,
#                 horizontalalignment='left')


plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4)


fig.savefig("Calcite_I_scattered.pdf",  bbox_inches='tight', pad_inches=0.3)#, tight_layout() )#, bbox_inches=bbox)


####### Calcite II scattered:
# In a new figure, each surface separately:
# set "fig" and "ax" varaibles
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the initial scattered points
ax.scatter(x_data_2, y_data_2, z_data_2, color='k', marker='o', label='SCELPHONO 4x5x3')
ax.scatter(x_data_2_small, y_data_2_small, z_data_2_small, color='m', marker='o', label='SCELPHONO 121 / 1-21 / 210') 

ax.set_xlabel('\nT (K)')
ax.set_ylabel('P (GPa)')
ax.set_zlabel('\nGibbs free energy / F.unit (a.u.)', linespacing=3)
ax.set_title('\n\nCalcite II', linespacing=3)
xlabels=[0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
ax.set_xticklabels(xlabels,rotation=90,
                  verticalalignment='baseline',#)#,
                  horizontalalignment='left')

ylabels = [-4, -2, 0, 2, 4, 6, 8, 10] 
ax.set_yticklabels(ylabels,rotation=0,
                  verticalalignment='baseline')#,
#                 horizontalalignment='left')


plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4)


fig.savefig("Calcite_II_scattered.pdf",  bbox_inches='tight', pad_inches=0.3)#, tight_layout() )#, bbox_inches=bbox)



####### Calcite I, 10.0K:
plt.figure()
# Plotting the scattered points: 
p1 = plt.scatter(YS_T_one, ZS_T_one, color='black', marker="o", facecolors='none', label='SCELPHONO 4x4x4', s=100)
p2 = plt.scatter(YS_SMALL_T_one, ZS_SMALL_T_one, color='magenta', marker="o", facecolors='none', label='SCELPHONO 2x2x2', s=100)

#fontP = FontProperties()
#fontP.set_size('small')
#plt.legend((p1, p2), ("SCELPHONO 4x4x4", "BM fit Calcite I", "Calcite II", 'BM fit Calcite II'), prop=fontP)

plt.legend() 
plt.xlabel('P (GPa)')
plt.ylabel('Gibbs free energy / F.unit (a.u.)')
plt.suptitle("Calcite I", fontsize=15)
plt.title("10.0 K")
#plt.ticklabel_format(useOffset=False)
plt.grid()
plt.savefig('Calcite_I_scelphono_444_and_scelphono_222__G_P_at_10K.pdf', bbox_inches='tight')


####### Calcite I, 271.31K:
plt.figure()
# Plotting the scattered points: 
p1 = plt.scatter(YS_T_two, ZS_T_two, color='black', marker="o", facecolors='none', label='SCELPHONO 4x4x4', s=100)
p2 = plt.scatter(YS_SMALL_T_two, ZS_SMALL_T_two, color='magenta', marker="o", facecolors='none', label='SCELPHONO 2x2x2', s=100)

#fontP = FontProperties()
#fontP.set_size('small')
#plt.legend((p1, p2), ("SCELPHONO 4x4x4", "BM fit Calcite I", "Calcite II", 'BM fit Calcite II'), prop=fontP)

plt.legend() 
plt.xlabel('P (GPa)')
plt.ylabel('Gibbs free energy / F.unit (a.u.)')
plt.suptitle("Calcite I", fontsize=15)
plt.title("271.31 K")
#plt.ticklabel_format(useOffset=False)
plt.grid()
plt.savefig('Calcite_I_scelphono_444_and_scelphono_222__G_P_at_271.31K.pdf', bbox_inches='tight')


####### Calcite I, 2000K:
plt.figure()
# Plotting the scattered points: 
p1 = plt.scatter(YS_T_three, ZS_T_three, color='black', marker="o", facecolors='none', label='SCELPHONO 4x4x4', s=100)
p2 = plt.scatter(YS_SMALL_T_three, ZS_SMALL_T_three, color='magenta', marker="o", facecolors='none', label='SCELPHONO 2x2x2', s=100)

#fontP = FontProperties()
#fontP.set_size('small')
#plt.legend((p1, p2), ("SCELPHONO 4x4x4", "BM fit Calcite I", "Calcite II", 'BM fit Calcite II'), prop=fontP)

plt.legend() 
plt.xlabel('P (GPa)')
plt.ylabel('Gibbs free energy / F.unit (a.u.)')
plt.suptitle("Calcite I", fontsize=15)
plt.title("2000.0 K")
#plt.ticklabel_format(useOffset=False)
plt.grid()
plt.savefig('Calcite_I_scelphono_444_and_scelphono_222__G_P_at_2000K.pdf', bbox_inches='tight')


####### Calcite II, 10.0K:
plt.figure()
# Plotting the scattered points:  XS_2_SMALL_T_one, XS_2_T_one,
p1 = plt.scatter(YS_2_T_one, ZS_2_T_one, color='black', marker="o", facecolors='none', label='SCELPHONO 4x5x3', s=100)
p2 = plt.scatter(YS_2_SMALL_T_one, ZS_2_SMALL_T_one, color='magenta', marker="o", facecolors='none', label='SCELPHONO 121 / 1-21 / 210', s=100)

#fontP = FontProperties()
#fontP.set_size('small')
#plt.legend((p1, p2), ("SCELPHONO 4x4x4", "BM fit Calcite I", "Calcite II", 'BM fit Calcite II'), prop=fontP)

plt.legend() 
plt.xlabel('P (GPa)')
plt.ylabel('Gibbs free energy / F.unit (a.u.)')
plt.suptitle("Calcite II", fontsize=15)
plt.title("10.0 K")
#plt.ticklabel_format(useOffset=False)
plt.grid()
plt.savefig('Calcite_II_scelphono_453_and_scelphono_121_1-21_210__G_P_at_10K.pdf', bbox_inches='tight')


####### Calcite II, 271.31K:
plt.figure()
# Plotting the scattered points: 
p1 = plt.scatter(YS_2_T_two, ZS_2_T_two, color='black', marker="o", facecolors='none', label='SCELPHONO 4x5x3', s=100)
p2 = plt.scatter(YS_2_SMALL_T_two, ZS_2_SMALL_T_two, color='magenta', marker="o", facecolors='none', label='SCELPHONO 121 / 1-21 / 210', s=100)

#fontP = FontProperties()
#fontP.set_size('small')
#plt.legend((p1, p2), ("SCELPHONO 4x4x4", "BM fit Calcite I", "Calcite II", 'BM fit Calcite II'), prop=fontP)

plt.legend() 
plt.xlabel('P (GPa)')
plt.ylabel('Gibbs free energy / F.unit (a.u.)')
plt.suptitle("Calcite II", fontsize=15)
plt.title("271.31 K")
#plt.ticklabel_format(useOffset=False)
plt.grid()
plt.savefig('Calcite_II_scelphono_453_and_scelphono_121_1-21_210__G_P_at_271.31K.pdf', bbox_inches='tight')


####### Calcite II, 2000K:
plt.figure()
# Plotting the scattered points: 
p1 = plt.scatter(YS_2_T_three, ZS_2_T_three, color='black', marker="o", facecolors='none', label='SCELPHONO 4x5x3', s=100)
p2 = plt.scatter(YS_2_SMALL_T_three, ZS_2_SMALL_T_three, color='magenta', marker="o", facecolors='none', label='SCELPHONO 121 / 1-21 / 210', s=100)

#fontP = FontProperties()
#fontP.set_size('small')
#plt.legend((p1, p2), ("SCELPHONO 4x4x4", "BM fit Calcite I", "Calcite II", 'BM fit Calcite II'), prop=fontP)

plt.legend() 
plt.xlabel('P (GPa)')
plt.ylabel('Gibbs free energy / F.unit (a.u.)')
plt.suptitle("Calcite II", fontsize=15)
plt.title("2000.0 K")
#plt.ticklabel_format(useOffset=False)
plt.grid()
plt.savefig('Calcite_II_scelphono_453_and_scelphono_121_1-21_210__G_P_at_2000K.pdf', bbox_inches='tight')

plt.show()

