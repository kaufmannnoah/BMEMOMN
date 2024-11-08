import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'cmu serif'

# Define the vertices of a tetrahedron
vertices_t = np.array([[-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]])
vertices_cu = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]])
vertices_oc = np.array([[ 1,  0,  0], [-1,  0,  0], [ 0,  1,  0], [ 0, -1,  0], [ 0,  0,  1], [ 0,  0, -1]])

# Define the faces of the tetrahedron by the vertices
faces_t = [[vertices_t[j] for j in face] for face in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]]
faces_oc = [[vertices_oc[j] for j in face] for face in [[0, 2, 4], [0, 3, 4], [1, 2, 4], [1, 3, 4], [0, 2, 5], [0, 3, 5], [1, 2, 5], [1, 3, 5]]]
faces_oc2 = [[vertices_oc[j] for j in face] for face in [[0, 1, 2, 3], [0, 1, 4, 5], [2, 3, 4, 5]]]
faces_cu = [[vertices_cu[j] for j in face] for face in [[0, 1, 2, 3], [4, 5, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 5, 4], [2, 3, 7, 6]]]


# Plot the tetrahedron
fig = plt.figure(figsize= [4.24, 4.24], tight_layout= {'pad': 0})
ax = fig.add_subplot(111, projection='3d')

label = [r'$\Psi^{-}$', r'$\Phi^{-}$', r'$\Phi^{+}$', r'$\Psi^{+}$']
#label = ["A", "B", "C", "D"]
label_r = [r'$r_{1}$', r'$r_{2}$', r'$r_{3}$']
fs = 12

lw_scale = 0.75
# Draw the faces
ax.add_collection3d(Poly3DCollection(faces_cu, facecolors=None, linewidth=1* lw_scale, ls= '-', lw= 0.5 * lw_scale, edgecolors='gray', alpha=0), zs= 0)
ax.add_collection3d(Poly3DCollection(faces_oc, facecolors='black', linewidth=1* lw_scale, edgecolors='black', ls= ":",  alpha=0.1), zs= 1)
#ax.add_collection3d(Poly3DCollection(faces_oc2, facecolors='black', linewidth=0, edgecolors='black', ls= "",  alpha=0.1), zs= 1)
ax.add_collection3d(Poly3DCollection(faces_t, facecolors='black', linewidth=1* lw_scale, edgecolors='black', alpha=0.1), zs= 2)
ax.scatter(vertices_t[:, 0], vertices_t[:, 1], vertices_t[:, 2], color='black', s=30, depthshade=False)

# Add labels next to the vertices
f_0 = [1.11, 1.1, 1.04, 1.2]
f_1 = [1.13, 1.12, 1.2, 1.18]
f_2 = [1.14, 1.1, 1.2, 1.18]
for i, txt in enumerate(label):
    ax.text(vertices_t[i, 0] * f_0[i], vertices_t[i, 1] * f_1[i], vertices_t[i, 2] *f_2[i], txt, size=fs, zorder=1, ha="center", va="center", color='black')

vertices_cor = np.array([[-0.95, -1, -1.04], [-0.96, 1, 1], [1.1, -0.4, 1], [1.04, 1, -0.94]])
cor = ['(-1, -1, -1)', '(-1, 1, 1)', '(1, -1, 1)', '(1, 1, -1)']
or_h = ['left', 'left', 'center', 'left']
or_v = ['top', 'bottom', 'center', 'bottom']
dir = ['x', 'x', 'y', 'z']
for i, txt in enumerate(cor):
    ax.text(vertices_cor[i, 0], vertices_cor[i, 1], vertices_cor[i, 2], txt, size=fs-4, zorder=1, ha=or_h[i], va=or_v[i], zdir= dir[i], color='dimgray')

pos = [[0, -1.1, -1.1], [1.1, 0, -1.1], [-1.1, -1.1, 0]]
for i, txt in enumerate(label_r):
    ax.text(*pos[i], txt, size=fs, zorder=1, ha="center", va="center", color='black')

# Set the aspect ratio for the plot to be equal
ax.set_box_aspect([1, 1, 1])

ax.set_axis_off()
ax.grid(None)
lim = 1.25
ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
ax.set_zlim([-lim, lim])

ax.view_init(elev=17, azim=-77)
plt.savefig("tetrahedron.pdf", format="pdf", bbox_inches='tight', pad_inches=-0.5)
plt.show()