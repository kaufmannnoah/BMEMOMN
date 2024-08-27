import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

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
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw the faces
ax.add_collection3d(Poly3DCollection(faces_cu, facecolors=None, linewidth=1, ls= '-', lw= 0.5, edgecolors='gray', alpha=0), zs= 0)
ax.add_collection3d(Poly3DCollection(faces_oc, facecolors='black', linewidth=1, edgecolors='black', ls= ":",  alpha=0.1), zs= 1)
#ax.add_collection3d(Poly3DCollection(faces_oc2, facecolors='black', linewidth=0, edgecolors='black', ls= "",  alpha=0.1), zs= 1)
ax.add_collection3d(Poly3DCollection(faces_t, facecolors='black', linewidth=1, edgecolors='black', alpha=0.1), zs= 2)
ax.scatter(vertices_t[:, 0], vertices_t[:, 1], vertices_t[:, 2], color='black', s=50, depthshade=False)

# Set the aspect ratio for the plot to be equal
ax.set_box_aspect([1, 1, 1])

ax.set_axis_off()
ax.grid(None)

lim = 1.25
ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
ax.set_zlim([-lim, lim])

plt.show()