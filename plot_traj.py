import numpy as np
from parameters import *
from mpl_toolkits import mplot3d
from functions import *
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from graph_inventory import G1
from tqdm import tqdm


plt.ion()
fig = plt.figure()

ax = fig.add_subplot(111,projection='3d')
traj = np.load('trajectory.npy')
# Data for a three-dimensional line
# zline = np.linspace(0, 15, 1000)
# xline = np.sin(zline)
# yline = np.cos(zline)
# ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# for i in range(traj.shape[0]):
# 	xdata = G1["init"][:,0] + i/100
# 	ydata = G1["init"][:,1] + i/150
# 	zdata = G1["init"][:,2] + i/120
# 	ax.scatter3D(xdata, ydata, zdata,  cmap='Greens');
# 	l = graph_l(G1)
# 	E = G1['edge']

# 	for k in range(l):
# 		xline = np.array([xdata[E[k][0] -1],xdata[E[k][1] -1]])
# 		yline = np.array([ydata[E[k][0] -1],ydata[E[k][1] -1]])
# 		zline = np.array([zdata[E[k][0] -1],zdata[E[k][1] -1]])
# 		ax.plot3D(xline, yline, zline, 'gray')

	
# 	ax.set_xlim(-20,20)
# 	ax.set_ylim(-20,20)
# 	ax.set_zlim(-20,20)
# 	plt.draw()
# 	plt.pause(0.01)
# 	plt.cla()

# for i in tqdm(range(traj.shape[0])):
# 	q = traj[i,:].reshape(n,m)
# 	xdata = q[:,0] 
# 	ydata = q[:,1] 
# 	zdata = q[:,2] 
# 	ax.scatter3D(xdata, ydata, zdata,  cmap='Greens');
# 	l = graph_l(G1)
# 	E = G1['edge']

# 	for k in range(l):
# 		xline = np.array([xdata[E[k][0] -1],xdata[E[k][1] -1]])
# 		yline = np.array([ydata[E[k][0] -1],ydata[E[k][1] -1]])
# 		zline = np.array([zdata[E[k][0] -1],zdata[E[k][1] -1]])
# 		ax.plot3D(xline, yline, zline, 'gray')

	
# 	ax.set_xlim(-5,5)
# 	ax.set_ylim(-5,5)
# 	ax.set_zlim(-5,5)
# 	plt.draw()
# 	plt.pause(0.000000001)
# 	plt.cla()


for i in tqdm(range(traj.shape[0])):
	q = traj[i,:].reshape(n,m)
	xdata = q[:,0] 
	ydata = q[:,1] 
	zdata = q[:,2] 
	ax.scatter3D(xdata, ydata, zdata,  cmap='Greens');
	l = graph_l(G1)
	E = G1['edge']

	for k in range(l):
		xline = np.array([xdata[E[k][0] -1],xdata[E[k][1] -1]])
		yline = np.array([ydata[E[k][0] -1],ydata[E[k][1] -1]])
		zline = np.array([zdata[E[k][0] -1],zdata[E[k][1] -1]])
		ax.plot3D(xline, yline, zline, 'gray')

	
	ax.set_xlim(-5,5)
	ax.set_ylim(-5,5)
	ax.set_zlim(-5,5)
	plt.draw()
	plt.pause(0.000000001)
	plt.cla()
