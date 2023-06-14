import numpy as np
from parameters import *
from graph_inventory import *
from functions import * 
from tqdm import tqdm

p = G1['init']
print(p)
print(e0_edge(p,G1))
global m
global n

global dt
t = 0

q = G1['init']


T = 0.03

traj = [q]

for i in tqdm(range(round(T/dt))):
	# print(q)
	# print(q,'qqqqq')
	u = control_input(G1,q,t)
	# print(u,'u')
	q = next_state(q,u).reshape(m*n,)

	traj.append(q)
	t = t+dt

print(np.array(traj).shape)
np.save('trajectory1', np.array(traj)) 







