import numpy as np
from parameters import *


def graph_l(G):
	edges = G["edge"]
	l = edges.shape[0]
	return l

def next_state(q,u):
	global dt
	global n
	global m
	if (q.shape[0] != m*n):
		print("error in q shape :", q.shape)
	if (u.shape[0] != m*n):
		print("error in q shape :", u.shape)
	q_new = q.reshape(m*n,1) + (u*dt).reshape(m*n,1)
	# print(q,u)
	# print(q_new,'------------------------')

	return q_new

def H_bar(G):
	global m
	global n
	E = G['edge']
	l = graph_l(G)
	H = np.zeros((l,n))
	for i in range(l):
		edge = E[i]
		# print(edge)
		for k in range(n):
			if round(edge[0]-1) == k:
				H[i,k] = 1
			if round(edge[1] -1) == k:
				H[i,k] = -1
	return np.kron(H,np.eye(m))

def P_diag(p,G):
	global m
	global n
	l = graph_l(G)

	H = H_bar(G)
	p_bar = np.matmul(H,p.reshape(m*n,1))
	P = np.zeros((m*l,l))
	for i in range(l):
		P[i*m:(i+1)*(m),i] = p_bar[i*m:(i+1)*(m)].reshape(3,)
	# print(P)
	return P


def rigidity(p,G):
	P = P_diag(p,G)
	H = H_bar(G)
	R = np.matmul(P.T,H)
	return R


def e0_edge(q,G):
	global m

	d = G['d']
	l = graph_l(G)
	H = H_bar(G)
	q_bar = np.matmul(H,q)
	norm_q = np.zeros(l)
	for i in range(l):
		vec = q[i*m:(i+1)*(m)]
		norm_q[i] = np.linalg.norm(vec)
	return (norm_q.reshape(l,) - d.reshape(l,))

def e_bounds(G,q):
	q = np.copy(G['init'])
	e0 = e0_edge(q,G)
	# print('e0',e0)
	global r_s
	global r_c
	global e_lower
	global e_upper
	global mew_lower
	global mew_upper
	global mew

	l = graph_l(G)



	e_lower = np.zeros(l)
	e_upper = np.zeros(l)
	d = G['d']


	E = G['edge']

	for k in range(l):
		i = E[k][0] - 1
		j = E[k][1] - 1
		dij = d[k]
		rsij = r_s[i] + r_s[j]
		rcij = np.min(np.array([r_c[i],r_c[j]]))

		# mew_upper = rcij - dij
		# mew_lower = dij - rsij
		eij = e0[k]
		if (eij >= 0):
			e_upper[k] = np.min(np.array([
				np.linalg.norm(eij) + mew,
				rcij - dij]))
			e_lower[k] = np.min(np.array([
				np.linalg.norm(eij) + mew,
				mew_lower]))
		else:
			e_upper[k] = np.min(np.array([
				np.linalg.norm(eij) + mew,
				mew_upper]))
			e_lower[k] = np.min(np.array([
				np.linalg.norm(eij) + mew,
				dij - rsij]))


	return e_lower,e_upper

def init_b_bounds(G,q):
	
	global b_upper
	global b_lower
	global e_upper
	global e_lower
	global rho_0

	e_lower,e_upper = e_bounds(G,q)



	l = graph_l(G)

	b_lower = np.zeros(l)
	b_upper = np.zeros(l)
	d = G['d']
	for k in range(l):
		eiju = e_upper[k]
		eijl = e_lower[k]
		b_upper[k] = (((eiju*eiju)) + ((2*eiju*d[k])))/rho_0
		b_lower[k] =   (((2*eijl*d[k])) - ((eijl*eijl)))/rho_0
	return b_lower,b_upper



def neta_t(G,q):
	global m
	d = G['d']
	l = graph_l(G)
	H = H_bar(G)
	q_bar = np.matmul(H,q)
	neta = np.zeros(l)
	for k in range(l):
		q_vec = q_bar[(k*m):(k+1)*m]

		neta[k] = (np.linalg.norm(q_vec)*np.linalg.norm(q_vec) ) - (d[k]*d[k])
	return neta


def performance(t):
	global rho_0
	global rho_inf
	global a
	rho_t = ((rho_0 - rho_inf)*(np.exp(-a*t))) + rho_inf
	return rho_t





def neta_cap(G,q,t):
	global rho_0

	neta = neta_t(G,q)
	neta_c = neta/performance(t)
	return neta_c

def zeta(G,q,t):
	global b_upper
	global b_lower
	b_lower,b_upper = init_b_bounds(G,q)
	neta_c = neta_cap(G,q,t)
	rho = performance(t)
	l = graph_l(G)
	zeta_value = np.zeros(l)
	for k in range(l):
		zeta_value[k] = ((1/(neta_c[k] + b_lower[k])) - (1/(neta_c[k] - b_upper[k])))/(rho)

	return np.diag(zeta_value)
def sigma(G,q,t):
	global b_upper
	global b_lower
	b_lower,b_upper = init_b_bounds(G,q)



	l = graph_l(G)
	neta_c = neta_cap(G,q,t)

	# print("------------------------")
	# print(b_upper,b_lower,neta_c)

	# print("------------------------")


	sig = np.zeros(l)
	print(b_upper,neta_c,"aoao")
	for k in range(l):

		value = ((b_upper[k]*neta_c[k])+(b_upper[k]*b_lower[k]))/((b_upper[k]*b_lower[k])-(b_lower[k]*neta_c[k]))
		print("value", value)
		value = np.log(value)
		value  = value/2
		sig[k] = value
	return sig

def K(G):
	global k_constant
	l = graph_l(G)
	return np.eye(l)*k_constant



def control_input(G,q,t):
	R = rigidity(q,G)
	zeta_v = zeta(G,q,t)
	sigma_v = sigma(G,q,t)
	K_v = K(G)
	l = graph_l(G)

	# print('pppp',R,'zeta','\n',zeta_v,'\n',sigma_v,'\n',K_v,'\n','inputs')

	u = np.matmul(R.T,zeta_v)
	u = np.matmul(u,K_v)
	u = np.matmul(u,sigma_v.reshape(l,1))
	u = -u
	return u







