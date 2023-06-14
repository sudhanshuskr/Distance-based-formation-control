import numpy as np
global m
global mew
global n
global dt
global r_s
global r_c
global e_lower
global e_upper
global mew_lower
global mew_upper
global neta_lower
global neta_upper
global b_lower
global b_upper
global a 
global rho_0
global rho_inf
global k_constant

k_constant = 0.1


a = 0.6
rho_0 = 1
rho_inf = 0.03



mew_lower = 0.3
mew_upper = 0.3
neta_lower = 0
neta_upper = 0
# b_lower = 0
# b_upper = 0



m = 3
n = 4
mew = 0.12
dt = 0.01

r_s_constant = 0.2
r_c_constant = 5
r_s = np.ones(n)*r_s_constant
r_c = np.ones(n)*r_c_constant
