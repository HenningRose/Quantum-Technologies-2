import numpy as np
import copy

# Initialize parameters
N_time = 51*10
N_E = 10000
sigma_st = 0.001
alpha = 0.1
gamma_z = 0.01
Omega_max = 10
N_m = 10

# Set up time series
times = np.linspace(0, 1, N_time)

# Initialize fidelity list
F_list = np.zeros(N_E)

# Initialize Hamiltonian components
h_x = np.random.normal(0,1,N_m)
h_y = np.random.normal(0,1,N_m)

# Time evol
rho_x_dot = lambda rho_x, rho_y, rho_z, h_x, h_y, gamma_z: 2*h_y*rho_z - 2*gamma_z*rho_x
rho_y_dot = lambda rho_x, rho_y, rho_z, h_x, h_y, gamma_z: -2*h_x*rho_z - 2*gamma_z*rho_y
rho_z_dot = lambda rho_x, rho_y, rho_z, h_x, h_y: 2*h_x*rho_y - 2*h_y*rho_x

dt = times[1] - times[0]

for j in range(N_E):
    # Initialize state variables
    rho_x_0, rho_y_0, rho_z_0 = np.random.normal(0, 1, 3)
    norm = np.sqrt(rho_x_0**2 + rho_y_0**2 + rho_z_0**2)
    rho_x_0, rho_y_0, rho_z_0 = rho_x_0/norm, rho_y_0/norm, rho_z_0/norm

    rho_x_eps = copy.deepcopy(rho_x_0)
    rho_y_eps = copy.deepcopy(rho_y_0) 
    rho_z_eps = copy.deepcopy(rho_z_0)

    rho_x = copy.deepcopy(rho_x_0)
    rho_y = copy.deepcopy(rho_y_0)
    rho_z = copy.deepcopy(rho_z_0)
    
    # Time evolution for params theta
    for t in times:
        # Create Hamiltonian for each time step
        H_x = sum([h_x[n]*np.sin(np.pi*n*t) for n in range(N_m)])
        H_y = sum([h_y[n]*np.sin(np.pi*n*t) for n in range(N_m)])
        H_x = H_x / max(1, 2*np.sqrt(H_x**2 + H_y**2)/Omega_max)
        H_y = H_y / max(1, 2*np.sqrt(H_x**2 + H_y**2)/Omega_max)

        # Update rho_x, rho_y, rho_z using explicit time evolution
        rho_x += rho_x_dot(rho_x, rho_y, rho_z, H_x, H_y, gamma_z) * dt
        rho_y += rho_y_dot(rho_x, rho_y, rho_z, H_x, H_y, gamma_z) * dt
        rho_z += rho_z_dot(rho_x, rho_y, rho_z, H_x, H_y) * dt

        # Normalize rho_x, rho_y, rho_z
#        norm = np.sqrt(rho_x**2 + rho_y**2 + rho_z**2)
#        rho_x, rho_y, rho_z = rho_x/norm, rho_y/norm, rho_z/norm

    # Choose theta_up to update
    up = np.random.randint(2*N_m)
    eps = np.random.normal(0, sigma_st,2+N_m)

    if up<N_m:
        h_x[up] += eps[up]
    else:
        h_y[up-N_m]+= eps[up-N_m]

    # Time evolution with updated theta_j -> theta_j + eps
    for t in times:
        H_x_eps = sum([h_x[n]*np.sin(np.pi*n*t) for n in range(N_m)])
        H_y_eps = sum([h_y[n]*np.sin(np.pi*n*t) for n in range(N_m)])
        H_x_eps = H_x_eps / max(1, 2*np.sqrt(H_x_eps**2 + H_y_eps**2)/Omega_max)
        H_y_eps = H_y_eps / max(1, 2*np.sqrt(H_x_eps**2 + H_y_eps**2)/Omega_max)

        # Update rho_x, rho_y, rho_z using explicit time evolution
        rho_x_eps += rho_x_dot(rho_x_eps, rho_y_eps, rho_z_eps, H_x_eps, H_y_eps, gamma_z) * dt
        rho_y_eps += rho_y_dot(rho_x_eps, rho_y_eps, rho_z_eps, H_x_eps, H_y_eps, gamma_z) * dt
        rho_z_eps += rho_z_dot(rho_x_eps, rho_y_eps, rho_z_eps, H_x_eps, H_y_eps) * dt

        # Normalize rho_x, rho_y, rho_z
#        norm = np.sqrt(rho_x_eps**2 + rho_y_eps**2 + rho_z_eps**2)
#        rho_x_eps, rho_y_eps, rho_z_eps = rho_x_eps/norm, rho_y_eps/norm, rho_z_eps/norm

    # Calculate fidelity
    F = 0.5 * (1 + rho_x_0 * rho_z + rho_z_0 * rho_x - rho_y_0 * rho_y)
    F_eps = 0.5 * (1 + rho_x_0 * rho_z_eps + rho_z_0 * rho_x_eps - rho_y_0 * rho_y_eps)
    F_list[j] = F

    # Make gradient step
    if up<N_m:
        h_x[up] += alpha * (F_eps - F) / eps[up] 
    else: 
        h_y[up-N_m] += alpha * (F_eps - F) / eps[up-N_m] 

# Write fidelity and params to files 
with open("fidelity.txt", "w") as file:
    # Write each fidelity value on a new line
    for F in F_list:
        file.write(str(F) + "\n")


with open("params_x.txt", "w") as file:
    # Write each param value on a new line
    for h in h_x:
        file.write(str(h) + "\n")

with open("params_y.txt", "w") as file:
    # Write each param value on a new line
    for h in h_y:
        file.write(str(h) + "\n")
