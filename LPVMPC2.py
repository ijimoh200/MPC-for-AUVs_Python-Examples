import numpy as np
from scipy.linalg import block_diag
import cvxopt
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

# Initial values of the state vector
x = 0.0
y = 0
z = 0
phi = 0
theta = 0
psi = 0
u = 0
v = 0
w = 0
p = 0
q = 0
r = 0
eta_ini = np.array([x, y, z, phi, theta, psi])
nv_ini = np.array([u, v, w, p, q, r])

# Sampling and simulation parameters
Ts = 0.05 # changing it to 0.05 works as well
Tf = 50
NoS = round(Tf / Ts)
t = np.arange(0, Tf, Ts)

# Generate reference trajectory and noise signals
noise = np.zeros((12, NoS))
yref = np.zeros((12, NoS))

nu_c = np.zeros((6, NoS))

for k in range(NoS):

    if k <= round(NoS / 4):
        yref[0:6, k] = np.array([0, 0, 0, 0, 0, 0])

    elif round(NoS / 4) <= k <= round(3 * NoS / 4):
        yref[0:6, k] = np.array([0.5, 0.5, 0, 0, 0, -0.1])
    else:
        yref[0:6, k] = np.array([0.5, 0.5, -0.5, 0, 0, -0.1])

    linearposition_noise = 0.005 * np.random.rand(3)
    angularposition_noise = (5.42 * 10 ** (-5)) * np.random.rand(3)
    linearvelocity_noise = 0.1 * np.random.rand(3)
    angularvelocity_noise = (5.42 * 10 ** (-3)) * np.random.rand(3)
    
    noise[:, k] = np.concatenate((linearposition_noise, angularposition_noise, 
                                  linearvelocity_noise, angularvelocity_noise))
    
    nuc_const = np.array([
        0.1 * np.cos(3/20 * np.pi * k * Ts) * np.cos(0.005 * k * Ts),
        0.1 * np.sin(0.1 * np.pi * k * Ts),
        0.1 * np.sin(0.002 * k * Ts) * np.cos(3/20 * np.pi * k * Ts),
        0,
        0,
        0])
    nuc_var = np.array([0.5, 1, 0.2, 0, 0, 0]) 
    if k <= round(NoS / 2):
        nu_c[:, k] = np.array([0, 0, 0.0, 0, 0, 0]) 
    else:
        nu_c[:, k] = nuc_var + nuc_const



# Helper functions for matrix manipulation

def awgn(signal, snr_db):
    snr = 10 * (snr_db / 10)
    power_signal = np.mean(np.abs(signal) ** 2)
    power_noise = power_signal / snr
    noise = np.sqrt(power_noise) * np.random.randn(*signal.shape)
    return signal + noise
import numpy as np

# Define predA, predB, predQ, solver_RK, AUV, and awgn as helper functions
def predA1(Aeta, N):
    temp = []
    for i in range(1, N + 1):
        temp.append(np.linalg.matrix_power(Aeta, i))
    
    A_p = np.vstack(temp)
    return A_p

def predB1(A, B, N, Nu):
    n, m = B.shape
    barC_ = np.zeros((N * n, Nu * m))
    
    # Construct the first column
    firstCol = []
    for i in range(N):
        firstCol.append(np.linalg.matrix_power(A, i) @ B)  # A^(i-1)*B
        
    firstCol = np.vstack(firstCol)  # Stack vertically
    barC_[:, :m] = firstCol
    
    # Fill the rest of the matrix based on the specific pattern
    for col in range(1, Nu):
        for row in range(n, N * n):
            barC_[row, col * m:(col + 1) * m] = barC_[row - n, (col - 1) * m:col * m]
    
    return barC_

def predQ(Q_p, S, N):
    # Create identity matrix of size (N-1)
    temp1 = np.eye(N-1)
    
    # Kronecker product of temp1 and Q_p
    temp2 = np.kron(temp1, Q_p)
    
    # Block diagonal matrix with temp2 and S
    Q_p = np.block([[temp2, np.zeros((temp2.shape[0], S.shape[1]))],
                    [np.zeros((S.shape[0], temp2.shape[1])), S]])
    
    return Q_p

def predW(C, N, Nu):
    p, n = C.shape
    barC_ = np.zeros((N * p, Nu * n))
    
    # Construct the first column
    firstCol = []
    for i in range(N):
        firstCol.append(C)
    
    firstCol = np.vstack(firstCol)  # Stack vertically
    barC_[:, :n] = firstCol
    
    # Fill the rest of the matrix based on the specific pattern
    for col in range(n, (Nu * n)):
        for row in range(p, (N * p)):
            barC_[row, col] = barC_[row - p, col - n]
    
    return barC_

def solver_RK(x, Ts, u, nu_c, dw):
    # RK = 4th-order Runge Kutta    
    k1, _, _, _, _, _ = BlueROV_sim(x, u, nu_c, dw)
    k2, _, _, _, _, _ = BlueROV_sim(x + k1.flatten() * (Ts / 2), u, nu_c, dw)
    k3, _, _, _, _, _ = BlueROV_sim(x + k2.flatten() * (Ts / 2), u, nu_c, dw)
    k4, _, _, _, _, _ = BlueROV_sim(x + k3.flatten() * Ts, u, nu_c, dw)

    xi = x + (Ts / 6) * (k1.flatten() + 2 * k2.flatten() + 2 * k3.flatten() + k4.flatten())
    return xi

def SS(a):
    """
    This function returns the skew-symmetric matrix for a 3-dimensional vector.
    Based on Fossen's definition (2011).
    """
    if len(a) != 3:
        raise ValueError('Input vector must have a dimension of 3!')
    a1, a2, a3 = a
    S_n = np.array([[0, -a3, a2],
                    [a3, 0, -a1],
                    [-a2, a1, 0]])
    return S_n

# This is the AUV model. You will need to create one for the BlueROV2
# AUV model for dynamic evolution considering disturbance

def BlueROV_sim(xi, ui, nu_c, dw):
    # State variables assignment
    x = xi[0]; y = xi[1]; z = xi[2]
    phi = xi[3]; theta = xi[4]; psi = xi[5]
    u = xi[6]; v = xi[7]; w = xi[8]
    p = xi[9]; q = xi[10]; r = xi[11]    
    nu = np.array([u, v, w, p, q, r]).reshape(6, 1) # velocity vector    
    # Compute relative velocity
    u_r = u - nu_c[0]; v_r = v-nu_c[1]; w_r = w-nu_c[0]
    nu_r = np.array([u_r, v_r, w_r, p, q, r]).reshape(6, 1)
    # Input variables
    tau_u, tau_v, tau_w, tau_p, tau_q, tau_r = ui
    # Vehicle's parameters
    m = 11.50; W = 112.80; B = 114.80; zg = 0.02
    Ixx = 0.16; Iyy = 0.16; Izz = 0.16

    # Nonlinear hydrodynamic damping coefficients
    Xuu = -18.18; Yvv = -21.66; Zww = -36.99
    Kpp = -1.55; Mqq = -1.55; Nrr = -1.55
    # Linear hydrodynamic damping coefficients
    Xu = -4.03; Yv = -6.22; Zw = -5.18
    Kp = -0.07; Mq = -0.07; Nr = -0.07
    # Added mass coefficients
    Xud = -5.50; Yvd = -12.70; Zwd = -14.60
    Kpd = -0.12; Mqd = -0.12; Nrd = -0.12
    # Kinematic model
    cos1, cos2, cos3 = np.cos(phi), np.cos(theta), np.cos(psi)
    sin1, sin2, sin3 = np.sin(phi), np.sin(theta), np.sin(psi)
    tan2 = np.tan(theta)
    # Rotation matrices
    J1 = np.array([
        [cos2*cos3, -sin3*cos1 + cos3*sin2*sin1, cos3*cos1*sin2 + sin3*sin1],
        [sin3*cos2, cos3*cos1 + sin1*sin2*sin3, sin2*sin3*cos1 - cos3*sin1],
        [-sin2, cos2*sin1, cos2*cos1]
    ])
    
    J2 = np.array([
        [1, sin1*tan2, cos1*tan2],
        [0, cos1, -sin1],
        [0, sin1/cos2, cos1/cos2]
    ])

    Jk = np.block([[J1, np.zeros((3, 3))], [np.zeros((3, 3)), J2]])

     # Dynamic model
    M_RB = np.array([[m, 0, 0, 0, m * zg, 0],
                     [0, m, 0, -m * zg, 0, 0],
                     [0, 0, m, 0, 0, 0],
                     [0, -m * zg, 0, Ixx, 0, 0],
                     [m * zg, 0, 0, 0, Iyy, 0],
                     [0, 0, 0, 0, 0, Izz]])

    # Added mass matrix MA
    M_A = -np.diag([Xud, Yvd, Zwd, Kpd, Mqd, Nrd])

    # Total mass matrix
    M = M_RB + M_A

    # Coriolis and centripetal matrix
    C_RB = np.array([[0, 0, 0, 0, m * w, -m * v],
                     [0, 0, 0, -m * w, 0, m * u],
                     [0, 0, 0, m * v, -m * u, 0],
                     [0, m * w, -m * v, 0, Izz * r, -Iyy * q],
                     [-m * w, 0, m * u, -Izz * r, 0, Ixx * p],
                     [m * v, -m * u, 0, Iyy * q, -Ixx * p, 0]])

    # Added mass Coriolis matrix CA
    C_A = np.array([[0, 0, 0, 0, Zwd * w_r, Yvd * v_r],
                    [0, 0, 0, -Zwd * w_r, 0, -Xud * u_r],
                    [0, 0, 0, -Yvd * v_r, Xud * u_r, 0],
                    [0, -Zwd * w_r, Yvd * v_r, 0, -Nrd * r, Mqd * q],
                    [Zwd * w_r, 0, -Xud * u_r, Nrd * r, 0, -Kpd * p],
                    [-Yvd * v_r, Xud * u_r, 0, -Mqd * q, Kpd * p, 0]])
 
    Ck = C_RB + C_A  
    
    # Hydrodynamic damping
    D11 = np.diag([Xu + Xuu*np.abs(u_r), Yv + Yvv*np.abs(v_r), Zw+Zww*np.abs(w_r)])
    D22 = np.diag([Kp + Kpp*np.abs(p), Mq+Mqq*np.abs(q), Nr + Nrr*np.abs(r)])
    Dk = np.block([[D11, np.zeros((3, 3))],[np.zeros((3, 3)), D22]])
    
    # Buoyancy forces and moments
    gk = np.array([
        (W - B) * sin2,
        -(W - B) * cos2 * sin1,
        -(W - B) * cos2 * cos1,
        zg * W * cos2 * sin1,
        zg * W * sin2,
        0
    ]).reshape(6,1)
    
    H = (np.array([tau_u, tau_v, tau_w, tau_p, tau_q, tau_r])).reshape(6,1)
    # Compute state vector derivative
    term1 = Jk @ nu_r 
    term2 = -np.linalg.inv(M) @ (C_RB @ nu + C_A  @ nu_r + Dk @ nu_r + gk)
    g_xi = np.vstack([term1, term2]) 
            
    h_xi_u = np.vstack([np.zeros((6, 1)), np.linalg.inv(M) @ H])
    xidot = g_xi + h_xi_u
    
    return xidot, M, Jk, Ck, Dk, gk


# Simulate the velocity MPC Controller with function:

def proposed2(eta_ini, nv_ini, yref, nu_c, Ts, Tf, noise, tau_wave):
    # This is the complete velocity form SD/LPV-MPC algorithm
    # This function returns the inouts, states and output variables
    # Available at: https://d197for5662m48.cloudfront.net/documents/publicationstatus/205391/preprint_pdf/dc7949b86f1bc8816fcceef10891d8dd.pdf
    NoS = round(Tf / Ts)            # Simulation duration
    nx = 12                         # number of state variables
    nu = 6                          #number of inputs. NB: This refers to velocity in the AUV model function
    ny = 12                          # number of outputs
    N = 30                          # Prediction horizon
    Nu = 3                        # control horizon

    # Memory locations
    x = np.zeros((nx, (NoS)))    
    x[:, 0] = np.concatenate((eta_ini, nv_ini))
    y = np.zeros((ny, NoS))
    tau = np.zeros((nu, NoS))       
    dx = np.zeros((nx, NoS))
    tau_ = np.zeros((nu, 1))        # previous input signal
    y_ = np.zeros((ny, 1))          # previous output signal
    x_ = np.zeros((nx, 1))          # previous state measurement
    G = np.hstack([np.eye(12)])    # Output measurement matrix

    # State and input constraints
    xmax = (np.array([1e12, 1e12, 1e12, 1e12, 2*np.pi/5, 1e12, 10, 10, 10, 1e12, 1e12, 1e12])).reshape(12,1)
    # The velocity of the vehicles showed be constrained to avoid excessively high speed navigation
    # At high-speed, coupling effects become greater
    xmin = -xmax
    tau_max = (1000 * np.array([2, 2, 2, 1, 1, 1])).reshape(6,1) # Define constraints on the input forces
    Xmax = np.kron(np.ones((N,1)), xmax)
    Xmin = np.kron(np.ones((N,1)), xmin)
    Taumax = np.kron(np.ones((Nu,1)), tau_max)
    Taumin = np.kron(np.ones((Nu,1)), -tau_max)

    # Weight matrices for MPC cost function
    Q = block_diag(1, 1, 1, 1, 1, 1, .005, .005, .005, .5, .5, .5)*1500 # np.block([[1000 * np.eye(12)]])
    R = block_diag(1, 1, 1, 1, 1, 1)*0.5
    S = 1 * Q # Terminal output weighting matrix

    J = np.zeros(NoS)

    for k in range(NoS-1):
        # Get current measurement
        y[:, k] = G @ x[:, k] 
        yref_p = np.kron(np.ones(N), yref[:, k])  # Trajectory prediction        
        dx[:, k] = (x[:, k].reshape(-1,1) - x_).reshape(-1)

        # Update AUV model
        _, M, Jk, Ck, Dk, _ = BlueROV_sim(x[:, k], tau_, 0*nu_c[:, k], 0*tau_wave) 
        # zero multiply disturbance beccause they are unknown to the controller
        Ak = np.block([
            [np.eye(6), Jk * Ts],
            [np.zeros((6, 6)), np.eye(6) - np.linalg.inv(M) @ (Ck + Dk) * Ts]
        ])
        Bk = np.block([[np.zeros((6, nu))], [np.linalg.inv(M) * Ts]])

        # Construct prediction model matrices
        A_bar = predA1(Ak, N)
        B_bar = predB1(Ak, Bk, N, Nu)
        G_bar = predW(G, N, N)
        I_y = np.kron(np.ones((N, 1)), np.eye(ny))
        A_p = G_bar @ A_bar
        B_p = G_bar @ B_bar
        Q_p = predQ(Q, S, N)
        R_p = np.kron(np.eye(Nu), R)
        H = 2 * (B_p.T @ Q_p @ B_p + R_p)
        Xi = I_y @ y_
        f = (2 * B_p.T @ Q_p @ (A_p @ dx[:, k].reshape(12,1) + Xi - yref_p.reshape((N*ny),1))).reshape(-1)
        

        Aqp = np.vstack([np.eye(Nu * nu), -np.eye(Nu * nu), B_bar, -B_bar])
        tau_p = np.kron(np.ones((Nu,1)), tau_)
        x_p = np.kron(np.ones((N,1)), x_)
        bqp = (np.concatenate([
            Taumax - tau_p,
            -Taumin + tau_p,
            Xmax - A_bar @ dx[:, k].reshape(nx,1) - x_p,
            -Xmin + A_bar @ dx[:, k].reshape(nx,1) + x_p
        ], axis=0)).reshape(-1)
        
        print(np.linalg.matrix_rank(Aqp))
        
        # Solve quadratic program using cvxopt
        H = matrix(((H+H.T)/2))         # Impose symmetry in the Hessian matrix
        f = matrix(f)
        Aqp = matrix(Aqp)
        bqp = matrix(bqp)
        solvers.options['show_progress'] = False
        solution = solvers.qp(H, f, Aqp, bqp)

        V = np.array(solution['x']).flatten()
        tau[:, k] = V[:nu] + tau_.flatten()

        # Apply input forces to the AUV dynamics
        x[:, k+1] = solver_RK(x[:, k], Ts, tau[:, k], nu_c[:, k], tau_wave)
        y_ = y[:, k].reshape(ny, 1)
        x_ = x[:, k].reshape(nx, 1)
        tau_ = tau[:, k].reshape(nu, 1)

        print(f"Iteration number = {k + 1} of {NoS}")
        
    xi = x
    yi = G @ x
    taui = tau

    return taui, xi, yi


# Call the functions with respective parameters
# tau1, x1, y1 = proposed1(eta_ini, nv_ini, yref, nu_c, Ts, Tf, noise)

tau_wave = 0
tau2, x2, y2 = proposed2(eta_ini, nv_ini, yref, nu_c, Ts, Tf, noise, tau_wave)

#tau3, x3, y3 = strategyI(eta_ini, nv_ini, yref, nu_c, Ts, Tf, noise)


tau2[:, -1] = tau2[:, -2]
# Create figure to plot results
plt.figure(figsize=(8, 8))

# Subplot 621
plt.subplot(6, 2, 1)
plt.step(t, yref[0, :], linestyle='-.', color='k', linewidth=1.2, label='yref')
plt.step(t, y2[0, :len(t)], color='g', linewidth=1.2, label='y_1')
plt.ylabel('$x$ [m]')
#plt.ylim([-0.4, 0.6])
plt.xlim(0, Tf)
plt.grid(True)

# Subplot 623
plt.subplot(6, 2, 3)
plt.step(t, yref[1, :], linestyle='-.', color='k', linewidth=1.2, label='yref')
plt.step(t, y2[1, :len(t)], color='g', linewidth=1.2, label='y1')
plt.ylabel('$y$ [m]')
#plt.ylim([-0.4, 0.6])
plt.xlim(0, Tf)
plt.grid(True)

# Subplot 625
plt.subplot(6, 2, 5)
plt.step(t, yref[2, :], linestyle='-.', color='k', linewidth=1.2, label='yref')
plt.step(t, y2[2, :len(t)], color='g', linewidth=1.2, label='y1')
plt.ylabel('$z$ [m]')
#plt.ylim([-0.15, 0.02])
plt.xlim(0, Tf)
plt.grid(True)

# Subplot 627
plt.subplot(6, 2, 7)
plt.step(t, yref[3, :], linestyle='-.', color='k', linewidth=1.2, label='yref')
plt.step(t, y2[3, :len(t)], color='g', linewidth=1.2, label='y1')
plt.ylabel('$\\phi$ [rad]')
#plt.ylim([-0.15, 0.1])
plt.xlim(0, Tf)
plt.grid(True)

# Subplot 629
plt.subplot(6, 2, 9)
plt.step(t, yref[4, :], linestyle='-.', color='k', linewidth=1.2, label='yref')
plt.step(t, y2[4, :len(t)], color='g', linewidth=1.2, label='y1')
plt.ylabel('$\\theta$ [rad]')
#plt.ylim([-0.04, 0.04])
plt.xlim(0, Tf)
plt.grid(True)

# Subplot 6,2,11
plt.subplot(6, 2, 11)
plt.step(t, yref[5, :], linestyle='-.', color='k', linewidth=1.2, label='yref')
plt.step(t, y2[5, :len(t)], color='g', linewidth=1.2, label='y1')
plt.ylabel('$\\psi$ [rad]')
plt.xlabel('Time [s]')
#plt.ylim([-0.15, 0.15])
plt.xlim(0, Tf)
plt.grid(True)

# Subplot 622
plt.subplot(6, 2, 2)
plt.step(t, tau2[0, :len(t)], color='g', linewidth=1.2)
plt.ylabel('$\\tau_X$ [N]')
plt.xlim(0, Tf)
plt.grid(True)

# Subplot 624
plt.subplot(6, 2, 4)
plt.step(t, tau2[1, :len(t)], color='g', linewidth=1.2)
plt.ylabel('$\\tau_Y$ [N]')
plt.xlim(0, Tf)
plt.grid(True)

# Subplot 626
plt.subplot(6, 2, 6)
plt.step(t, tau2[2, :len(t)], color='g', linewidth=1.2)
plt.ylabel('$\\tau_Z$ [N]')
plt.xlim(0, Tf)
plt.grid(True)

# Subplot 628
plt.subplot(6, 2, 8)
plt.step(t, tau2[3, :len(t)], color='g', linewidth=1.2)
plt.ylabel('$\\tau_K$ [Nm]')
plt.xlim(0, Tf)
plt.grid(True)

# Subplot 6,2,10
plt.subplot(6, 2, 10)
plt.step(t, tau2[4, :len(t)], color='g', linewidth=1.2)
plt.ylabel('$\\tau_M$ [Nm]')
plt.xlim(0, Tf)
plt.grid(True)

# Subplot 6,2,12
plt.subplot(6, 2, 12)
plt.step(t, tau2[5, :len(t)], color='g', linewidth=1.2)
plt.xlabel('Time [s]')
plt.ylabel('$\\tau_N$ [Nm]')
plt.xlim(0, Tf)
plt.grid(True)


plt.figure()

# Subplot 1
plt.subplot(311)
plt.step(t, x2[6, :len(t)], '-r', linewidth=1.5)  
plt.ylabel(r'$u$ [m/s]', fontsize=12)
plt.grid(True)
plt.xlim(0, Tf)

# Assuming tau2 has at least two columns
tau2[:, -1] = tau2[:, -2]

# Subplot 2
plt.subplot(312)
plt.step(t, x2[7, :len(t)], '-r', linewidth=1.5)  
plt.ylabel(r'$v$ [m/s]', fontsize=12)
plt.grid(True)
plt.xlim(0, Tf)

# Subplot 3
plt.subplot(313)
plt.step(t, x2[8, :len(t)], '-r', linewidth=1.5) 
plt.ylabel(r'$w$ [m/s]', fontsize=12)
plt.grid(True)
plt.xlim(0, Tf)



# Display the figure
plt.tight_layout()
plt.show()