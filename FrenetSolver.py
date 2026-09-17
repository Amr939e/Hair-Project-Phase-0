import numpy as np
import scipy
from scipy.interpolate import interp1d
from Support import T

class Frenet:
    def __init__(self,k_func, tau_func, points):
        self.k_func = k_func if callable(k_func) else lambda s: k_func
        self.tau_func = tau_func if callable(tau_func) else lambda s: tau_func
        self.points = points
        self.T_sol = None
        self.N_sol = None
        self.B_sol= None
        self.r_sol = None
        self.y0 = None

        self.all_points_with_vars = None
        self.R_disk = None
        self.r_points = None
        self.num_points = None
        self.T = T()
        self.intervals = None
        self.param_matrix = None

    def ode_func(self,s,y):
        k = self.k_func(s)
        tau = self.tau_func(s)
        t = np.arange(len(y))

        block = 3
        gap = 6
        T_mask = (t - 0) % (block + gap) < block
        N_mask = (t - 3) % (block + gap) < block
        B_mask = (t - 6) % (block + gap) < block
        T = y[np.where(T_mask)[0].reshape(self.num_points,3)]
        N = y[np.where(N_mask)[0].reshape(self.num_points,3)]
        B = y[np.where(B_mask)[0].reshape(self.num_points,3)]
        T_ = k[:,None] * N
        N_ = -k[:,None] * T + tau[:,None] * B
        B_ = -tau[:,None] * N
        t = np.column_stack([T_,N_,B_]).reshape(-1,9).flatten()
      
        return t

    def interp_func(self, point_data, full_t, new_times):
        f = interp1d(full_t, point_data, axis = 0, kind ='linear')
        return f(new_times)

    def Frenet_solve(self, s_span, n, orient = False):
        if not orient:
            T_0 = np.tile( [0,0,1], (self.num_points,1))
            N_0 = np.tile( [1,0,0], (self.num_points,1))
            B_0 = np.tile( [0,1,0], (self.num_points,1))
            y0 = np.column_stack([T_0,N_0,B_0]).reshape(-1,9).flatten()
            self.y0 = y0

        s_eval = np.linspace(s_span[0] , s_span[1],n)
        sol = scipy.integrate.solve_ivp(self.ode_func, s_span, self.y0,
                       method = 'DOP853', t_eval = s_eval, rtol = 1e-12, atol=1e-15, dense_output = True)

        s_fine = np.linspace(s_span[0], s_span[1], n * 10)
        sol_fine = sol.sol(s_fine)

        t = np.arange(len(sol_fine))
        block = 3
        gap = 6
        T_mask = (t - 0) % (block + gap) < block
        N_mask = (t - 3) % (block + gap) < block
        B_mask = (t - 6) % (block + gap) < block
        T = sol_fine[np.where(T_mask)[0].reshape(self.num_points, 3)].transpose(0, 2, 1)
        N = sol_fine[np.where(N_mask)[0].reshape(self.num_points, 3)].transpose(0, 2, 1)
        B = sol_fine[np.where(B_mask)[0].reshape(self.num_points, 3)].transpose(0, 2, 1)

        T_ortho, N_ortho, B_ortho = self.T.orthonormalize_array_frame(T, N, B)  #around 1e-14 error, just good practice

        self.T_sol = T_ortho
        self.N_sol = N_ortho
        self.B_sol = B_ortho

        return self.T_sol, self.N_sol, self.B_sol

    def compute_position(self,s_span,n):
        if self.T_sol is None:
            self.Frenet_solve(s_span,n)

        s_eval = np.linspace(s_span[0], s_span[1], 10*n)
        ds = s_eval[1] - s_eval[0]
        points_array, curtors, bases, direc_vecs = self.all_points_with_vars
        self.r_sol = np.zeros((self.num_points,10*n,3))
        self.r_sol[:,0,:] = points_array

        p = self.param_matrix
        constant = 0.3 * np.exp( -0.6*(p[0][1]+p[0][2])/2 )
        c_pareto = (p[0][0] * (np.min(curtors[:,0])) ** p[0][0]) / (curtors[:, 0] ** (p[0][0] + 1))
        weights = 2e-3*self.num_points*c_pareto / c_pareto.sum(axis=0)
        N = 200
        
        for i in range(1,N+1):
            T_avg = (self.T_sol[:, i-1, :] + self.T_sol[:, i, :]) / 2
            self.r_sol[:, i,:] = self.r_sol[:, i - 1,:] + T_avg * ds
            
        for i in range(N+1,10*n):
            T_avg = (self.T_sol[:, i - 2, :] + 4 * self.T_sol[:, i -1, :] + self.T_sol[:, i, :]) / 6 
            self.r_sol[:, i, :] = self.r_sol[:, i - 2, :] + T_avg * 2 * ds
    
            z = self.R_disk**2 - self.r_sol[:,i,0]**2 - self.r_sol[:,i,1]**2
            mask = z>0
            sub = constant*(abs(self.R_disk -self.r_sol[:, i, 2])).sum(axis=0) / self.num_points
            self.r_sol[:,i,2][mask] -= weights[mask]*(self.r_sol[:,i,2][mask] - np.sqrt(z[mask]))
            self.r_sol[:, i, 2][~mask] -= weights[~mask] *sub

        vec_interp = np.vectorize(self.interp_func, signature='(m,3),(t),(n)->(n,3)')
        self.r_sol = vec_interp(self.r_sol, s_eval, self.intervals)

        curves_mirr = self.r_sol.copy()
        curves_mirr[:, :, 1] *= -1
        self.r_sol = np.concatenate([self.r_sol, curves_mirr], axis =0)

        return self.r_sol
