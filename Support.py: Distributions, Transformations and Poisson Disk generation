import numpy as np
from scipy import stats
from scipy.optimize import root
import random
from matplotlib import pyplot as plt
from matplotlib import patches


class Distributions:
    def __init__(self, param, interval, num):
        self.param = param
        self.interval = interval
        self.num = num

    def inv_Pareto(self):
        lower, upper = self.interval[0], self.interval[1]
        F_lower = stats.pareto.cdf(lower, self.param, scale=lower)
        F_upper = stats.pareto.cdf(upper, self.param, scale=lower)

        u = np.random.random(self.num)
        u1 = F_lower + u * (F_upper - F_lower)
        x = stats.pareto.ppf(u1, self.param, scale=lower)
        mean = stats.pareto.mean(self.param, scale = lower)
        return x, mean

    def inv_normal(self, sigma):
        lower, upper = self.interval[0], self.interval[1]
        F_lower = stats.norm.cdf(lower, self.param, scale=sigma)
        F_upper = stats.norm.cdf(upper, self.param, scale=sigma)

        u = np.random.random(self.num)
        u1 = F_lower + u * (F_upper - F_lower)
        x = stats.norm.ppf(u1, self.param, scale=sigma)
        return x

class T:
    def rotation_tang_naxis(self, theta, vecs):
        e1, e2, e3 = vecs[:,0], vecs[:,1], vecs[:,2]

        new_e1 = e1
        new_e2 = np.cos(theta)[:,None] * e2 + np.sin(theta)[:,None] * e3
        new_e3 = -np.sin(theta)[:, None]*e2 + np.cos(theta)[:, None]*e3

        return np.stack([new_e1, new_e2, new_e3], axis = 1)
      
    def tangent_plane_arrays(self, P,R):
        P = np.array(P)
        e3 = P / R

        vx = np.array([0,0,1])
        v1 = np.cross(e3, vx)

        v1 = v1/np.linalg.norm(v1, axis = 1)[:,None]
        v2 = np.cross(e3,v1)

        norms = np.linalg.norm(v2, axis = 1)
        e1 = v2 / norms[:, None]
        e2 = np.cross(e3,e1)

        T0 = e3
        N0 = e1
        B0 = e2
        tan_unit_vecs = np.stack([T0,N0,B0], axis = 1)
        return tan_unit_vecs

    def orthonormalize_array_frame(self,T,N,B):

        T_norm = np.linalg.norm(T, axis=2)
        T_norm = np.maximum(T_norm, 1e-15)
        T_ = T / T_norm[:,:,None]

        N_ = N - np.sum(N * T_, axis=2)[:,:,None] * T_
        N_norm = np.linalg.norm(N_, axis=2)
        N_norm = np.maximum(N_norm, 1e-15)
        N_ = N_ / N_norm[:,:,None]

        B_ = np.cross(T_, N_, axis=2)
        B_norm = np.linalg.norm(B_, axis=2)
        B_norm = np.maximum(B_norm, 1e-15)
        B_ = B_ / B_norm[:,:,None]

        return T_, N_, B_

    def gc_comp_func(self,parting_vars,R,s):
        n1 = np.cos(parting_vars[2]*s)
        n2 = np.sin(parting_vars[2]*s)
      
        x = parting_vars[1][0]*n1+ parting_vars[0][0]*R*n2
        y = parting_vars[1][1]*n1+ parting_vars[0][1]*R*n2
        z = parting_vars[1][2]*n1+ parting_vars[0][2]*R*n2
      
        return np.column_stack([x, y, z])
                                                                              #parting_vars = initial direction, initial point, frequency
    def gc_comptan_func(self, parting_vars,R, s):
        n1 = np.cos(parting_vars[2] * s)
        n2 = np.sin(parting_vars[2] * s)
      
        x = parting_vars[0][0]*R*parting_vars[2]*n1 - parting_vars[2]*parting_vars[1][0] * n2
        y = parting_vars[0][1]*R*parting_vars[2]*n1 - parting_vars[2]*parting_vars[1][1] * n2
        z = parting_vars[0][2]*R*parting_vars[2]*n1 - parting_vars[2]*parting_vars[1][2] * n2
      
        return np.column_stack([x/parting_vars[2], y/parting_vars[2], z/parting_vars[2]])


class PoissonDisk:
    def __init__(self, width, num_points):
        self.width = width
        self.num_points = num_points
        self.center = [self.width / 2, self.width / 2]
        self.R_disk = self.width / 2
        self.constant = 1.4644786   
        self.r_points = self.constant * self.R_disk / np.sqrt(self.num_points)
        self.num_gen_hairs = None
        self.num_guide_hairs = None

    def poisson_disk_sampling(self, k=45):
        """Generate Poisson disk samples in 2D."""
        cell_size = self.r_points / np.sqrt(2)
        cols = int(self.width / cell_size) + 1
        rows = int(self.width / cell_size) + 1

        grid = [[None for _ in range(rows)] for _ in range(cols)]

        samples = []
        active = []
      
        t1 = random.uniform(0, self.width)
        sqrt = np.sqrt(self.R_disk**2 - (t1 - self.center[0])**2)
        b1 = self.center[1] - sqrt
        b2 = self.center[1] + sqrt
        p0 = (t1, random.uniform(b1, b2))
        samples.append(p0)
        active.append(p0)
        grid[int(p0[0] / cell_size)][int(p0[1] / cell_size)] = p0

        while active and len(samples) < self.num_points:
            idx = random.randint(0, len(active) - 1)
            p = active[idx]
            found = False

            for _ in range(k):
           
                angle = random.uniform(0, 2 * np.pi)
                dist = random.uniform(self.r_points, 2 * self.r_points)
                q = (p[0] + dist * np.cos(angle), p[1] + dist * np.sin(angle))
              
                if (q[0] - self.center[0])**2 + (q[1] - self.center[1])**2 < self.R_disk**2:        
                    cell_x = int(q[0] / cell_size)
                    cell_y = int(q[1] / cell_size)

                    valid = True
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = cell_x + dx, cell_y + dy
                            if 0 <= nx < cols and 0 <= ny < rows:
                                neighbor = grid[nx][ny]
                                if neighbor:
                                    dx_val = q[0] - neighbor[0]
                                    dy_val = q[1] - neighbor[1]
                                    if dx_val * dx_val + dy_val * dy_val < self.r_points * self.r_points:
                                        valid = False
                                        break
                        if not valid:
                            break

                    if valid:
                        samples.append(q)
                        active.append(q)
                        grid[cell_x][cell_y] = q
                        found = True
                        break

            if not found:
                active.pop(idx)
        self.num_gen_hairs = self.num_points - int(self.num_guide_hairs/2)
        choice = np.random.choice(np.arange(len(samples)), size=self.num_gen_hairs, replace=False)
        samples = np.array(samples)
        return samples[choice].tolist(), cell_size

    def draw_grid_with_points(self):
        sam, cell_size = self.poisson_disk_sampling()
        fig, ax = plt.subplots(figsize=(10, 10))
        circle = plt.Circle((self.width/2, self.width/2), self.R_disk, color = 'b', alpha = 0.1, linestyle = '--', linewidth = 1.2)
        ax.add_patch(circle)

        cols = int(self.width / cell_size) + 1
        rows = int(self.width / cell_size) + 1

        for x in range(cols + 1):
            ax.axvline(x * cell_size, color='lightgray', linewidth=0.5, alpha=0.5)
        for y in range(rows + 1):
            ax.axhline(y * cell_size, color='lightgray', linewidth=0.5, alpha=0.5)

        if sam:
            x_c = [p[0] for p in sam]
            y_c = [p[1] for p in sam]
            ax.scatter(x_c, y_c, c='blue', s=20, zorder=5)

        for p in sam:
            circle = patches.Circle(p, self.r_points, fill=False, color='red',
                                    linewidth=0.5, alpha=0.3)
            ax.add_patch(circle)

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.width)
        ax.set_aspect('equal')
        ax.set_title(f'Poisson Disk Sampling (radius={self.r_points:.3f}) within a Disk of radius: {self.R_disk}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(False)
        plt.show()
