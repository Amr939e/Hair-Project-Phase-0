import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import interp1d
from Support import Distributions
from Support import T
from Support import PoissonDisk
from  FrenetSolver import Frenet
import SkinHairmaps

class HairGen:
    def __init__(self,diameter, num_points, plot_points, guide_factor, guides_used, parting_vars):
        self.diameter = diameter
        self.center = [self.diameter/2, self.diameter/2]
        self.R_disk = self.diameter/ 2
        self.num_points = num_points
        self.plot_points = plot_points
        self.r_points = None

        self.guide_factor = guide_factor
        self.guides_used = guides_used
        self.guide_and_vars = None
        self.others_and_vars = None
        self.param_matrix = None
        self.num_guide_hairs = None

        self.T = T()
        self.parting_curve = lambda s: self.T.gc_comp_func(parting_vars, self.R_disk, s)
        self.Jacobian = lambda s: self.T.gc_comptan_func(parting_vars, self.R_disk, s)
        self.lengths = None
      
################################ KEY FUNCTIONS ######################################################

    def h_inverse_func(self, f, x):
        x_interp = np.linspace(-self.R_disk, self.R_disk, 1000)
        y_interp = f(x_interp)

        index = np.argsort(y_interp)
        sort_y = y_interp[index]
        sort_x = x_interp[index]
        inv_f = interp1d(sort_y, sort_x, kind='cubic')
        return inv_f(x)

    def Dance_func(self,params,A,x):
        a,c,d,N  = params[0], params[1], params[2], params[3]
        sq2 = np.sqrt(2)
        r_0 = (c-d)/(2*a)
        m = -(N * np.pi) / (a * sq2)
        if N < 1 or not isinstance(N, int):
            raise ValueError(f'ERROR: Order of function (N = {N}) must be >= 1 and must be an integer')
        if A > 1/abs(m) or A<0:
            raise ValueError(f'ERROR:Amplitude must be in the interval [0,{1/abs(m)}], for Dance function of order {N}')

        l = lambda t: A  * np.sin(m * sq2 * t)
        h = lambda t: t - l(t)/np.sqrt(2)
        g = lambda t: t + a + d / r_0 + l(t) / sq2

        t_values = self.h_inverse_func(h,x)
        unscaled = g(t_values)
        return r_0 * unscaled

    def parting_func(self, P):
        v = np.linspace(0.01,1,400)
        curve_points = self.parting_curve(v)
        deriv_curve_points = self.Jacobian(v)

        check = np.sum(deriv_curve_points[None,:,:]*P[:,None,:], axis =2)

        t = [ np.where( arr[:-1]*arr[1:] < 0)[0] for arr in check]
        arr = np.array(t, dtype=object)
        size = np.array([len(a) for a in arr])
        indices = np.where(size != 1)[0]
      
        if len(indices) >0:
            Test = np.linalg.norm(P[indices][:,None,:] - curve_points[None,:,:], axis =2)[:,0:396]
            index = np.argmin(Test, axis =1)
            arr[indices] = [ np.atleast_1d(x) for x in index]
        arr = arr.flatten()
        S = np.array([(v[arr[i]] + v[arr[i]+1]) / 2  for i in range(len(P))])
      
        parting_points = self.parting_curve(S)
        differences = P - parting_points
        norms = np.linalg.norm(differences, axis =1)
        index = np.where( norms < 1e-5)[0]
        if len(index) >0:
            norms[index] = 1
            differences[index] = np.array([0,1,0])
        D = differences/ norms[:,None]

        return D, norms, parting_points

    def length_func(self,k,z):
        params = self.param_matrix
        q = 0.8
        k_p = 1
        p = params[0][1]/ params[0][2]
        k_ave = (params[0][1]+params[0][2])/2

        if (1-q)*k_p < k_ave <= q*k_p:
            c_0 = (   (17*q-4)*k_p - 9* k_ave)/(2*q -1)
        else:
            c_0 =  6 * k_p

        Grav = lambda s: c_0 * self.R_disk * p ** (1 + (s / self.R_disk))
        grad = lambda s: - (k_p - s) ** 2

        k[np.where(k > q * k_p)[0]] = q * k_p
        k[np.where(k < (1-q) * k_p)[0]] = (1-q) * k_p
        length = (k * (1-p)*Grav(z) )  / -grad(k)
      
        return length,k, grad(k), Grav(z)

    def curvature_func(self,k,grad,grav,s):
        return k + (grad / grav) *s

    def other_vars_func(self, norms, vecs,theta, h_n):
        p = self.param_matrix
        guide_array, curtor, utv, theta_2A, direction_vecs = self.guide_and_vars
        d_t_sorted = np.sort(norms, axis=1)[:,0:h_n]
        index = np.argsort(norms, axis=1)[:, 0:h_n]

        d_pareto_t = (p[0][0] * self.r_points ** p[0][0]) / (d_t_sorted ** (p[0][0] + 1))
        weights_t = d_pareto_t / d_pareto_t.sum(axis=1)[:, None]
        theta_2B = (weights_t* theta_2A[index]).sum(axis=1)
        weighted_curtor = np.array((weights_t[:, :, None] * curtor[index] ).sum(axis=1))
        part_weight_angle = theta + theta_2B

        basis_vectors = self.T.rotation_tang_naxis(part_weight_angle, vecs)

        return weighted_curtor, basis_vectors
############## MAIN GENERATION ####################################################################################

    def assign_strand_vars(self,parting_line = False):
        self.num_guide_hairs = int(self.num_points * self.guide_factor)
        if self.guides_used > self.num_guide_hairs:
            raise ValueError(f' number of guides used must be less than or equal to {self.num_guide_hairs} given a guide factor of {self.guide_factor}')
        self.num_points = int(self.num_points / 2)
        self.num_guide_hairs = int(self.num_guide_hairs/2)

        Poisson = PoissonDisk(self.diameter, self.num_points)
        Poisson.num_guide_hairs = self.num_guide_hairs
        points_2d, cell_size = Poisson.poisson_disk_sampling()
        self.r_points = Poisson.r_points

        move = np.full((Poisson.num_gen_hairs,2), self.center)
        poi = np.subtract(np.array(points_2d), move)
        zoi = np.sqrt( self.R_disk**2 - poi[:,0]**2 - poi[:,1]**2)
        points_3d = np.column_stack((poi,zoi))

        #GUIDE HAIRS SETUP
        split_mask = np.where(points_3d[:, 1] < 0)[0]
        guide = np.random.choice(np.arange(len(split_mask)), size=int(self.num_guide_hairs/2), replace=False)
        guide_index = split_mask[guide]

        g_side1 = points_3d[guide_index]
        g_side2 = g_side1.copy()
        g_side2[:,1] *= -1
        guide_hairs = np.concatenate([g_side1, g_side2], axis = 0)

        mask = np.ones(Poisson.num_gen_hairs, dtype=bool)
        mask[guide_index] = False
        other_hairs = points_3d[mask]
        n2 = self.num_points - self.num_guide_hairs
      
        # CURVATURE AND TORSION SETUP
        p = self.param_matrix
        if p[0][0] < 1 or p[0][0] >2:
            raise ValueError(f' Curvature shape parameter must be in the interval [1,2]')
        elif p[1][0] < 1 or p[1][0] >2:
            raise ValueError(f' Torsion shape parameter must be in the interval [1,2]')
        elif p[0][1] == p[0][2] or p[0][2] <= p[0][1]:
            raise ValueError(f' Please ensure curvature interval has size greater than 0, and that the upper bound is greater than the lower')
        elif p[1][1] == p[1][2] or p[1][2] <= p[1][1]:
            raise ValueError(f' Please ensure torsion interval has size greater than 0, and that the upper bound is greater than the lower')
        elif p[1][0] <0 or p[1][1] >1:
            raise ValueError(f'Please ensure torsion bounds are in the interval [0,1]')
        elif p[1][1] > 0.7  and abs(p[0][1] - p[1][1]) >=0.3:
            raise ValueError(f' Please ensure the gap between the lower bounds is less than 0.3 given the higher torsion (subject to change depending on k_p)')

        curvature = Distributions(p[0][0], [p[0][1], p[0][2]], int(self.num_guide_hairs/2))
        curv, mean_curv = curvature.inv_Pareto()
        curv = np.tile( curv, 2) 

        torsion = Distributions(p[1][0], [p[1][1], p[1][2]], self.num_guide_hairs)
        tau, mean_tau = torsion.inv_Pareto()
        curtor = np.column_stack((np.array(curv), np.array(tau)))
      
        #DIRECTION SETUP
        guide_tan_uvecs = self.T.tangent_plane_arrays(guide_hairs, self.R_disk)
        guide_array = np.array(guide_hairs)

        if parting_line is True:
            direction_vecs, norms, parting_points = self.parting_func(guide_array)

            b_2 = np.sum(direction_vecs*guide_tan_uvecs[:,1], axis =1)
            b_3 = np.sum(direction_vecs * guide_tan_uvecs[:, 2], axis=1)
            theta_1A = np.atan2(-b_2,b_3)

            theta_2A = self.Dance_func([50, 0, np.pi / 2, 1], 10, guide_array[:,0])
            r_angle = theta_1A + theta_2A
            utv = self.T.rotation_tang_naxis(r_angle, guide_tan_uvecs)

        else:
            direction_vecs = np.tile([0,0,0],(self.num_guide_hairs,1))
            theta_2A = np.random.uniform(-2*np.pi,2*np.pi, self.num_guide_hairs)
            utv = self.T.rotation_tang_naxis(theta_2A, guide_tan_uvecs)

        self.guide_and_vars = guide_array, curtor,utv, theta_2A, direction_vecs
      
        #GUIDE HAIR PROPAGATION, PARETO WEIGHTS
        other_tan_uvecs = self.T.tangent_plane_arrays(other_hairs, self.R_disk)
        other_array = np.array(other_hairs)
        other_vecs = np.tile([0,0,0], (n2,1))

        if parting_line is True:
            weighted_curtor = []
            oav = []
            other_vecs, norms, other_ppoints = self.parting_func(other_array)

            condition1 = other_vecs[:, 1] < 0
            condition2 = direction_vecs[:, 1] < 0
            indices = [[np.where(condition1)[0], np.where(condition2)[0]],
                       [np.where(~condition1)[0], np.where(~condition2)[0]]]

            c_2 = np.sum(other_vecs * other_tan_uvecs[:, 1], axis=1)
            c_3 = np.sum(other_vecs * other_tan_uvecs[:, 2], axis=1)
            theta_1B = np.atan2(-c_2, c_3)

            for side in indices: # always 2 sides
                o_points = other_array[side[0]]
                vecs = other_tan_uvecs[side[0]]
                g_points = guide_array[side[1]]
                theta = theta_1B[side[0]]

                d = o_points[:, None, :] - g_points[None, :, :]
                d_norms = np.linalg.norm(d, axis=2)
                h_n = min(len(side[1]),int(np.floor(self.guides_used/2)))  
                wv, bv = self.other_vars_func(d_norms, vecs,theta,h_n)

                weighted_curtor.append(wv)
                oav.append(bv)
            weighted_curtor = np.concatenate(weighted_curtor)
            oav = np.concatenate(oav)
            other_array= np.concatenate([other_array[indices[0][0]],other_array[indices[1][0]]], axis =0)
            other_vecs = np.concatenate([other_vecs[indices[0][0]], other_vecs[indices[1][0]]], axis =0)

            self.others_and_vars = other_array, weighted_curtor, oav, other_vecs
        else:
            theta = np.zeros(n2)
            d_total = other_array[:, None, :] - guide_array[None, :, :]
            d_norms_t = np.linalg.norm(d_total, axis=2)
            weighted_curtor, oav= self.other_vars_func(d_norms_t, other_tan_uvecs, theta,self.guides_used)
            self.others_and_vars = other_array, weighted_curtor, oav, other_vecs

        points_array = np.concatenate([guide_array, other_array], axis =0)
        curtors = np.concatenate([curtor, weighted_curtor], axis =0)
        bases = np.concatenate([utv, oav], axis=0)
        direc_vecs = np.concatenate([direction_vecs, other_vecs], axis=0)
        self.t.checkpoint("Completed full other hair distribution")
        return points_array, curtors, bases, direc_vecs
#########################CURVATURE DISTRIBUTION PLOT#############################################################
    def practice_curvtor_plot(self):
        p = self.param_matrix
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi / 2, 100)

        Theta, Phi = np.meshgrid(theta, phi)
        x = self.R_disk * np.sin(Phi) * np.cos(Theta)
        y = self.R_disk * np.sin(Phi) * np.sin(Theta)
        z = self.R_disk * np.cos(Phi)

        fig = plt.figure(figsize=(10, 12))
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect([1, 1, 0.5])
        ax.plot_surface(x, y, z, color='cyan', alpha=0.3, zorder=1)

        points_array, curtors, bases, direc_vecs = self.assign_strand_vars()
        n = 2 * self.num_guide_hairs
        lis = np.linspace(p[0][1], p[0][2], n)
        c = np.linspace(0, 100, n)
        segments = list(zip(lis[:-1], lis[1:], c[:-1], c[1:]))   

        for j, (x0, x1,c0,c1) in enumerate(segments):
            mask = (curtors[:,0] > x0) & (curtors[:,0] <= x1)
            sel = points_array[mask]
            if sel.shape[0] == 0:
                continue
            colours = np.linspace(c0, c1, sel.shape[0])
            ax.scatter(sel[:, 0], sel[:, 1], sel[:, 2], c=colours, cmap='plasma', s=20)

        ax.set_title(f'Colour map according to curvature')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
###############FINAL PLOT##############################################################################
    def plot_hair_strands(self, param_matrix,haircolour, skincolour, parting_line = False ):
        self.param_matrix = param_matrix
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi / 2, 100)

        Theta, Phi = np.meshgrid(theta, phi)
        x = self.R_disk * np.sin(Phi) * np.cos(Theta)
        y = self.R_disk * np.sin(Phi) * np.sin(Theta)
        z =  self.R_disk * np.cos(Phi)

        fig = plt.figure(figsize=(10, 12))
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect([1, 1, 0.5])
        ax.plot_surface(x, y, z, cmap=SkinHairmaps.SKIN_CMAPS[skincolour], alpha=0.8, zorder=1)
      
        if parting_line is True:
            points_array, curtors, bases, direc_vecs = self.assign_strand_vars(parting_line = True)
        else:
            points_array, curtors, bases, direc_vecs = self.assign_strand_vars()

        self.lengths,curtors[:,0], grad, grav = self.length_func(curtors[:,0], points_array[:,2])
      
        average_curv = (param_matrix[0][1]+param_matrix[0][2])/2
        amp2 = (param_matrix[1][2] + param_matrix[1][1])/2 + average_curv/4
        tau_pert = lambda s:  amp2 * np.sin((average_curv*curtors[:,0]*s))
      
        curve = Frenet(lambda s: self.curvature_func(curtors[:,0], grad,grav, s),

                       lambda s: curtors[:,1] + tau_pert(s),
                        points_array)
        curve.all_points_with_vars = points_array, curtors, bases, direc_vecs
        curve.R_disk = self.R_disk
        curve.num_points = self.num_points
        curve.y0 = bases.reshape(-1, 9).flatten()
        curve.param_matrix = self.param_matrix

        v = np.linspace(0, 1, 1000)
        starts = np.zeros(self.num_points)
        curve.intervals = starts[:, None] + v[None, :] * self.lengths[:, None]

        curve.Frenet_solve([0, max(self.lengths)], 1000, orient=True)
        curve.compute_position([0,max(self.lengths)], 1000)

        plott = np.random.choice(np.arange(2*self.num_points), size=self.plot_points, replace=False)
        curve.r_sol = curve.r_sol[plott]
        xc = curve.r_sol[:, :,0]
        yc = curve.r_sol[:, :,1]
        zc = curve.r_sol[:, :,2]
        hairs = Line3DCollection(curve.r_sol, colors=SkinHairmaps.HAIR[haircolour], linewidths=2)

        ax.add_collection(hairs)
        ax.scatter(xc[:,0], yc[:,0], zc[:,0], color='g', s=5)
        ax.scatter(xc[:,-1], yc[:,-1], zc[:,-1], color='y', s=2)
        ax.set_title(f'Hair Model: Curvature = {[ param_matrix[0][1], param_matrix[0][2]]}, Torsion ={[param_matrix[1][1], param_matrix[1][2]]}')
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        self.t.checkpoint("Everything done")
        plt.show()
      
#EXAMPLE PLOTS
# P1 = HairGen(100,1500,1000, 1/5,300, [[0,0,1], [50,0,0], np.pi])
# P1.plot_hair_strands([[1.2, 0.29,0.36], [1.4, 0.2, 0.28]],'hair_11','skin_19', parting_line=True)
#P1.practice_curvtor_plot()
