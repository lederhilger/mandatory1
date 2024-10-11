import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse = False):
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xji, self.yij = ...
        self.L = 1.
        x = np.linspace(0,self.L,N+1)
        y = np.linspace(0,self.L,N+1)
        mesh = np.meshgrid(x, y, indexing = "ij")
        self.xij, self.yij = mesh
        self.nx = len(self.xij); self.ny = len(self.xij[0])
        self.h = (self.L/(self.nx-1) + self.L/(self.ny-1))/2
        #raise NotImplementedError

    def D2(self, N):
        """Return second order differentiation matrix"""
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (self.nx, self.ny), 'lil')
        D2[0, :4] = 2, -5, 4, -1
        D2[-1, -4:] = -1, 4, -5, 2
        return D2
        raise NotImplementedError

    @property
    def w(self):
        """Return the dispersion coefficient"""
        kx = self.mx*np.pi; ky = self.my*np.pi
        k_abs = np.sqrt(kx**2 + ky**2)
        w = k_abs*self.c
        return w
        raise NotImplementedError

    def ue(self, mx, my):
        """Return the exact standing wave"""
        ue = sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)
        return ue

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.mx = mx; self.my = my
        self.Unm1, self.Un, self.Unp1 = np.zeros((3, self.nx, self.ny))
        self.Unm1[:] = sp.lambdify((x, y, t), self.ue(mx, my))(self.xij, self.yij, 0)
        #raise NotImplementedError

    @property
    def dt(self):
        """Return the time step"""
        dt = self.courant*self.h/self.c
        return dt
        raise NotImplementedError

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        dx = self.L/(self.nx-1); dy = self.L/(self.ny-1)
        UE = sp.lambdify((x, y, t), self.ue(self.mx, self.my))(self.xij, self.yij, t0)
        l2_error = np.sqrt(dx*dy*np.sum((UE - u)**2))
        return l2_error
        raise NotImplementedError

    def apply_bcs(self):
        self.Unp1[0] = self.Unp1[-1] = 0
        self.Unp1[:, 0] = self.Unp1[:, -1] = 0
        #raise NotImplementedError

    def __call__(self, N, Nt, courant=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        courant : number
            The COURANT number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.courant = courant; self.c = c
        self.create_mesh(N); self.initialize(N, mx, my)
        D = self.D2(N)/self.h**2
        self.Un[:] = self.Unm1[:] + .5*(self.c*self.dt)**2*(D @ self.Unm1 + self.Unm1 @ D.T)
        plotdata = {0: self.Unm1.copy()}; l2_error = []
        t = self.dt
        if store_data == 1:
            plotdata[1] = self.Un.copy()
            l2_error.append(self.l2_error(self.Un, t))
        for n in range(1, Nt):
            self.Unp1[:] = 2*self.Un - self.Unm1 + (self.c*self.dt)**2*(D @ self.Un + self.Un @ D.T)
            # Set boundary conditions
            self.apply_bcs()
            # Swap solutions
            self.Unm1[:] = self.Un
            self.Un[:] = self.Unp1
            t += self.dt
            if n % store_data == 0:
                plotdata[n] = self.Unm1.copy() # Unm1 is now swapped to Un
                l2_error.append(self.l2_error(self.Un, t))
        if store_data == -1:
            return self.h, l2_error
        else:
            return plotdata
        raise NotImplementedError

    def convergence_rates(self, m=4, courant=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        courant : number
            The courant number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            print(f"m: {m}")
            dx, err = self(N0, Nt, courant=courant, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (self.nx, self.ny), 'lil')
        D2[0, :2] = -2, 2
        D2[-1, -2:] = 2, -2
        return D2
        raise NotImplementedError

    def ue(self, mx, my):
        ue = sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)
        return ue
        raise NotImplementedError

    def apply_bcs(self):
        pass
        #raise NotImplementedError

    def Animate(self, N, Nt = 10, mx = 2, my = 2):
        import matplotlib.animation as animation
        from matplotlib import cm, rc
        rc("text", usetex = True)
        data = self(N, Nt, mx = mx, my = my, store_data = 1)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        frames = []
        for n, val in data.items():
            #frame = ax.plot_wireframe(self.xij, self.yij, val, rstride=2, cstride=2);
            frame = ax.plot_surface(self.xij, self.yij, val, vmin=-0.5*data[0].max(),
                                    vmax=data[0].max(), cmap=cm.YlGn,
                                    linewidth=0, antialiased=False)
            frames.append([frame])

        animate = animation.ArtistAnimation(fig, frames, interval=400, blit=True, repeat_delay=1000)
        animate.save('NeumannWave.gif', writer='pillow', fps=24)

def test_convergence_wave2d():
    solD = Wave2D()
    r, E, h = solD.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    solD = Wave2D()
    r, E, h = solD.convergence_rates(m = 4, courant = np.sqrt(2)/2, mx = 1, my = 1)
    assert abs(E[-1]) < 1e-6
    #raise NotImplementedError

def makeAnimation():
    Wave2D_Neumann().Animate(32, 48)
makeAnimation()