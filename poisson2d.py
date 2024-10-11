import numpy as np
import sympy as sp
import scipy.sparse as sparse

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xij, self.yij ...
        x = np.linspace(0,self.L,N+1)
        y = np.linspace(0,self.L,N+1)
        mesh = np.meshgrid(x, y, indexing = "ij")
        self.xij, self.yij = mesh
        self.nx = len(self.xij); self.ny = len(self.xij[0])
        self.h = (self.L/(self.nx-1) + self.L/(self.ny-1))/2
        print("create_mesh")
        #raise NotImplementedError

    def D2(self):
        """Return second order differentiation matrix"""
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (self.nx, self.ny), 'lil')
        D2[0, :4] = 2, -5, 4, -1
        D2[-1, -4:] = -1, 4, -5, 2
        print("D2")
        return D2
        raise NotImplementedError

    def laplace(self):
        """Return vectorized Laplace operator"""
        D2 = self.D2()
        D2x = ((1/self.h)**2)*D2; D2y = ((1/self.h)**2)*D2
        laplace = (sparse.kron(D2x, sparse.eye(self.ny)) + sparse.kron(sparse.eye(self.nx), D2y))
        print("laplace")
        return laplace
        raise NotImplementedError

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        B = np.ones((self.nx, self.ny), dtype=bool)
        B[1:-1, 1:-1] = 0
        get_boundary_indices = np.where(B.ravel() == 1)[0]
        print("get_boundary_indices")
        return get_boundary_indices
        raise NotImplementedError

    def assemble(self):
        """Return assembled matrix A and right hand side vector b"""
        # return A, b
        bnds = self.get_boundary_indices()
        A = self.laplace()
        A = A.tolil()
        F = sp.lambdify((x, y), self.f)(self.xij, self.yij)
        UE = sp.lambdify((x, y), self.ue)(self.xij, self.yij)
        UE_ = UE.ravel()
        b = F.ravel()
        for i in bnds:
            A[i] = 0
            A[i, i] = 1
            b[i] = UE_[i]
        A = A.tocsr()
        print("A, b")
        return A, b
        raise NotImplementedError

    def l2_error(self, u):
        """Return l2-error norm"""
        dx = self.L/(self.nx-1); dy = self.L/(self.ny-1)
        UE = sp.lambdify((x,y), self.ue)(self.xij, self.yij)
        l2_error = np.sqrt(dx*dy*np.sum((UE - u)**2))
        print("l_2 error")
        return l2_error
        raise NotImplementedError

    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A, b.flatten()).reshape((self.nx, self.ny))
        print("self.U")
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

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
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
            print(f"convergence_rates {m}")
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, x, y):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        from scipy.interpolate import interpn
        u_interpolated = interpn((self.xij[:,0], self.yij[0,:]), self.U, np.array([x,y]), method = "cubic")
        print(f"u_interpolated: {u_interpolated}")
        return u_interpolated[0]
        raise NotImplementedError

def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2



def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h/2, y: 1-sol.h/2}).n()) < 1e-3