from dolfin import *
# 
#     Solve the 1D system of equations for the membrane-cortex mechanics

#             dX/dt + (eta/mu) d^2 V/dxi^2 + (1/mu)dSigma/dx - V  = 0
#             dX/dt - (k/mu )d^2 X/dxi^2 -v = 0
#             x(\xi = 0) = 0, and x(\xi = L) = \deltaL
#             v(\xi = 0) = 0, and v(\xi = L) = \bar v

#         \xi: the spatial coordinate
#         t : time coordinate
#         x  : Displacement field of the plasma membrane [um]
#         v  : velocity field of the actin cortex [um/s]
#         eta : 2D effective viscosity of the cortex [pN s / um]
#         k : Elastic constant of the membrane [pN/um]
#         mu : Friction coefficient of the cortex-membrane interaction [pN s/ um^3]

#     The trap force is simulated by a Dirichlet boundary condition on the displacement field x(\xi = L)
#     The active force that comes from the optogenetic manipulations can be simulated either by a Dirichlet bc on v(\xi =L) = \bar v
        
#     We solve the system for the displacement x(\xi, t) and velocity v(\xi, t) fields.

# 
class Solver:
    def __init__(
        self,
        L,
        n_elements,
        k,
        eta,
        mu,
        imposed_displacement,
        imposed_velocity,
        time_step,
    ):
        self.k = Constant(k)
        self.L = L
        self.eta = Constant(eta)
        self.mu = Constant(mu)
        self.imposed_displacement = imposed_displacement
        self.imposed_velocity = Expression('v', degree = 1, v = imposed_velocity)
        self.n_elements = n_elements
        self.time_step_length = time_step
        self.sigma_D = Expression('-(x[0]-L)*exp(-0.5*pow((x[0]-L)/std, 2))/pow(std,2)*strength', 
                         degree = 1, std = L/20, L = L, strength = 0)
        self.set_geometry_and_function_space()

    def set_geometry_and_function_space(self):
         # Definition of the geometry
        self.mesh = IntervalMesh(self.n_elements, 0, self.L)
         # Define a Function Space

        U = FiniteElement("CG", self.mesh.ufl_cell(), 2)
        V = FiniteElement("CG", self.mesh.ufl_cell(), 2)
        self.Q = FunctionSpace(self.mesh, U*V)

        self.q_ = TrialFunction(self.Q)
        self.dq_ = TestFunction(self.Q)
        (self.x_, self.v_) = split(self.q_)       # x_: displacement, v_: cortical velocity
        (self.dx_, self.dv_) = split(self.dq_)    # dx_, dv_ : test functions
        self.x_old = interpolate(Constant(0.0), self.Q.sub(0).collapse())
        self.q_solution = Function(self.Q)


    def set_boundary_conditions(self):
        def left_end(x, on_boundary):
            return near(x[0], 0) and on_boundary

        def right_end(x, on_boundary):
            return near(x[0], self.L) and on_boundary


        # self.bc = ([DirichletBC(self.Q, Constant((0.,0.)), left_end),
        #             DirichletBC(self.Q.sub(1), self.imposed_velocity, right_end),
        #             DirichletBC(self.Q.sub(0), self.imposed_displacement, right_end)])


        self.bc = ([DirichletBC(self.Q, Constant((0.,0.)), left_end),
                    DirichletBC(self.Q.sub(1), self.imposed_velocity, right_end)])
    
    def solve(self, bc):
        # self.set_geometry_and_function_space()
        # self.set_boundary_conditions()
        weak_form = (
                    ((self.x_ - self.x_old) / self.time_step_length) * self.dv_ * dx + \
                    - (self.eta / self.mu) * dot(grad(self.v_), grad(self.dv_)) * dx + \
                    (1 / self.mu) * (self.sigma_D) * self.dv_ * dx - self.v_ * self.dv_ * dx +  \
                    ((self.x_ - self.x_old) / self.time_step_length) * self.dx_ * dx + \
                    (self.k / self.mu) * dot(grad(self.x_), grad(self.dx_)) * dx - self.v_ * self.dx_ * dx                    
                )
            
        weak_form_lhs = lhs(weak_form)
        weak_form_rhs = rhs(weak_form)
        
        solve(weak_form_lhs == weak_form_rhs, self.q_solution, bc)
        self.x_s, self.v_s = self.q_solution.split(deepcopy=True)
        self.x_old = project(self.q_solution.sub(0),self.Q.sub(0).collapse())


