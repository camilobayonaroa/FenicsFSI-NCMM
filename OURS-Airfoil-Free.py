from __future__ import print_function
from mshr import *
from dolfin import *
import os
import numpy as np
from numpy import array, zeros, ones, any, arange, isnan
import ctypes, ctypes.util, numpy, scipy.sparse, scipy.sparse.linalg, collections

processID = MPI.rank(mpi_comm_world())
# get file name
fileName = os.path.splitext(__file__)[0]

p0file = File("%s.results/pressure0.pvd" % (fileName))
u0file = File("%s.results/velocity0.pvd" % (fileName))
p1file = File("%s.results/pressure1.pvd" % (fileName))
u1file = File("%s.results/velocity1.pvd" % (fileName))

# Time parameters and init condition
dt = Constant(1e-3)
T = 1.

#Solid's parameters
rho_s = 1e3     # Density, in tonne/m^3
nu_s = 0.4      # Poisson ration, in 1
mu_s = 1e5    	# in ton/(ms^2), original value 2e3
lam_s = 2*nu_s*mu_s / (1-2*nu_s)

#Fluid's parameters
rho = Constant(1e1) 
mu = Constant(1e-1) 
la = Constant(1.0) 
beta = Constant(1e8)    #Working with 1.0 
theta = [1,-1,0] 

#Fluid's stabilization parameters
#VMS
SC1 = 12.0
SC2 = 4.0
SC3 = 1.0
SC4 = 1.0
#Weak Imposition
SW1 = 500
SW2 = 500
#Ghost Penalty
sigmag  = 0.1
su = 0.1
sp = -0.01

class InitialVelGradient(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
        values[2] = 0.0
        values[3] = 0.0
    def value_shape(self):
        return (2,2)

class InitialPGradient(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
    def value_shape(self):
        return (2,)

class InitialCondition(UserExpression):
    def eval(self, values, x):
        values[0] = 1.0
        values[1] = 0.0
        values[2] = 0.0
    def value_shape(self):
        return (3,)

ic=InitialCondition(degree =2)
iug=InitialVelGradient(degree =2)
ipg=InitialPGradient(degree =2)

#Geometric parameters and mesh
BoxLength = 10
BoxHeigth = BoxLength/2
resolution_back = 20

# Create list of polygonal domain vertices for the wing airfoil
domain_vertices = [Point(1, 0 , 0  ),
 Point( 0.986596,  0.004976,  0. ),
 Point( 0.927615,  0.020294,  0. ),
 Point( 0.826928,  0.043305,  0. ),
 Point( 0.693763,  0.067967,  0. ),
 Point( 0.540785,  0.088125,  0. ),
 Point( 0.382787,  0.098537,  0. ),
 Point( 0.234002,  0.092400,  0. ),
 Point( 0.112880,  0.069743,  0. ),
 Point( 0.032437,  0.038325,  0. ),
 Point( 0.000000,  0.000000,  0. ),
 Point( 0.017579, -0.016779, 0. ),
 Point( 0.043684, -0.023825, 0. ),
 Point( 0.126714, -0.029000, 0. ),
 Point( 0.243500, -0.025401, 0. ),
 Point( 0.383767, -0.018677, 0. ),
 Point( 0.537674, -0.012432, 0. ),
 Point( 0.688920, -0.006829, 0. ),
 Point( 0.822520, -0.003392, 0. ),
 Point( 0.925025, -0.001853, 0. ),
 Point( 0.985774, -0.001335, 0. ),
 Point( 1 , 0 , 0)]

blade = Polygon(domain_vertices);
blade = CSGRotation(blade,Point(0, 0),-0.05*pi)
blade = CSGTranslation(blade,Point(BoxLength/2, BoxHeigth/2))
resolution_front = resolution_back/4
rectangle = Rectangle(Point(0,0), Point(BoxLength,BoxHeigth))
back_mesh = generate_mesh(rectangle, resolution_back)
front_mesh = generate_mesh(blade, resolution_front)

i, j, k, l, m = indices(5)
delta = Identity(2)

# Forces
f = Constant((0.,-0.981))
# Inflow profile
#U = 1.0
#inflow_profile = Expression(('U_bar','0.0'),U_bar=U, degree=3)
inflow_profile = Expression(("4.0*U_bar*x[1]*(Heigth - x[1])/(Heigth*Heigth)", "0.0"),
                      degree=2, U_bar=0.3, Heigth = BoxHeigth)

class OutBoundary(SubDomain):
   def inside(self, x, on_boundary):
      return on_boundary 
 
class WallsBoundary(SubDomain):
   def inside(self, x, on_boundary):
      return on_boundary and (near(x[1], 0.0) or near(x[1], BoxHeigth))
    
class InflowBoundary(SubDomain):
   def inside(self, x, on_boundary):
      return on_boundary and near(x[0], 0.0)

class OutflowBoundary(SubDomain):
   def inside(self, x, on_boundary):
      return on_boundary and near(x[0], BoxLength)

def tensor_jump(v,n):
   return outer(v('+'),n('+')) + outer(v('-'),n('-'))

def B_h(u, uk, v, rho, mu):  #Galerkin non-linear (uk) terms
   return mu*inner(grad(u), grad(v))*dX + rho*inner(dot(grad(u),uk),v)*dX 

def b_h(u, p, n):
   return -div(u)*p*dX + jump(u, n)*avg(p)*dI #Galerkin pressure + Cut interface Nitsche method

def W_h(u, v, n, h, mu, alpha): #Cut interface # Weak Imposition 
   return - mu*inner(avg(nabla_grad(u) + nabla_grad(u).T), tensor_jump(v, n))*dI \
        - mu*inner(avg(nabla_grad(v)+ nabla_grad(v).T), tensor_jump(u, n))*dI \
        + alpha/avg(h)*inner(jump(u), jump(v))*dI 

def l_h(v, f):
   return  inner(f, v)*dX

def s_O (u,p,v,q,h,beta): #Enforcement of front mesh's dirichlet solution to the background
   return (beta/avg(h)**2)*inner(jump(nabla_grad(u) + nabla_grad(u).T),jump(nabla_grad(v)+ nabla_grad(v).T))*dO  #Overlapping terms 

def s_GP1(u,v,gamma1): #Ghost penalty vel-pres terms only at cut elements
   return gamma1*inner(div(grad(u)),div(grad(v)))*dC

def s_GP2(p,q,gamma2): #Ghost penalty vel-pres terms only at cut elements
   return gamma2*inner(grad(p),grad(q))*dC

def l_GP1(grad_u_h,v,gamma1): #Ghost penalty vel-pres terms only at cut elements
   return gamma1*inner(div(grad_u_h),div(grad(v)))*dC

def solve_move():
    multimesh = MultiMesh()
    multimesh.add(back_mesh)
    multimesh.add(front_mesh)
    multimesh.build()

    # Define functionspace
    P3 = TensorElement("P", multimesh.ufl_cell(), 1) #Mainly for traction at the interface
    P2 = VectorElement("P", multimesh.ufl_cell(), 2) #Second order for velocity
    P1 = FiniteElement("P", multimesh.ufl_cell(), 1) #First order for pressure
    P0 = FiniteElement("Discontinuous Lagrange", multimesh.ufl_cell(), 0) 
    TH = MixedElement([P2, P1])
    W = MultiMeshFunctionSpace(multimesh, TH)   #Our multimesh functional space
    MP = MultiMeshFunctionSpace(multimesh, P0)   #Our multimesh functional space
    MP2 = MultiMeshFunctionSpace(multimesh, P2)   #Our multimesh functional space
    MP3 = MultiMeshFunctionSpace(multimesh, P3)   #Our multimesh functional space
 
    #Tractions space
    Fsig = FunctionSpace(back_mesh, P3)
    Ssig = FunctionSpace(front_mesh, P3)

    # Separated spaces for FSI
    V0 = VectorFunctionSpace(back_mesh, "Lagrange", 1)
    S0 = FunctionSpace(back_mesh, "Lagrange", 1)
    S00 = FunctionSpace(back_mesh, "Discontinuous Lagrange", 0)
    V1 = VectorFunctionSpace(front_mesh, "Lagrange", 1)
    S1 = FunctionSpace(front_mesh, "Lagrange", 1)
    vk0 = Function(V0)
    v0 = Function(V0)
    s0 = Function(S0)
    v1 = Function(V1)
    s1 = Function(S1)

    convel = Function(V0)
    unorm = Function(S00)
    chale = Function(S00)
    invcha = Function(S00)
    freto = Function(S00)
    timom = Function(S00)
    tidiv = Function(S00)
    alpha = Function(S00)
    gamma1 = Function(S00)
    gamma2 = Function(S00)
 
    # Initialize boundary conditions
    # Boundary on channel
    inflow   =InflowBoundary() 
    walls = WallsBoundary()
    outflow  =OutflowBoundary() 
    chanfunc = MeshFunction('size_t', back_mesh, back_mesh.topology().dim()-1)
    chanfunc.set_all(0)
    inflow.mark(chanfunc,1)
    walls.mark(chanfunc, 2)
    outflow.mark(chanfunc, 3)
    bc_inflow = MultiMeshDirichletBC(MultiMeshSubSpace(W,0), inflow_profile , chanfunc, 1, 0)
    bc_walls = MultiMeshDirichletBC(MultiMeshSubSpace(W,0), Constant((0, 0)), chanfunc, 2, 0)
    bc_out = MultiMeshDirichletBC(MultiMeshSubSpace(W,1), Constant(0.), chanfunc, 3, 0)
    # Boundary on Structure
    outbound   =OutBoundary() 
    outfunc = MeshFunction('size_t', front_mesh, front_mesh.topology().dim()-1)
    outfunc.set_all(0)
    outbound.mark(outfunc,1)
    cirbfunc = MeshFunction('size_t', front_mesh, front_mesh.topology().dim()-1)
    cirbfunc.set_all(1)
    bc_cirbu = MultiMeshDirichletBC(MultiMeshSubSpace(W,0), Constant((0, 0)), cirbfunc, 1, 1)
    bc_cirbp = MultiMeshDirichletBC(MultiMeshSubSpace(W,1), Constant((0)), cirbfunc, 1, 1)
    

    cells_s = MeshFunction('size_t', front_mesh, front_mesh.topology().dim())
    dsf = Measure("ds", domain=front_mesh, subdomain_data=outfunc)
    dV = Measure('dx', domain=front_mesh, subdomain_data=cells_s, metadata={'quadrature_degree': 2})

    #Solid's velocity for weak impossition 
    w_dir = MultiMeshFunction(W)
    w_dir.assign(interpolate(ic,W))

    # Non-linear and old solutions
    wk = MultiMeshFunction(W)
    wk.assign(interpolate(ic,W))

    w_old = MultiMeshFunction(W)
    w_old.assign(interpolate(ic,W))

    w_oold = MultiMeshFunction(W)
    w_oold.assign(interpolate(ic,W))

    # Define trial and test functions and right-hand side
    w = TrialFunction(W)
    z = TestFunction(W)
    
    dirvel, dummyq = split(w_dir)
    uk, pk = split(wk)
    u_oold, p_oold = split(w_oold)
    u_old, p_old = split(w_old)
    u, p = split(w)
    v, q = split(z)

    #Output initial fields
    output_sol(v0,s0,v1,s1)

    #Solid's displacement space
    nsol = FacetNormal(front_mesh)
    u_s = Function(V1)
    uk_s = Function(V1)
    u0_s = Function(V1)
    u00_s = Function(V1)
    del_u = TestFunction(V1)
    du = TrialFunction(V1)

    # Define facet normal and mesh size
    n = FacetNormal(multimesh)
    h = 2.0*Circumradius(multimesh)
    h_b = 2.0*Circumradius(back_mesh)

    # Solving linear system
    w_inc = MultiMeshFunction(W)

    #Ghost Penalty   
    grad_p_h = MultiMeshFunction(MP2)
    grad_p_h.assign(interpolate(ipg, MP2))
    grad_u_h = MultiMeshFunction(MP3)
    grad_u_h.assign(interpolate(iug, MP3))

    # Define solid deformation bilinear forms
    F_s = as_tensor(u_s[k].dx(i) + delta[k,i], (k,i) )
    J_s = det(F_s)
    C_s = as_tensor(F_s[k,i]*F_s[k,j], (i,j) )
    E_s = as_tensor(1./2.*(C_s[i,j]-delta[i,j]), (i,j) )
    S_s = as_tensor(lam_s*E_s[k,k]*delta[i,j] + 2.*mu_s*E_s[i,j], (i,j) )
    P_s = as_tensor(F_s[i,j]*S_s[j,k], (k,i) )
    
    bc_s = []

    if processID == 0: print('Starting transient simulation... \n')

    count =0
    tol = 1e-16     	# tolerance
    Coup_maxiter = 2 	# iteration limit		
    InitialTransient = 0.01 	# Until the pressure stabilizes, don't solve the solid	

    t = float(dt)

    while (t <= T):
	count += 1
	if processID == 0: print('SOLVING STEP:', count, ', AT TIME: ', t, '\n')
        t += float(dt)
       
        #Update fluid's fields
        w_oold.assign(interpolate(w_old,W))
        w_old.assign(interpolate(wk,W))

        #Update Solid's displacements
        u00_s.assign(interpolate(u0_s,V1))
	u0_s.assign(interpolate(u_s,V1))
	uk_s.assign(interpolate(u_s,V1))

        #Calculate linear variational terms
        Ftt= Constant(1/dt)*Constant(theta[1])*rho*inner(u_old, v)*dX 
        Fttt= Constant(1/dt)*Constant(theta[2])*rho*dot(u_oold, v)*dX 
        Ftt_s= -rho_s*2.*u0_s[i]/(dt*dt)*del_u[i]*dV 
        Fttt_s= rho_s*u00_s[i]/(dt*dt)*del_u[i]*dV 
 
        #Picard's scheme for coupling
	Coup_L2_abs = 1.0	# Initialize L2-norm for structure deformation (coupling measure)
	Coup_it = 0
	while Coup_L2_abs > tol and Coup_it < Coup_maxiter:

                Coup_it +=1
	        if processID == 0: print('     SOLVING COUPLING PICARD ITERATION:', Coup_it, '\n')

	        #Begin to Solve fluid's field
		if processID == 0: print('     Solving FLUID flow field...', '\n')
	
	        #Picard's scheme for non-linear fluid flow solution
	        L2_error = 1.0  # error measure ||u-u_k||
	        it = 0          # iteration counter
                maxiter = 3
	        while L2_error > tol and it < maxiter:
		
		    it += 1		
	
                    #Calculate mesh dependent algorithmic parameters
                    
                    #Variational Multiscale
                    convel.assign(project(wk.part(0).sub(0)))
                    unorm.assign(project(sqrt(abs(inner(convel,convel)))))
                    freto.assign(project(Constant(SC1)*Constant(mu)/(h_b*h_b)+Constant(SC2)*Constant(rho)*abs(unorm)/h_b))
                    timom.assign(project(conditional(lt(freto,1.0E-12),1.0E12,1.0/abs(freto))))
                    tidiv.assign(project(Constant(SC3)*Constant(mu)/Constant(rho)+Constant(SC4)*h_b*abs(unorm)))
            
                    timomMM = MultiMeshFunction(MP)
                    tidivMM = MultiMeshFunction(MP)
                    timomMM.interpolate(timom)
                    tidivMM.interpolate(tidiv)
            
                    #Weak Imposition 
                    alpha.assign(project(Constant(SW1)*Constant(mu)+Constant(SW2)*Constant(rho)*h_b*abs(unorm)))
                    alphaMM = MultiMeshFunction(MP)
                    alphaMM.interpolate(alpha)
            
                    #Ghost Penalty
                    gamma1.assign(project(Constant(sigmag)*Constant(su)*h_b*h_b/abs(timom)))
                    gamma2.assign(project(Constant(sigmag)*Constant(sp)*h_b*abs(timom)))
                    gamma1MM = MultiMeshFunction(MP)
                    gamma2MM = MultiMeshFunction(MP)
                    gamma1MM.interpolate(gamma1)
                    gamma2MM.interpolate(gamma2)
		
	            # Define temporal form
	            Ft= Constant(1/dt)*Constant(theta[0])*rho*inner(u,v)*dX 
	            # Define linear form
                    L= inner(f,v)*dX - Ftt - Fttt\
                        + l_GP1(grad_u_h,v,gamma1MM) 
	           
	            # Define Form 
	            #Global scheme + Weak Imposition + Ghost Penalty
                    a= B_h(u,uk,v,rho,mu) \
                     + W_h(u,v,n,h,mu,alphaMM) \
                     + b_h(v,p,n) + b_h(u,q,n)\
                     + s_O(u,p,v,q,h_b,beta)\
                     + s_GP1(u,v,gamma1MM) + s_GP2(p,q,gamma2MM)
	            F = Ft + a 
	            # Assemble linear system
	            A = assemble_multimesh(F)
	            b = assemble_multimesh(L)
	            bc_inflow.apply(A,b)
	            bc_walls.apply(A,b)
	            bc_out.apply(A,b)
	            bc_cirbu.apply(A,b)
	            bc_cirbp.apply(A,b)
	            W.lock_inactive_dofs(A,b)
	            #Solve linear system 
	            solve(A, w_inc.vector(), b)
	            vk0.assign(project(wk.part(0).sub(0),V0))
	            v0.assign(project(w_inc.part(0).sub(0),V0))
		    L2_error = assemble(((v0-vk0)**2)*dX)
		    print ('it=%d: L2-error=%g' % (it, L2_error))
	            #Update Picard's solution guess
	            wk.assign(w_inc)


                    # Compute FEM Projections for next iteration GPOP
                    u_inc, p_inc = split(w_inc)
        	    grad_p_h = project(grad(p_inc), MP2)
        	    grad_u_h = project(grad(u_inc), MP3)

		
		    if it == maxiter: print ('Solver did not converge!')
	
	        #Project solution onto separated fields
	        v1.assign(project(wk.part(1).sub(0),V1))
	        s0.assign(project(wk.part(0).sub(1),S0))
	        s1.assign(project(wk.part(1).sub(1),S1))
	
		if t > InitialTransient: 
		        # Compute fluid stress on current configuration
			d_ = as_tensor( 1./2.*( v0[i].dx(j)+v0[j].dx(i) ) , [i,j] )
			tau_ = as_tensor( la*d_[k,k]*delta[i,j] + 2.*mu*d_[i,j] , [i,j] )
			sigma_f = project(-s0*delta+tau_, Fsig, solver_type="mumps", \
			 	form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs"} )
		
			# Project the values on the structure domain in order to get the traction vector
			sigma_s = project(sigma_f, Ssig, solver_type="mumps", \
			 	form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs"} )
		
		
			# Compute the traction vector over the solid's surface (Updated Lagrange Configuration)
			P_f_ref = as_tensor(J_s * inv(F_s)[k,j] * sigma_s[j,i], (k,i) )
			t_s_hat = as_tensor(nsol[k] * P_f_ref[k,i], (i,) )

                        Drag = t_s_hat[0]*dsf
                        Lift = t_s_hat[1]*dsf
                        drag = assemble(Drag)
                        lift = assemble(Lift)
		
		        #Deform Solid's domain
			if processID == 0: print('\n     Solving STRUCTURE deformation...', '\n')
		
		        # Define Solid's Form 
			Form_s = rho_s*u_s[i]/(dt*dt)*del_u[i]*dV \
				+ P_s[k,i]*del_u[i].dx(k)*dV - rho_s*f[i]*del_u[i]*dV \
		                + Ftt_s + Fttt_s \
		                - t_s_hat[i]*del_u[i]*dsf     #Tractions over the boundary
		 
		        #Newton's method Jacobian
			Gain_s = derivative(Form_s, u_s, du)
		
		        #Newton's method solution
			solve(Form_s == 0, u_s, bc_s, J=Gain_s, \
				solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-3, "maximum_iterations":10000}},
				form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs"})			
			#dirvel = project((u_s-u0_s)/dt, V1, solver_type="mumps", form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs"})
	                #w_dir.assign(interpolate((u_s-u0_s)/dt,MultiMeshSubSpace(W,0)))
	
		        # Compute the L2-Norm of the structure	
		        Coup_L2_abs = assemble((((u_s-uk_s)**2.0)**0.5)*dV)
		
		        if processID == 0: print ('     Coupling Absolute L2-norm for convergence:', Coup_L2_abs, '\n')
	
	          	# Assign values for next Coupling-Iteration
		        uk_s.assign(u_s)
	
		        # Deform the solid's mesh
			for x in front_mesh.coordinates(): x[:] += u_s(x)[:]
			front_mesh.bounding_box_tree().build(front_mesh)
		
		        # Update overlapping meshes and new solid's functional spaces
			multimesh.build()
			
			# Define functionspace
		        W = MultiMeshFunctionSpace(multimesh, TH)
		        V1 = VectorFunctionSpace(front_mesh, "Lagrange", 1)
		        S1 = FunctionSpace(front_mesh, "Lagrange", 1)
		        v1 = Function(V1)
		        s1 = Function(S1)


	#Ended Picard's Coupling-scheme.

	#Output fields
	output_sol(v0,s0,v1,s1)

        
def output_sol(u_0,p_0,u_1,p_1):
    u_0.rename("v0", "velocity") ; u0file << u_0
    p_0.rename("p0", "pressure") ; p0file << p_0
    u_1.rename("v1", "velocity") ; u1file << u_1
    p_1.rename("p1", "pressure") ; p1file << p_1

if __name__ == '__main__':
    solve_move()
