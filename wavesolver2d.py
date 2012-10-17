"""
Finite difference solver the 2D wave equation with damping, 
source term and variable velocity on a domain [0,Lx]x[0,Ly]:

Differential equation:
    u_tt + b*u_t= [q(x,y)*u_x]_x + [q(x,y)*u_y]_y + f(x,t).
Boundary condition:
    du/dn= 0
Initial conditions: 
    u(x,y,0) = I(x,y) and u_t(x,y,0) = V(x,y).

Nx and Ny are the total number of mesh cells in the x and y
directions. The mesh points are numbered as (0,0), (1,0), (2,0),
..., (Nx,0), (0,1), (1,1), ..., (Nx, Ny).

Ghost points are used to implement the reflective boundary condition, 
i.e. the mesh is of size (Nx+2)*(Ny+2))

dt is the time step. If dt<=0, an optimal time step is used.
T is the stop time for the simulation.

I, V, f, q are functions: I(x,y), V(x,y), f(x,y,t) q(x,y,t).

b is the damping coefficient.
    
user_action: function of (u, x, y, t, n) called at each time
level (x and y are one-dimensional coordinate vectors).
This function allows the calling code to plot the solution,
compute errors, etc.
"""

import numpy
from scitools.std import *
from scipy import weave
from scipy.weave import converters
import sys, time

class Grid:
    
    """A simple grid class that stores the details of the computational grid,
    including function values in arrays."""
    
    def __init__(self, Nx=10, Ny=10, Lx=0.0, Ly=1.0):
        self.Lx, self.Ly = Lx, Ly
        self.dx = Lx/float(Nx)
        self.dy = Ly/float(Ny)
	self.x = numpy.linspace(self.dx/2, Lx-self.dx/2, Nx)
	self.y = numpy.linspace(self.dy/2, Ly-self.dy/2, Ny)
	x = numpy.linspace(self.dx/2, Lx-self.dx/2, Nx)
	y = numpy.linspace(self.dy/2, Ly-self.dy/2, Ny)
	#self.xv = self.x[:,newaxis]		Does not work anymore for some strange reason...
        #self.yv = self.y[newaxis,:]
        self.u = numpy.zeros((Nx+2, Ny+2), 'd')        
	self.u_1 = numpy.zeros((Nx+2, Ny+2), 'd')        
	self.u_2 = numpy.zeros((Nx+2, Ny+2), 'd')     
	self.qpx = numpy.zeros((Nx, Ny), 'd')
	self.qmx = numpy.zeros((Nx, Ny), 'd')
	self.qpy = numpy.zeros((Nx, Ny), 'd')
	self.qmy = numpy.zeros((Nx, Ny), 'd')
    	self.f_a = numpy.zeros((Nx, Ny), 'd')

    def stabilityLimit(self, max_c):
	""" Function to determine the stability limit for the grid, i.e. max dt.""" 
        self.stability_limit = min(self.dx,self.dy)/max_c

    def Initialize(self,I = None, V = None, q = 'constant', q_type = 'arithmetic'):
	""" Function to set initial conditions and store the (static) q values."""   
	Nx, Ny = self.qpx.shape
        if (q == 'constant' or q == 1):	
	    max_q = 1.0
        else:
	    max_q = 0.0
	self.xv = numpy.ones((Nx, Ny), 'd')     
	self.yv = numpy.ones((Nx, Ny), 'd')     
	q_a = numpy.ones((Nx+2, Ny+2), 'd')     
	self.V_a = numpy.zeros((Nx, Ny), 'd')     
	# Set q values in cells	
	for i in range(1,Nx+1):        
	    for j in range(1,Ny+1):
		self.xv[i-1,j-1] = self.x[i-1] 
		self.yv[i-1,j-1] = self.y[j-1] 
		if not (I == None or I == 0): self.u_1[i,j] = I(self.x[i-1], self.y[j-1]) 		
            	if not (V == None or V == 0): self.V_a[i-1,j-1] = V(self.x[i-1], self.y[j-1])
		if not (q == 'constant' or q == 1) and not q_type == 'evaluate':
			q_a[i,j] = q(self.x[i-1], self.y[j-1])
            		if q_a[i,j] > max_q: max_q = q_a[i,j]
	
	# Determine stability limit	
	self.stabilityLimit(numpy.sqrt(float(max_q)))

	# Set values in auxiliary q arrays
	if  q_type == 'evaluate':
	    for i in range(0, Nx):
                for j in range(0, Ny):
	    	    self.qpx[i,j] = q(self.x[i] + self.dx/2, self.y[j])
            	    self.qmx[i,j] = q(self.x[i] - self.dx/2, self.y[j])
            	    self.qpy[i,j] = q(self.x[i], self.y[j] + self.dy/2)
            	    self.qmy[i,j] = q(self.x[i], self.y[j] - self.dy/2)        
	else:
	    # Set values in ghost cells
            for i in range(0, Nx+2):
                q_a[i,0] =  q_a[i,1]
	        q_a[i,Ny+1] =  q_a[i,Ny]
            for j in range(0, Ny+2):
	        q_a[0,j] =  q_a[1,j]
                q_a[Nx+1,j] =  q_a[Nx,j]
	    if  q_type == 'harmonic':
                for i in range(0, Nx):
                    for j in range(0, Ny):
      	                self.qpx[i,j] = 2*q_a[i+1,j]*q_a[i,j]/(q_a[i+1,j] + q_a[i,j])
                        self.qmx[i,j] = 2*q_a[i-1,j]*q_a[i,j]/(q_a[i-1,j] + q_a[i,j])
                        self.qpy[i,j] = 2*q_a[i,j+1]*q_a[i,j]/(q_a[i,j+1] + q_a[i,j])
                        self.qmy[i,j] = 2*q_a[i,j-1]*q_a[i,j]/(q_a[i,j-1] + q_a[i,j])
	    else:
                for i in range(0, Nx):
                    for j in range(0, Ny):
            	        self.qpx[i,j] = 0.5*(q_a[i+1,j] + q_a[i,j])
            	        self.qmx[i,j] = 0.5*(q_a[i-1,j] + q_a[i,j])
            	        self.qpy[i,j] = 0.5*(q_a[i,j+1] + q_a[i,j])
            	        self.qmy[i,j] = 0.5*(q_a[i,j-1] + q_a[i,j])
	
	# Set u_1 values in ghost cells
	for i in range(0, Nx+2):
            self.u_1[i,0] = self.u_1[i,1]
	    self.u_1[i,Ny+1] = self.u_1[i,Ny]
        for j in range(0, Ny+2):
	    self.u_1[0,j] = self.u_1[1,j]
            self.u_1[Nx+1,j] = self.u_1[Nx,j]

class WaveSolver:
    
    """A 2D wave solver that can use different schemes to
    solve the problem."""
    
    def __init__(self, grid, f = None, b = 0.0, T = 1.0, dt = 0.0, dt_safety_factor = 1, user_action = None, version ='scalar'):
	self.grid = grid	
	self.f = f
	self.version = version
        self.setAdvance(version)
	self.user_action = user_action
	
        # Set time step
	stability_limit = self.grid.stability_limit
        if dt <= 0:                
            self.dt = dt_safety_factor*stability_limit
        elif dt > stability_limit:
           print 'error: dt=%g exceeds the stability limit %g' % (dt, stability_limit)
    	self.N = int(round(T/float(self.dt))) 
	
	# Define help variables
    	self.Cx2 = (self.dt/self.grid.dx)**2
    	self.Cy2 = (self.dt/self.grid.dy)**2
    	self.dt2 = self.dt**2
    	self.B1 = 1.0/(1.0 + 0.5*b*self.dt)
    	self.B2 = 1.0 - 0.5*b*self.dt

	#if not self.version == 'scalar':
	#    Nx, Ny = self.grid.qpx.shape
	#   self.f_a = numpy.zeros((Nx, Ny), 'd') 

    def advanceFirstStep(self):
	Nx, Ny = self.grid.qpx.shape
	u = self.grid.u
	u_1 = self.grid.u_1
	u_2 = self.grid.u_2
	V_a = self.grid.V_a
	qpx = self.grid.qpx
	qmx = self.grid.qmx
	qpy = self.grid.qpy
	qmy = self.grid.qmy
	x = self.grid.x
	y = self.grid.y
	f_a = self.grid.f_a

	if self.version == 'scalar':
            for i in range(1, Nx+1):
                for j in range(1, Ny+1):
                    u[i,j] = u_1[i,j] + self.dt*self.B2*V_a[i-1,j-1] + \
                        0.5*self.Cx2*(qpx[i-1,j-1]*(u_1[i+1,j] - u_1[i,j]) - qmx[i-1,j-1]*(u_1[i,j] - u_1[i-1,j])) + \
                        0.5*self.Cy2*(qpy[i-1,j-1]*(u_1[i,j+1] - u_1[i,j]) - qmy[i-1,j-1]*(u_1[i,j] - u_1[i,j-1])) + \
                        0.5*self.dt2*self.f(x[i-1], y[j-1], 0)

        else:  # use vectorized version
            self.grid.f_a[:,:] = self.f(self.grid.xv , self.grid.yv ,0)
            u[1:-1,1:-1] = u_1[1:-1,1:-1] + self.dt*self.B2*V_a +\
            	0.5*self.Cx2*(qpx[:,:]*(u_1[2:,1:-1] - u_1[1:-1,1:-1]) - qmx[:,:]*(u_1[1:-1,1:-1] - u_1[:-2,1:-1])) +\
	        0.5*self.Cy2*(qpy[:,:]*(u_1[1:-1,2:] - u_1[1:-1,1:-1]) - qmy[:,:]*(u_1[1:-1,1:-1] - u_1[1:-1,:-2])) +\
 		0.5*self.dt2*f_a[:,:]

    def setAdvance(self, version='scalar'):        
        """Selects the version to be used to advance each timestep ['scalar', 'vectorized', 'blitz',
        'weave_inline', 'weave_fastinline']."""        
        if version == 'scalar':
            self.advance = self.scalarAdvance
        elif version == 'vectorized':
            self.advance = self.vectorizedAdvance
        elif version == 'blitz':
            self.advance = self.blitzAdvance
        elif version == 'weave_inline':
            self.advance = self.weaveInlineAdvance
        elif version.lower() == 'weave_fastinline':
            self.advance = self.fastWeaveInlineAdvance
        elif version.lower() == 'weave_fastinline_omp':
            self.advance = self.fastWeaveInlineOmpAdvance
        else:
            self.advance = self.numericTimeStep       

    def scalarAdvance(self, dt = 0.0, user_action = None):
        """Takes a time step using normal Python loops."""
        Nx, Ny = self.grid.qpx.shape
	u = self.grid.u
	u_1 = self.grid.u_1
	u_2 = self.grid.u_2
	V_a = self.grid.V_a
	qpx = self.grid.qpx
	qmx = self.grid.qmx
	qpy = self.grid.qpy
	qmy = self.grid.qmy
	x = self.grid.x
	y = self.grid.y

    	# Update ghost cells
    	for i in range(1, Nx+1): u_1[i,0] = u_1[i,1]
    	for i in range(1, Nx+1): u_1[i,Ny+1] = u_1[i,Ny]
    	for j in range(1, Ny+1): u_1[0,j] = u_1[1,j]
    	for j in range(01, Ny+1): u_1[Nx+1,j] = u_1[Nx,j] 
      
    	# Update interior cells
    	for i in range(1, Nx+1):
            for j in range(1, Ny+1):
                u[i,j] = self.B1*(2*u_1[i,j] - self.B2*u_2[i,j] + \
                    self.Cx2*(qpx[i-1,j-1]*(u_1[i+1,j] - u_1[i,j]) - qmx[i-1,j-1]*(u_1[i,j] - u_1[i-1,j])) + \
                    self.Cy2*(qpy[i-1,j-1]*(u_1[i,j+1] - u_1[i,j]) - qmy[i-1,j-1]*(u_1[i,j] - u_1[i,j-1])) + \
                    self.dt2*self.f(x[i-1], y[j-1], self.t))  
       		
    def vectorizedAdvance(self, dt = 0.0, user_action = None):
        """Takes a time step using NumPy arrays and vector operations."""
	u = self.grid.u
	u_1 = self.grid.u_1
	u_2 = self.grid.u_2
	V_a = self.grid.V_a
	qpx = self.grid.qpx
	qmx = self.grid.qmx
	qpy = self.grid.qpy
	qmy = self.grid.qmy
	f_a = self.grid.f_a
	f_a[:,:] = self.f(self.grid.xv, self.grid.yv, self.t)

	# Boundary condition du/dn=0
    	u_1[:,0]   = u_1[:,1]
    	u_1[:,-1] = u_1[:,-2]
    	u_1[0,:]   = u_1[1,:]
    	u_1[-1,:] = u_1[-2,:]
 
    	# Update interior cells
    	u[1:-1,1:-1] = self.B1*(2*u_1[1:-1,1:-1] - self.B2*u_2[1:-1,1:-1] + \
           self.Cx2*(qpx*(u_1[2:,1:-1] - u_1[1:-1,1:-1]) - qmx*(u_1[1:-1,1:-1] - u_1[:-2,1:-1])) +\
           self.Cy2*(qpy*(u_1[1:-1,2:] - u_1[1:-1,1:-1]) - qmy*(u_1[1:-1,1:-1] - u_1[1:-1,:-2])) +\
           self.dt2*f_a[:,:])  

    def blitzAdvance(self, dt=0.0, user_action = None):        
 	"""Takes a time step using a numeric expression that is converted to 
        Blitz using weave."""        
	u = self.grid.u
	u_1 = self.grid.u_1
	u_2 = self.grid.u_2
	V_a = self.grid.V_a
	qpx = self.grid.qpx
	qmx = self.grid.qmx
	qpy = self.grid.qpy
	qmy = self.grid.qmy        
	f_a = self.grid.f_a
	f_a[:,:] = self.f(self.grid.xv, self.grid.yv, self.t)
	
	# Define help variables
    	Cx2 = float(self.Cx2)
    	Cy2 = float(self.Cy2)
    	dt2 = float(self.dt2)
    	B1 = float(self.B1)
    	B2 = float(self.B2)
	
	# Boundary condition du/dn=0
    	u_1[:,0]   = u_1[:,1]
    	u_1[:,-1] = u_1[:,-2]
    	u_1[0,:]   = u_1[1,:]
    	u_1[-1,:] = u_1[-2,:]
	
        # The actual iteration
	expr = "u[1:-1,1:-1] = B1*(2*u_1[1:-1,1:-1] - B2*u_2[1:-1,1:-1] + Cx2*(qpx*(u_1[2:,1:-1] - u_1[1:-1,1:-1]) - qmx*(u_1[1:-1,1:-1] - u_1[:-2,1:-1])) + Cy2*(qpy*(u_1[1:-1,2:] - u_1[1:-1,1:-1]) - qmy*(u_1[1:-1,1:-1] - u_1[1:-1,:-2])) + dt2*f_a)"
        weave.blitz(expr, check_size=0)
	

    def weaveInlineAdvance(self, dt=0.0, user_action = None):
    	"""Takes a time step using C code inlined using weave and using Blitz arrays."""
	Nx, Ny = self.grid.qpx.shape
	u  = self.grid.u
	u_1  = self.grid.u_1
	u_2  = self.grid.u_2
	V_a  = self.grid.V_a
	qpx  = self.grid.qpx
	qmx  = self.grid.qmx
	qpy  = self.grid.qpy
	qmy  = self.grid.qmy  
	f_a  = self.grid.f_a
	f_a[:,:] = self.f(self.grid.xv, self.grid.yv, self.t)

	# Define help variables
    	Cx2 = float(self.Cx2)
    	Cy2 = float(self.Cy2)
    	dt2 = float(self.dt2)
    	B1 = float(self.B1)
    	B2 = float(self.B2)
	
	code = """
	int i, j;	
	for (i = 1; i < Nx+1; i++) {
		u_1(i,0) = u_1(i,1); 
		u_1(i,Ny+1) = u_1(i,Ny);
	}
        for (j = 1; j < Ny+1; j++) {
		u_1(0,j) = u_1(1,j);
		u_1(Nx+1,j) = u_1(Nx,j);
	}
	for (i = 1; i < Nx+1; i++) {
            for (j = 1; j < Ny+1; j++) {
		 u(i,j) = B1*(2*u_1(i,j) - B2*u_2(i,j) +
		    Cx2*(qpx(i-1,j-1)*(u_1(i+1,j) - u_1(i,j)) - qmx(i-1,j-1)*(u_1(i,j) - u_1(i-1,j))) +
                    Cy2*(qpy(i-1,j-1)*(u_1(i,j+1) - u_1(i,j)) - qmy(i-1,j-1)*(u_1(i,j) - u_1(i,j-1))) + 
                    dt2*f_a(i-1, j-1));
	    }
        }
	"""
        weave.inline(code, ['u','u_1','u_2','f_a','qpx', 'qmx', 'qpy', 'qmy','B1','B2','Cx2','Cy2','dt2','Nx','Ny'], extra_compile_args =['-O3'], type_converters=weave.converters.blitz)

    def fastWeaveInlineAdvance(self, dt=0.0, user_action = None):
    	"""Takes a time step using C code inlined using weave and array pointers"""
	Nx, Ny = self.grid.qpx.shape
	u  = self.grid.u
	u_1  = self.grid.u_1
	u_2  = self.grid.u_2
	V_a  = self.grid.V_a
	qpx  = self.grid.qpx
	qmx  = self.grid.qmx
	qpy  = self.grid.qpy
	qmy  = self.grid.qmy    
	f_a  = self.grid.f_a
	f_a[:,:] = self.f(self.grid.xv, self.grid.yv, self.t)

	# Define help variables
    	Cx2 = float(self.Cx2)
    	Cy2 = float(self.Cy2)
    	dt2 = float(self.dt2)
    	B1 = float(self.B1)
    	B2 = float(self.B2)

	# Boundary condition du/dn=0
    	u_1[:,0]  = u_1[:,1]
    	u_1[:,-1] = u_1[:,-2]
    	u_1[0,:]   = u_1[1,:]
    	u_1[-1,:] = u_1[-2,:]

	code = """
	int i, j;
	int ny = Ny+2;	
        double *uc, *uc_1, *uc_2;
	double *uu_1, *ud_1, *ul_1, *ur_1;
	double *qpxc, *qmxc, *qpyc, *qmyc, *f_ac;

	for (i = 1; i < Nx+1; i++) {
 	    uc = u+i*ny+1;
 	    uc_1 = u_1+i*ny+1;
 	    uc_2 = u_2+i*ny+1;
            ur_1 = u_1+i*ny+2;     ul_1 = u_1+i*ny;
            uu_1 = u_1+(i+1)*ny+1; ud_1 = u_1+(i-1)*ny+1;
            
	    qpxc = qpx+(i-1)*Ny;
            qmxc = qmx+(i-1)*Ny;
            qpyc = qpy+(i-1)*Ny;
            qmyc = qmy+(i-1)*Ny;
            f_ac = f_a+(i-1)*Ny;

            for (j = 1; j < Ny+1; j++) {
		 *uc = B1*(2.0*(*uc_1) - B2*(*uc_2) +
		  Cx2*((*qpxc)*(*ur_1 - *uc_1) - (*qmxc)*(*uc_1 - *ul_1)) +
                  Cy2*((*qpyc)*(*uu_1 - *uc_1) - (*qmyc)*(*uc_1 - *ud_1)) + 
                  dt2*(*f_ac));
                  uc++; uc_1++; uc_2++;
		  ur_1++; ul_1++; ud_1++; uu_1++;
		  qpxc++; qmxc++; qpyc++; qmyc++; f_ac++;
	    }
        }
	"""
        weave.inline(code, ['u','u_1','u_2','f_a','qpx', 'qmx', 'qpy', 'qmy','B1','B2','Cx2','Cy2','dt2','Nx','Ny']) 
	#, extra_compile_args =['-O3'])

    def fastWeaveInlineOmpAdvance(self, dt=0.0, user_action = None):
    	"""Takes a time step using C code inlined using weave and array pointers. 
	Paralellized using OpenMP"""
    	threads = 4	
	Nx, Ny = self.grid.qpx.shape
	u  = self.grid.u
	u_1  = self.grid.u_1
	u_2  = self.grid.u_2
	V_a  = self.grid.V_a
	qpx  = self.grid.qpx
	qmx  = self.grid.qmx
	qpy  = self.grid.qpy
	qmy  = self.grid.qmy    
	f_a  = self.grid.f_a
	f_a[:,:] = self.f(self.grid.xv, self.grid.yv, self.t)

	# Define help variables
    	Cx2 = float(self.Cx2)
    	Cy2 = float(self.Cy2)
    	dt2 = float(self.dt2)
    	B1 = float(self.B1)
    	B2 = float(self.B2)

	# Boundary condition du/dn=0
    	u_1[:,0]  = u_1[:,1]
    	u_1[:,-1] = u_1[:,-2]
    	u_1[0,:]  = u_1[1,:]
    	u_1[-1,:] = u_1[-2,:]

	code = """
	int i, j;
	int ny = Ny+2;	
	double *uc, *uc_1, *uc_2;
	double *uu_1, *ud_1, *ul_1, *ur_1;
	double *qpxc, *qmxc, *qpyc, *qmyc, *f_ac;	
	#pragma omp parallel for num_threads(threads) private(i,j,uc,uc_1,uc_2,uu_1,ud_1,ul_1,ur_1,qpxc,qmxc,qpyc,qmyc,f_ac) schedule(dynamic, Nx/threads) 
	for (i = 1; i < Nx+1; i++) {
 	    uc = u+i*ny+1;
 	    uc_1 = u_1+i*ny+1;
 	    uc_2 = u_2+i*ny+1;
            ur_1 = u_1+i*ny+2;     ul_1 = u_1+i*ny;
            uu_1 = u_1+(i+1)*ny+1; ud_1 = u_1+(i-1)*ny+1;
            
	    qpxc = qpx+(i-1)*Ny;
            qmxc = qmx+(i-1)*Ny;
            qpyc = qpy+(i-1)*Ny;
            qmyc = qmy+(i-1)*Ny;
            f_ac = f_a+(i-1)*Ny;

            for (j = 1; j < Ny+1; j++) {
		 *uc = B1*(2.0*(*uc_1) - B2*(*uc_2) +
		  Cx2*((*qpxc)*(*ur_1 - *uc_1) - (*qmxc)*(*uc_1 - *ul_1)) +
                  Cy2*((*qpyc)*(*uu_1 - *uc_1) - (*qmyc)*(*uc_1 - *ud_1)) + 
                  dt2*(*f_ac));
                  uc++; uc_1++; uc_2++;
		  ur_1++; ul_1++; ud_1++; uu_1++;
		  qpxc++; qmxc++; qpyc++; qmyc++; f_ac++;
	    }
        }
	"""
        weave.inline(code, ['u','u_1','u_2','f_a','qpx', 'qmx', 'qpy', 'qmy','B1','B2','Cx2','Cy2','dt2','Nx','Ny','threads'], extra_compile_args =['-O3 -fopenmp'], extra_link_args=['-lgomp'], headers = ['<omp.h>'])

    def solve(self):
	""" Function to solve problem."""
	# Advance first time step
	self.t = 0.0
	self.advanceFirstStep()
	self.grid.u_2[:,:] = self.grid.u_1
	self.grid.u_1[:,:] = self.grid.u
	self.t = self.t + self.dt
	if not self.user_action == None:
    	    self.user_action(self.grid.u, self.grid.xv, self.grid.yv, self.t)
	# Begin timeloop
    	t0 = time.clock()
	for n in range(1,self.N):
	   self.advance()
	   self.grid.u_2[:,:] = self.grid.u_1
	   self.grid.u_1[:,:] = self.grid.u
	   self.t = self.t + self.dt
	   if not self.user_action == None:
    	       self.user_action(self.grid.u, self.grid.xv, self.grid.yv, self.t)
	return time.clock() - t0, int(self.t/self.dt)

def speed_test(nx=100, ny=100, Lx = 1, Ly = 1, T = 1, dt = 0.0, version ='scalar'):
    """Sample code to demonstrate how to set up the solver. It is used to measure the 
	time spent in the timeloop, for comparison of the different versions."""
    b = 0.1
    omega = 0.05
    def I(x,y): return exp(-x**2 - y**2)
    def V(x,y): return 0.0
    def q(x,y): return 1.1
    def f(x,y,t): return 0.1*cos(omega*t)

    g = Grid(nx, ny, Lx, Ly)
    g.Initialize(I, V, q, q_type = 'arithmetic')
    s = WaveSolver(g, f, b, T, dt, dt_safety_factor = 0.9, user_action = None, version = version)
    return s.solve()

def main(n=100, L = 1, T = 1, dt = 0.0):
    """Example code to demonstrate how to set up the solver."""
    b = 0.1
    omega = 0.5
    def I(x,y): return cos(x*pi/L)*cos(y*pi/L)
    def V(x,y): return 0.0
    def q(x,y): return 0.25
    def f(x,y,t): return 0.0 #return 0.1*cos(omega*t)

    g = Grid(n, n, L, L)
    g.Initialize(I, V, q, q_type = 'arithmetic')
    s = WaveSolver(g, f, b, T, dt, dt_safety_factor = 0.5, user_action = plot_u, version = 'weave_fastinline')
    s.solve()

def plot_u(u, xv, yv, t):
    if t == 0:
        time.sleep(2)
    surfc(xv, yv, u[1:-1,1:-1], title='t=%g' % t, zlim=[-1, 1],
        colorbar=True, colormap=hot(), caxis=[-1,1], shading='flat')
    time.sleep(0)


if __name__ == "__main__":
    main()
