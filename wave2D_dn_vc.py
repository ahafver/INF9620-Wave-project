"""
2D wave equation with damping and reflective boundary conditions 
    solved by finite differences::

  dt, cpu_time = solver(I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T,
                        user_action=None, version='scalar',
                        dt_safety_factor=1)

Solve the 2D wave equation u_tt + b*u_t= [q(x,y)*u_x]_x + [q(x,y)*u_y]_y + f(x,t) on [0,Lx]x[0,Ly].
    Boundary condition: du/dn= 0
    Initial conditions: u(x,y,0) = I(x,y) and u_t(x,y,0) = V(x,y).

Nx and Ny are the total number of mesh cells in the x and y
directions. The mesh points are numbered as (0,0), (1,0), (2,0),
..., (Nx,0), (0,1), (1,1), ..., (Nx, Ny).
    (In the implementation we use Ghost points, i.e. the mesh is of
    size (Nx+2)*(Ny+2))

dt is the time step. If dt<=0, an optimal time step is used.
T is the stop time for the simulation.

I, V, f, q are functions: I(x,y), V(x,y), f(x,y,t) q(x,y,t). V and f
can be specified as None or 0, resulting in V=0 and f=0.

b is the damping coefficient.
    
user_action: function of (u, x, y, t, n) called at each time
level (x and y are one-dimensional coordinate vectors).
This function allows the calling code to plot the solution,
compute errors, etc.
"""

import time
from scitools.std import *
from numpy import max, min
import scipy.weave

def solver(I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T, user_action=None, version='scalar', dt_safety_factor=1):
    
    # Check for version
    if version == 'scalar':
        advance = advance_scalar
    elif version == 'vectorized':
        advance = advance_vectorized
    elif version == 'weave_inline':
        advance = advance_weave_inline
    elif version == 'weave_inline_omp':
        advance = advance_weave_inline_omp
   
    
    # Allow f to be None or 0
    if f is None or f == 0:
        f = (lambda x, y, t: 0) if version == 'scalar' else \
            lambda x, y, t: zeros((x.shape[0], y.shape[1]))

    # Define mesh
    x = linspace(0, Lx, Nx+1)  # mesh points in x dir
    y = linspace(0, Ly, Ny+1)  # mesh points in y dir
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    x = x[:-1] + dx/2;
    y = y[:-1] + dy/2;
    xv = x[:,newaxis]
    yv = y[newaxis,:]
                
    # Initialize solution vectors
    order = 'Fortran' if version == 'f77' else 'C'
    u   = zeros((Nx+2,Ny+2), order=order)   # solution array
    u_1 = zeros((Nx+2,Ny+2), order=order)   # solution at t-dt
    u_2 = zeros((Nx+2,Ny+2), order=order)   # solution at t-2*dt
    
    # Calculate and store function values in matrices
    q_a = zeros((Nx+2,Ny+2), order=order)   # for compiled loops
    f_a = zeros((Nx+2,Ny+2), order=order)   # for compiled loops
    V_a = zeros((Nx,Ny), order=order)   # for compiled loops
    max_q = 0

    for i in range(1,Nx+1):
        for j in range(1,Ny+1):
            q_a[i,j] = q(x[i-1], y[j-1])
            if q_a[i,j] > max_q: max_q = q_a[i,j]
            V_a[i-1,j-1] = V(x[i-1], y[j-1])

    for i in range(0, Nx+2): q_a[i,0] = q_a[i,1]
    for i in range(0, Nx+2): q_a[i,Ny+1] = q_a[i,Ny]
    for j in range(0, Ny+2): q_a[0,j] = q_a[1,j]
    for j in range(0, Ny+2): q_a[Nx+1,j] = q_a[Nx,j]
    
                      
    # Check for stability
    stability_limit = min(dx,dy)/sqrt(float(max_q))
            #stability_limit = (1/float(max_q))*(1/sqrt(1/dx**2 + 1/dy**2))
    if dt <= 0:                # max time step?
        dt = dt_safety_factor*stability_limit
    elif dt > stability_limit:
        print 'error: dt=%g exceeds the stability limit %g' % \
        (dt, stability_limit)
    
    N = int(round(T/float(dt)))
    t = linspace(0, N*dt, N+1)    # mesh points in time
    
    # Define help variables
    Cx2 = (dt/dx)**2
    Cy2 = (dt/dy)**2
    dt2 = dt**2
    B1 = 1.0/(1.0 + 0.5*b*dt)
    B2 = 1.0 - 0.5*b*dt
    
    # Set initial condition
    for i in range(1, Nx+1):
        for j in range(1, Ny+1):
            u_1[i,j] = I(x[i-1], y[j-1])
    if version == 'scalar':
        # Set ghost cells
        for i in range(0, Nx+2): u_1[i,0] = u_1[i,1]
        for i in range(0, Nx+2): u_1[i,Ny+1] = u_1[i,Ny]
        for j in range(0, Ny+2): u_1[0,j] = u_1[1,j]
        for j in range(0, Ny+2): u_1[Nx+1,j] = u_1[Nx,j]
    else: # use vectorized version
        # Set ghost cells
        u_1[:,0]   = u_1[:,1]
        u_1[:,Ny+1] = u_1[:,Ny]
        u_1[0,:]   = u_1[1,:]
        u_1[Nx+1,:] = u_1[Nx,:]

    if user_action is not None:
        user_action(u_1, xv, yv, t, 0)

    # Special formula for first time step
    n = 0
    if version == 'scalar':
        for i in range(1, Nx+1):
            for j in range(1, Ny+1):
                
                qpx = 4*q_a[i+1,j]*q_a[i,j]/(q_a[i+1,j] + q_a[i,j])
                qmx = 4*q_a[i-1,j]*q_a[i,j]/(q_a[i-1,j] + q_a[i,j])
                qpy = 4*q_a[i,j+1]*q_a[i,j]/(q_a[i,j+1] + q_a[i,j])
                qmy = 4*q_a[i,j-1]*q_a[i,j]/(q_a[i,j-1] + q_a[i,j])
                #qpx = (q_a[i+1,j] + q_a[i,j])
                #qmx = (q_a[i-1,j] + q_a[i,j])
                #qpy = (q_a[i,j+1] + q_a[i,j])
                #qmy = (q_a[i,j-1] + q_a[i,j])
                u[i,j] = u_1[i,j] + dt*B2*V(x[i-1], y[j-1]) + \
                    0.25*Cx2*(qpx*(u_1[i+1,j] - u_1[i,j]) - qmx*(u_1[i,j] - u_1[i-1,j])) + \
                    0.25*Cy2*(qpy*(u_1[i,j+1] - u_1[i,j]) - qmy*(u_1[i,j] - u_1[i,j-1])) + \
                    0.5*dt2*f(x[i-1], y[j-1], t[n])

    else:  # use vectorized version
        for i in range(1,Nx+1):
            for j in range(1,Ny+1):
                f_a[i,j] = f(x[i-1], y[j-1],0)
        #f_a[1:-1,1:-1] = f(xv, yv, t[n])
	qpx = 4*q_a[2:,1:-1]*q_a[1:-1,1:-1]/(q_a[2:,1:-1] + q_a[1:-1,1:-1])
        qmx = 4*q_a[:-2,1:-1]*q_a[1:-1,1:-1]/(q_a[:-2,1:-1] + q_a[1:-1,1:-1])
        qpy = 4*q_a[1:-1,2:]*q_a[1:-1,1:-1]/(q_a[1:-1,2:] + q_a[1:-1,1:-1])
        qmy = 4*q_a[1:-1,:-2]*q_a[1:-1,1:-1]/(q_a[1:-1,:-2] + q_a[1:-1,1:-1])
        u[1:-1,1:-1] = u_1[1:-1,1:-1] + dt*B2*V_a + \
        0.25*Cx2*(qpx*(u_1[2:,1:-1] - u_1[1:-1,1:-1]) - qmx*(u_1[1:-1,1:-1] - u_1[:-2,1:-1])) +\
        0.25*Cy2*(qpy*(u_1[1:-1,2:] - u_1[1:-1,1:-1]) - qmy*(u_1[1:-1,1:-1] - u_1[1:-1,:-2])) +\
        0.5*dt2*f_a[1:-1,1:-1]

    if user_action is not None:
        user_action(u, xv, yv, t, 1)

    u_2[:,:] = u_1; u_1[:,:] = u

    t0 = time.clock()
    for n in range(1, N):
        if version == 'scalar':
            u = advance(u, u_1, u_2, f, q_a, x, y, t, n, B1, B2, Cx2, Cy2, dt2)
        else:
            for i in range(1,Nx+1):
                for j in range(1,Ny+1):
                    f_a[i,j] = f(x[i-1], y[j-1],t[n])
            #f_a[1:-1,1:-1] = f(xv, yv, t[n])  # precompute, size as u
            u = advance(u, u_1, u_2, f_a, q_a, x, y, B1, B2, Cx2, Cy2, dt2)
        if version == 'f77':
            for a in 'u', 'u_1', 'u_2', 'f_a':
                if not isfortran(eval(a)):
                    print '%s: not Fortran storage!' % a

        if user_action is not None:
            user_action(u, xv, yv, t, n+1)

        u_2[:,:], u_1[:,:] = u_1, u

    t1 = time.clock()
    # dt might be computed in this function so return the value
    return dt, t1 - t0

def advance_scalar(u, u_1, u_2, f, q_a, x, y, t, n, B1, B2, Cx2, Cy2, dt2):
    Nx = u.shape[0]-2;  Ny = u.shape[1]-2
    # Update ghost cells
    for i in range(0, Nx+2): u_1[i,0] = u_1[i,1]
    for i in range(0, Nx+2): u_1[i,Ny+1] = u_1[i,Ny]
    for j in range(0, Ny+2): u_1[0,j] = u_1[1,j]
    for j in range(0, Ny+2): u_1[Nx+1,j] = u_1[Nx,j]        
    # Update interior cells
    for i in range(1, Nx+1):
        for j in range(1, Ny+1):
            qpx = 4*q_a[i+1,j]*q_a[i,j]/(q_a[i+1,j] + q_a[i,j])
            qmx = 4*q_a[i-1,j]*q_a[i,j]/(q_a[i-1,j] + q_a[i,j])
            qpy = 4*q_a[i,j+1]*q_a[i,j]/(q_a[i,j+1] + q_a[i,j])
            qmy = 4*q_a[i,j-1]*q_a[i,j]/(q_a[i,j-1] + q_a[i,j])
            #qpx = (q_a[i+1,j] + q_a[i,j])
            #qmx = (q_a[i-1,j] + q_a[i,j])
            #qpy = (q_a[i,j+1] + q_a[i,j])
            #qmy = (q_a[i,j-1] + q_a[i,j])
            u[i,j] = B1*(2*u_1[i,j] - B2*u_2[i,j] + \
                0.5*Cx2*(qpx*(u_1[i+1,j] - u_1[i,j]) - qmx*(u_1[i,j] - u_1[i-1,j])) + \
                0.5*Cy2*(qpy*(u_1[i,j+1] - u_1[i,j]) - qmy*(u_1[i,j] - u_1[i,j-1])) + \
                dt2*f(x[i-1], y[j-1], t[n]))
    return u

def advance_vectorized(u, u_1, u_2, f_a, q_a, x, y, B1, B2, Cx2, Cy2, dt2):
    # Boundary condition u=0
    Nx = u.shape[0]-2;  Ny = u.shape[1]-2
    u_1[: ,0]   = u_1[:,1]
    u_1[:,Ny+1] = u_1[:,Ny]
    u_1[0 ,:]   = u_1[1, :]
    u_1[Nx+1,:] = u_1[Nx, :]
    # Update interior cells
    qpx = 4*q_a[2:,1:-1]*q_a[1:-1,1:-1]/(q_a[2:,1:-1] + q_a[1:-1,1:-1])
    qmx = 4*q_a[:-2,1:-1]*q_a[1:-1,1:-1]/(q_a[:-2,1:-1] + q_a[1:-1,1:-1])
    qpy = 4*q_a[1:-1,2:]*q_a[1:-1,1:-1]/(q_a[1:-1,2:] + q_a[1:-1,1:-1])
    qmy = 4*q_a[1:-1,:-2]*q_a[1:-1,1:-1]/(q_a[1:-1,:-2] + q_a[1:-1,1:-1])
    u[1:-1,1:-1] = B1*(2*u_1[1:-1,1:-1] - B2*u_2[1:-1,1:-1] + \
        0.5*Cx2*(qpx*(u_1[2:,1:-1] - u_1[1:-1,1:-1]) - qmx*(u_1[1:-1,1:-1] - u_1[:-2,1:-1])) +\
        0.5*Cy2*(qpy*(u_1[1:-1,2:] - u_1[1:-1,1:-1]) - qmy*(u_1[1:-1,1:-1] - u_1[1:-1,:-2])) +\
        dt2*f_a[1:-1,1:-1])
    return u


def advance_weave_inline(u, u_1, u_2, f_a, q_a, x, y, B1, B2, Cx2, Cy2, dt2):
    Nx = u.shape[0]-2
    Ny = u.shape[1]-2

    code = """
	double qpx, qmx, qpy, qmy;
	int i, j;
	for (i = 0; i < Nx+2; i++) {u_1(i,0) = u_1(i,1);}
        for (i = 0; i < Nx+2; i++) {u_1(i,Ny+1) = u_1(i,Ny);}
        for (j = 0; j < Ny+2; j++) {u_1(0,j) = u_1(1,j);}
        for (j = 0; j < Ny+2; j++) {u_1(Nx+1,j) = u_1(Nx,j);}	
	for (i = 1; i < Nx+1; i++) {
            for (j = 1; j < Ny+1; j++) {
		qpx = q_a(i+1,j)*q_a(i,j)/(q_a(i+1,j) + q_a(i,j));
                qmx = q_a(i-1,j)*q_a(i,j)/(q_a(i-1,j) + q_a(i,j));
                qpy = q_a(i,j+1)*q_a(i,j)/(q_a(i,j+1) + q_a(i,j));
                qmy = q_a(i,j-1)*q_a(i,j)/(q_a(i,j-1) + q_a(i,j));         
		u(i,j) = double(B1)*(2*u_1(i,j) - double(B2)*u_2(i,j) +
		    2*double(Cx2)*(qpx*(u_1(i+1,j) - u_1(i,j)) - qmx*(u_1(i,j)   - u_1(i-1,j))) +
                    2*double(Cy2)*(qpy*(u_1(i,j+1) - u_1(i,j)) - qmy*(u_1(i,j) - u_1(i,j-1))) + 
                    double(dt2)*f_a(i, j));
	    }
        }
		
	"""
    scipy.weave.inline(code, ['&u','&u_1','&u_2', '&f_a','&q_a','B1','B2','Cx2','Cy2','dt2','Nx','Ny'], extra_compile_args =['-O3'], type_converters=scipy.weave.converters.blitz)
    return u

def advance_weave_inline_omp(u, u_1, u_2, f_a, q_a, x, y, B1, B2, Cx2, Cy2, dt2):
    Nx = u.shape[0]-2
    Ny = u.shape[1]-2
    threads = 4
    code = """
	double qpx, qmx, qpy, qmy;
	int i, j;
	for (i = 0; i < Nx+2; i++) {u_1(i,0) = u_1(i,1);}
        for (i = 0; i < Nx+2; i++) {u_1(i,Ny+1) = u_1(i,Ny);}
        for (j = 0; j < Ny+2; j++) {u_1(0,j) = u_1(1,j);}
        for (j = 0; j < Ny+2; j++) {u_1(Nx+1,j) = u_1(Nx,j);}

	#pragma omp parallel for num_threads(threads) default(shared) private(i,j) schedule(static) 
	for (i = 1; i < Nx+1; i++) {
            for (j = 1; j < Ny+1; j++) {
		qpx = q_a(i+1,j)*q_a(i,j)/(q_a(i+1,j) + q_a(i,j));
                qmx = q_a(i-1,j)*q_a(i,j)/(q_a(i-1,j) + q_a(i,j));
                qpy = q_a(i,j+1)*q_a(i,j)/(q_a(i,j+1) + q_a(i,j));
                qmy = q_a(i,j-1)*q_a(i,j)/(q_a(i,j-1) + q_a(i,j));         
		u(i,j) = double(B1)*(2*u_1(i,j) - double(B2)*u_2(i,j) +
		    2*double(Cx2)*(qpx*(u_1(i+1,j) - u_1(i,j)) - qmx*(u_1(i,j)   - u_1(i-1,j))) +
                    2*double(Cy2)*(qpy*(u_1(i,j+1) - u_1(i,j)) - qmy*(u_1(i,j) - u_1(i,j-1))) + 
                    double(dt2)*f_a(i, j));
	    }
        }
		
	"""
    scipy.weave.inline(code, ['u','u_1','u_2', 'f_a','q_a','B1','B2','Cx2','Cy2','dt2','Nx','Ny','threads'], extra_compile_args =['-O3 -fopenmp'], extra_link_args=['-lgomp'], headers = ['<omp.h>'], type_converters=scipy.weave.converters.blitz)
    return u

def eval_f_in_c(res,f,x,y,t):
    Nx = u.shape[0]-2
    Ny = u.shape[1]-2
    code = """
	for (i = 1; i < Nx+1; i++) {
            for (j = 1; j < Ny+1; j++) {
		res(i,j) = f(x(i,j), y(i,j), t);
	    }
        }
	"""
    res = scipy.weave.inline(code, ['res','f','x', 'y','t','Nx','Ny'], extra_compile_args =['-O3'], type_converters=scipy.weave.converters.blitz)
    return res


def plot_u(u, xv, yv, t, n):
    if t[n] == 0:
        time.sleep(2)
    surfc(xv, yv, u[1:-1,1:-1], title='t=%g' % t[n], zlim=[-1, 1],
        colorbar=True, colormap=hot(), caxis=[-1,1], shading='flat')
    time.sleep(0)


