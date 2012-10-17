import wavesolver2d
import numpy
import nose.tools as nt
from math import pi, sqrt, cos, sin, exp

versions = ['scalar', 'vectorized',  'blitz', 'weave_inline', 'weave_fastinline', 'weave_fastinline_omp']

def test_constant_solution():
    """ Tests that all versions can reproduce a constant solution."""
    print 'Testing constant solution:'
    Lx = 10.0; Ly = 5.0
    Nx = 20; Ny = 20; T = 1; dt = 0.0
    b = 0.0
    const = 1.2

    def I(x, y): return const
    def q(x, y): return 0.9
    def V(x, y): return 0.0
    def f(x, y, t): return 0.0;
    
    def check_if_constant(u, xv, yv, t):
	diff = numpy.abs(numpy.max(u[1:-1,1:-1]) - const)
        nt.assert_almost_equal(0.0, diff, places=15, msg='diff=%s' % diff)
	diff = numpy.abs(numpy.min(u[1:-1,1:-1]) - const)
        nt.assert_almost_equal(0.0, diff, places=15, msg='diff=%s' % diff)
    
    for version in versions:
        print '\t', version, ": ",	
	g = wavesolver2d.Grid(Nx, Ny, Lx, Ly)    
        g.Initialize(I, V, q, q_type = 'arithmetic')
	s = wavesolver2d.WaveSolver(g, f, b, T, dt, user_action = check_if_constant, version = version)
	s.solve()
    	print 'Passed test'

def test_plug_pulse():
    """ Tests that all versions can propagate a plug_pulse exactly, 
	i.e that it returns to its original state after one period"""
    print 'Testing plug pulse:'
    Lx = 1; Ly = 1
    Nx = 10; Ny = 10; T = 4; dt = 0.0
    b = 0.0
    const = 2
    c = 0.5

    def I(x, y): 
	if x < Lx/2.0: return const
	else: return 0
    def q(x, y): return c**2
    def V(x, y): return 0.0
    def f(x, y, t): return 0.0;
    	
    def return_u(u, xv, yv, t):
	u_returned[:,:] = u
	#print u_returned
              	
    u_first = numpy.zeros((Nx+2, Ny+2), 'd')        
    u_returned = numpy.zeros((Nx+2, Ny+2), 'd')        
            
    for version in versions:
        print '\t', version, ": ",	
	g = wavesolver2d.Grid(Nx, Ny, Lx, Ly)    
        g.Initialize(I, V, q, q_type = 'arithmetic')
	u_first[:,:] = g.u_1 
	s = wavesolver2d.WaveSolver(g, f, b, T, dt, user_action = return_u, version = version)
	s.solve()
	for i in range(1,Nx+1):
            for j in range(1,Ny+1):
                nt.assert_almost_equal(u_first[i,j], u_returned[i,j], places=13, msg='Failed plug wave test')
    	print 'Passed test'

def test_standing_wave():
 
    print 'Testing standing wave:'
    T = 0.5;
    dt = 0.0
    L = 1; mx = 2; my = 1
    b=0
    omega = sqrt(mx**2 + my**2)*pi

    def exact_solution(x,y,t): return cos(mx*x*pi/L)*cos(my*y*pi/L)*cos(omega*t)
    def I(x,y): return exact_solution(x,y,0)
    def V(x,y): return 0
    def q(x,y): return 1
    def f(x,y,t): return 0
    
    def max_diff_from_exact(u, xv, yv, t):
        Nx = u.shape[0]-3; Ny = u.shape[1]-3;
        max_diff = 0;
        for i in range(1,Nx+2):
            for j in range(1,Ny+2):
                diff = abs(u[i,j] - exact_solution(xv[i-1,j-1],yv[i-1,j-1],t))
                if diff > max_diff:
                    max_diff = diff
        DIFF.append(diff)
 	#wave2D_dn_vc.plot_u(u, xv, yv, t, n)
    
    Nx = [8, 16, 32, 64,128];
    for version in versions:
        print '\t', version, ": "
        print '\tNx \tdx  \tErr  \tErr/dx^2'
        for nx in Nx:
            DIFF = []
            g = wavesolver2d.Grid(nx, nx, L, L)    
            g.Initialize(I, V, q, q_type = 'arithmetic')
	    s = wavesolver2d.WaveSolver(g, f, b, T, dt, dt_safety_factor = 0.5, user_action = max_diff_from_exact, version = version)
	    s.solve()
            print '\t%d \t%0.4f \t%0.4f \t%0.4f' %(nx, L/float(nx), DIFF[-1], DIFF[-1]*(nx/L)**2)
     
def test_manufactured():
 
    print 'Testing manufactured solution:'
    L = pi
    b = 0.1
    T = 0.5
    dt = 0.0

    def exact_solution(x,y,t): return cos(x)*cos(y)*exp(-b*t)
    def I(x,y): return exact_solution(x,y,0)
    def V(x,y): return -b*exact_solution(x,y,0)
    def q(x,y): return sin(x)*sin(y)
    def fs(x,y,t):  # Scalar
	return sin(2*x)*sin(2*y)*exp(-b*t)
    def fv(x,y,t):  # Vector
	return numpy.sin(2*x)*numpy.sin(2*y)*exp(-b*t) 
    
    def max_diff_from_exact(u, xv, yv, t):
        Nx = u.shape[0]-2; Ny = u.shape[1]-2;
        max_diff = 0;
        for i in range(1,Nx+1):
            for j in range(1,Ny+1):
                diff = abs(u[i,j] - exact_solution(xv[i-1,0],yv[0,j-1],t))
                if diff > max_diff:
                    max_diff = diff
        DIFF.append(diff)
    
    Nx = [8, 16, 32, 64, 128];
    for version in versions:
        print '\t', version, ": "
	if version == 'scalar':
	    f = fs
	else: f = fv
        print '\tNx \tdx  \tErr  \tErr/dx^2'
        for nx in Nx:
            DIFF = []
            g = wavesolver2d.Grid(nx, nx, L, L)    
            g.Initialize(I, V, q, q_type = 'arithmetic')
	    s = wavesolver2d.WaveSolver(g, f, b, T, dt, dt_safety_factor = 0.5, user_action = max_diff_from_exact, version = version)
	    s.solve()
            print '\t%d \t%0.4f \t%0.4f \t%0.4f' %(nx, L/float(nx), DIFF[-1], DIFF[-1]*(nx/L)**2)
      
def test_speed(nx=100, ny=100, Lx = 1, Ly = 1, T = 1, dt = 0.0, version ='scalar'):
    """Code used to measure the time spent in the timeloop, for comparison of the different versions."""
    b = 0.1
    omega = 0.05
    def I(x,y): return exp(-x**2 - y**2)
    def V(x,y): return 0.0
    def q(x,y): return 1.1
    def f(x,y,t): return 0.1*cos(omega*t)
    print ' '*15, ''.join(['%-13s' % version for version in versions])
    Nx = [25, 50, 100, 150, 200, 300];
    for nx in Nx:
        cpu = {}
        for version in versions:
            g = wavesolver2d.Grid(nx, ny, Lx, Ly)
            g.Initialize(I, V, q, q_type = 'arithmetic')
            s = wavesolver2d.WaveSolver(g, f, b, T, dt, dt_safety_factor = 0.9, user_action = None, version = version)
    	    cpu_, n_iter = s.solve()
	    cpu[version] = cpu_
        cpu_min = min(list(cpu.values()))
        if cpu_min < 1E-6:
            print 'Ignored %dx%d grid (too small execution time)' % (nx, nx)
        else:
	    print '%-15s' % '%dx%d' % (nx, nx),
	    print ''.join(['%13.1f' % cpu[version] for version in versions])
            cpu = {version: cpu[version]/cpu_min for version in cpu}
            print '\t\t',''.join(['%13.1f' % cpu[version] for version in versions])

print 'RUNNING UNIT TESTS:'
test_constant_solution()
test_plug_pulse()
test_standing_wave()
test_manufactured()
test_speed()
