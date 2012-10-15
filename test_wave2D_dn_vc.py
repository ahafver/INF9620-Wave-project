import wave2D_dn_vc
from scitools.std import *
import nose.tools as nt
from math import sqrt, cos, sin
from numpy import max, min, asarray
from distutils import *

def test_constant_solution():
    
    print 'Testing constant solution:'
    
    Lx = 10.0; Ly = 5.0
    
    b = 0.0
    const = 1.2
    def I(x, y): return const
    def q(x, y): return 1.1
    def V(x, y): return 0.0
    def f(x, y, t): return 0.0;
    def check_if_constant(u, xv, yv, t, n):
        nt.assert_almost_equal(const, max(u[1:-1,1:-1]), places=15, msg='diff=%s' % diff)
        nt.assert_almost_equal(const, min(u[1:-1,1:-1]), places=15, msg='diff=%s' % diff)
    
    Nx = 20; Ny = 20; T = 20; dt = 0.1

    wave2D_dn_vc.solver(I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T,
        user_action=check_if_constant, version='scalar', dt_safety_factor = 1)
    wave2D_dn_vc.solver(I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T,
        user_action=check_if_constant, version='vectorized', dt_safety_factor = 1)
    wave2D_dn_vc.solver(I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T,
        user_action=check_if_constant, version='weave_inline', dt_safety_factor = 1)
    wave2D_dn_vc.solver(I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T,
        user_action=check_if_constant, version='weave_inline_omp', dt_safety_factor = 1)
  
    print '\tPassed test'

def test_plug_pulse():
    print 'Testing plug pulse:'

    Lx = 1.0; Ly = 1.0
    b = 0.0
    c = 0.5;
    
    # def I(x,y):
    # return 1-x
        #I = inline("double I(double x, double y){ if (x <= 0.5) return 1.0; else return 0.0; }")

    I = lambda x, y: 1 if x <= 0.5*Lx else 0
    V = lambda x, y: 0
    f = lambda x, y, t: 0
    q = lambda x, y: c**2
    V = lambda x, y: 0
    def store_u(u,xv, yv, t, n):
    #    wave2D_dn_vc.plot_u(u, xv, yv, t, n)
        all_u.append(u.copy())
        return all_u
    
    Nx = 40; Ny = 5; T = 4.0;
    dt = 0.0
    
    all_u = []
    wave2D_dn_vc.solver(I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T,
        user_action=store_u, version='scalar', dt_safety_factor = 1)

    u_first = all_u[0]
    u_last = all_u[-1]
    
    for i in range(1,Nx+1):
        for j in range(1,Ny+1):
            nt.assert_almost_equal(u_first[i,j], u_last[i,j], places=5, msg='diff=%s' % diff)

    all_u = []
    wave2D_dn_vc.solver(I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T,
        user_action=store_u, version='vectorized', dt_safety_factor = 1)
    u_first = all_u[0]
    u_last = all_u[-1]
    
    for i in range(1,Nx+1):
        for j in range(1,Ny+1):
            nt.assert_almost_equal(u_first[i,j], u_last[i,j], places=5, msg='diff=%s' % diff)
    
    all_u = []
    wave2D_dn_vc.solver(I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T,
        user_action=store_u, version='weave_inline', dt_safety_factor = 1)
    u_first = all_u[0]
    u_last = all_u[-1]
    
    for i in range(1,Nx+1):
        for j in range(1,Ny+1):
            nt.assert_almost_equal(u_first[i,j], u_last[i,j], places=5, msg='diff=%s' % diff)
    

    print '\tPassed test'

def test_standing_wave():
 
    print 'Testing standing wave:'
    L = 1; mx = 2; my = 2

    b=0
    omega = sqrt(mx**2 + my**2)*pi

    exact_solution = lambda x,y,t: cos(mx*x*pi/L)*cos(my*y*pi/L)*cos(omega*t)
    I = lambda x, y: exact_solution(x,y,0)
    V = lambda x, y: 0
    q = lambda x, y: 1
    f = lambda x, y, t: 0
    
    def max_diff_from_exact(u, xv, yv, t, n):
        Nx = u.shape[0]-3; Ny = u.shape[1]-3;
        max_diff = 0;
        for i in range(1,Nx+2):
            for j in range(1,Ny+2):
                diff = abs(u[i,j] - exact_solution(xv[i-1,0],yv[0,j-1],t[n]))
                if diff > max_diff:
                    max_diff = diff
        DIFF.append(diff)
 	#wave2D_dn_vc.plot_u(u, xv, yv, t, n)

    T = 0.5;
    dt = 0.0
    
    Nx = [8, 16, 32, 64];
    ratio = []
    print '\tscalar:'
    print '\tNx \tdx  \tErr  \tErr/dx^2'
    for nx in Nx:
        DIFF = []
        wave2D_dn_vc.solver(I, V, f, q, b, L, L, nx, nx, dt, T,
            user_action= max_diff_from_exact, version='scalar', dt_safety_factor = 0.1)
	print '\t%d \t%0.4f \t%0.4f \t%0.4f' %(nx, L/double(nx), DIFF[-1], DIFF[-1]*(nx/L)**2)
     
    
    Nx = [8, 16, 32, 64];
    ratio = []
    print '\tvectorized:'
    print '\tNx \tdx  \tErr  \tErr/dx^2'
    for nx in Nx:
        DIFF = []
        wave2D_dn_vc.solver(I, V, f, q, b, L, L, nx, nx, dt, T,
            user_action= max_diff_from_exact, version='vectorized', dt_safety_factor = 0.1)
	print '\t%d \t%0.4f \t%0.4f \t%0.4f' %(nx, L/double(nx), DIFF[-1], DIFF[-1]*(nx/L)**2)
   

    Nx = [8, 16, 32, 64];
    ratio = []
    print '\tweave_inline:'
    print '\tNx \tdx  \tErr  \tErr/dx^2' 
    for nx in Nx:
        DIFF = []
        wave2D_dn_vc.solver(I, V, f, q, b, L, L, nx, nx, dt, T,
                            user_action= max_diff_from_exact, version='weave_inline', dt_safety_factor = 0.1)
	print '\t%d \t%0.4f \t%0.4f \t%0.4f' %(nx, L/double(nx), DIFF[-1], DIFF[-1]*(nx/L)**2)
  
    
    Nx = [8, 16, 32, 64];
    print '\tweave_inline_omp:'
    print '\tNx \tdx  \tErr  \tErr/dx^2'
    for nx in Nx:
        DIFF = []
        wave2D_dn_vc.solver(I, V, f, q, b, L, L, nx, nx, dt, T,
                            user_action= max_diff_from_exact, version='weave_inline_omp', dt_safety_factor = 0.1)
	print '\t%d \t%0.4f \t%0.4f \t%0.4f' %(nx, L/double(nx), DIFF[-1], DIFF[-1]*(nx/L)**2)
    

def test_manufactured():
 
    print 'Testing manufactured solution:'
    L = pi
    b = 0.1
    
    exact_solution = lambda x,y,t: cos(x)*cos(y)*exp(-b*t)
    I = lambda x, y: exact_solution(x,y,0)
    V = lambda x, y: -b*exact_solution(x,y,0)
    q = lambda x, y: sin(x)*sin(y)
    f = lambda x, y, t: sin(2*x)*sin(2*y)*exp(-b*t)
    
    def max_diff_from_exact(u, xv, yv, t, n):
        Nx = u.shape[0]-2; Ny = u.shape[1]-2;
        max_diff = 0;
        for i in range(1,Nx+1):
            for j in range(1,Ny+1):
                diff = abs(u[i,j] - exact_solution(xv[i-1,0],yv[0,j-1],t[n]))
                if diff > max_diff:
                    max_diff = diff
        DIFF.append(diff)
 	#wave2D_dn_vc.plot_u(u, xv, yv, t, n)
    
    T = 0.5;
    dt = 0.0
    
    Nx = [8, 16, 32, 64];
    print '\tscalar:'
    print '\tNx \tdx  \tErr  \tErr/dx^2'
    for nx in Nx:
        DIFF = []
        wave2D_dn_vc.solver(I, V, f, q, b, L, L, nx, nx, dt, T,
            user_action= max_diff_from_exact, version='scalar', dt_safety_factor = 0.1)
	print '\t%d \t%0.4f \t%0.6f \t%0.4f' %(nx, L/double(nx), DIFF[-1], DIFF[-1]*(nx/L)**2)
    
    Nx = [8, 16, 32, 64];
    Nx = [8, 16, 32, 64];
    print '\tvectorized:'
    print '\tNx \tdx  \tErr  \tErr/dx^2'
    for nx in Nx:
        DIFF = []
        wave2D_dn_vc.solver(I, V, f, q, b, L, L, nx, nx, dt, T,
            user_action= max_diff_from_exact, version='vectorized', dt_safety_factor = 0.1)
	print '\t%d \t%0.4f \t%0.4f \t%0.4f' %(nx, L/double(nx), DIFF[-1], DIFF[-1]*(nx/L)**2)    

    Nx = [8, 16, 32, 64];
    print '\tweave_inline:'
    print '\tNx \tdx  \tErr  \tErr/dx^2'
    for nx in Nx:
        DIFF = []
        wave2D_dn_vc.solver(I, V, f, q, b, L, L, nx, nx, dt, T,
                            user_action= max_diff_from_exact, version='weave_inline', dt_safety_factor = 0.1)
	print '\t%d \t%0.4f \t%0.4f \t%0.4f' %(nx, L/double(nx), DIFF[-1], DIFF[-1]*(nx/L)**2)
    
    Nx = [8, 16, 32, 64];
    print '\tweave_inline_omp:'
    print '\tNx \tdx  \tErr  \tErr/dx^2'
    for nx in Nx:
        DIFF = []
        wave2D_dn_vc.solver(I, V, f, q, b, L, L, nx, nx, dt, T,
                            user_action= max_diff_from_exact, version='weave_inline_omp', dt_safety_factor = 0.1)
	print '\t%d \t%0.4f \t%0.4f \t%0.4f' %(nx, L/double(nx), DIFF[-1], DIFF[-1]*(nx/L)**2)


def test_efficiency():

    print 'Testing efficiency:'

    I = lambda x,y: 0 if abs(x-Lx/2.0) > 0.1 else 1
    q = lambda x, y: c**2
    V = lambda x, y: 0
    f = lambda x, y, t: 0 	
    Lx = 10;  Ly = 10
    c = 1.5
    T = 100
    b = 0.05
    
    versions = ['scalar','vectorized', 'weave_inline', 'weave_inline_omp']
    print ' '*15, ''.join(['%-13s' % v for v in versions])
    for Nxy in 15, 30, 45, 60, 75:
        cpu = {}
        for version in versions:
            dt, cpu_ = wave2D_dn_vc.solver(I, V, f, q, b, Lx, Ly, Nxy, Nxy, 0, T, user_action=None,
                              version=version)
            cpu[version] = cpu_
        cpu_min = min(list(cpu.values()))
        if cpu_min < 1E-6:
            print 'Ignored %dx%d grid (too small execution time)' \
                % (Nxy, Nxy)
        else:
            cpu = {version: cpu[version]/cpu_min for version in cpu}
            print '%-15s' % '%dx%d' % (Nxy, Nxy),
            print ''.join(['%13.1f' % cpu[version] for version in versions])



test_efficiency()
test_constant_solution()
test_plug_pulse()
test_manufactured()
test_standing_wave()

