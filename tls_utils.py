import numpy as np
from utils import c_conj

def solve_cubic_batch(c3, c2, c1, c0):
    # Assumes cubics in form c3 * z^3 + c2 * z^2 + c1 * z + c0
    
    M = c0.shape[0]
    
    roots = np.zeros([M,3]).astype('complex64')
    
    c1 = c1.flatten()
    c0 = c0.flatten()
    
    delta_0 = c2**2 - 3*c3*c1
    delta_1 = 2*(c2**3) - 9*c3*c2*c1 + 27*(c3**2)*c0
    
    C = (0.5*(delta_1 + (delta_1**2 - 4*(delta_0**3) + 0j)**0.5))**(1/3.0)
        
    mask = np.abs(C) == 0
    C[mask] = (0.5*(delta_1[mask] - (delta_1[mask]**2 - 4*(delta_0[mask]**3) + 
                                     0j)**0.5))**(1/3.0)

    rotation = 0.5*(-1 + np.sqrt(-3+0j))
    for i in range(3):
        roots[:,i] = -(1/(3*c3)) * (c2 + (rotation**i)*C + 
                                    (delta_0/((rotation**i)*C)))
    
    return roots

def solve_cubic_pr(c3, c1, c0):
    # Returns roots of special cubic in paper
    
    r_0 = solve_cubic_batch(c3, 0, c1, np.abs(c0))
    r_0 = np.real(r_0)
    phi_0 = np.angle(c0)
    
    r_pi = solve_cubic_batch(c3, 0, c1, -np.abs(c0))
    r_pi = np.real(r_pi)
    phi_pi = np.angle(c0) - np.pi
    
    roots = np.hstack((r_0*np.exp(1j*phi_0), r_pi*np.exp(1j*phi_pi)))
    
    return roots

def tls_update_a(y, A, x, norm_estimate, lam_a, lam_y):
    # Update all sensing vectors
    
    M = A.shape[0]
    
    x_unit = x / np.linalg.norm(x)
    
    c3 = 2*lam_y*(np.linalg.norm(x)**2)
    c1 = 1*lam_a - 2*lam_y*y*(np.linalg.norm(x)**2)
    c0 = lam_a*(- c_conj(x) @ c_conj(A)).T
    
    roots = solve_cubic_pr(c3, c1, c0)
    roots = np.transpose(roots)
    
    x_direction = (roots/np.linalg.norm(x)).reshape([6, 1, M]) * x_unit
    
    x_perp_direction =  c_conj(A) - (c_conj(x_unit) @ (c_conj(A)))*x_unit
           
    a_all = np.conj(np.transpose(x_direction + x_perp_direction, (0,2,1)))
    
    # Calculate objective function value
    function_vals = lam_a*(np.linalg.norm(a_all - A, axis=2)**2) + lam_y*(
        (y - np.abs(a_all @ x)**2)**2).reshape([-1, M])
    
    # For each measurement choose the updated sensing vector that gives the 
    # lowest objective function value 
    mins = np.argmin(function_vals, axis=0)
    chosen_roots = np.array(np.zeros(M), dtype=A.dtype)
    for i in range(6):
        args = np.argwhere(mins==i)
        chosen_roots[args] = roots[i, args]
    
    chosen_a = (chosen_roots/np.linalg.norm(x)) * x_unit + x_perp_direction
    
    return c_conj(chosen_a)
