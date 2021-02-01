import numpy as np
from scipy.linalg import dft

def c_conj(A):
    # Conjugate transpose
    
    return np.conj(A.T)

def c_distance(x1, x2):
    # Distance between complex signals
    
    x1 = x1.reshape([-1,1])
    x2 = x2.reshape([-1,1])
    return np.linalg.norm(x1 - np.exp(-1j * np.angle(c_conj(x1)@x2))*x2)

def create_error(truth, perturbation, target_SNR):
    # Scale perturbation so that perturbation is of specified SNR
    
    original_shape = perturbation.shape
    
    truth = truth.flatten()
    perturbation = perturbation.flatten()
    truth_norm = np.linalg.norm(truth)
    perturbation_norm = np.linalg.norm(perturbation)
    
    k = 1 / ((perturbation_norm / truth_norm)*(10**(target_SNR/20)))
    perturbation = k*perturbation
    
    return perturbation.reshape(original_shape)

def octanary_pattern(N, mask_type):
    # Generate N-dimensional octanary pattern
    
    pattern1 = np.array(np.random.choice([1, -1, -1j, 1j], size=N), 
                        dtype=mask_type)
    pattern2 = np.array(np.random.choice([np.sqrt(2)/2, np.sqrt(3)], size=N, 
                                         p=[0.8, 0.2]), dtype=mask_type)
    pattern = pattern1*pattern2
    
    return pattern

def make_CDP_operator(N, M_over_N, matrix_type, mask_type):
    # Makes CDP operator
    
    F = np.array(dft(N), dtype='complex64')
    A = np.array(np.zeros([M_over_N*N, N]), dtype=F.dtype)
    for i in range(M_over_N):
        
        d = octanary_pattern(N, matrix_type)
        
        A[i*N:(i+1)*N] = F * d
        
    return A
