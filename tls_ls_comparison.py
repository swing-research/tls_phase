import numpy as np

from utils import c_conj, c_distance, create_error, make_CDP_operator
from tls_utils import tls_update_a

def norm_estimate(y):
    # Estimate norm of signal from measurements
    
    return np.sqrt(0.5*np.sum(y)/y.shape[0])

def spectral_initialization(y, A, N, iterations):     
    S = np.array(c_conj(A) @ (y*A), dtype=A.dtype)
    
    x0 = np.array(np.random.normal(size=[N, 1]) + 
                  1j*np.random.normal(size=[N, 1]), dtype=A.dtype)
    x0 = x0 / np.linalg.norm(x0)
    
    # Power iterations to calculate leading eigenvector of S
    for i in range(iterations):
        v = S @ x0
        x0 = v / np.linalg.norm(v)
    
    return x0

def x_update_grad(y, A, x):
    # Gradient update for signal
    
    M = A.shape[0]
    grad = c_conj(A) @ ((np.abs(A @ x)**2 - y)*(A @ x) / M) 
    
    return grad

def ls(y, A, x0, lr, n_iter, norm_estimate):
    x = x0.copy()
    
    loss_prev = np.inf
    
    for i in range(n_iter):
        # Update signal
        x -=(lr/(norm_estimate**2))*x_update_grad(y, A, x)
        
        # Evaluate loss
        loss = np.linalg.norm((y - np.abs(A@x)**2))**2 / (
            y.shape[0] * (norm_estimate**4))
        
        loss_diff = np.abs(loss - loss_prev)
        if (loss_diff < 1e-6):
            break
        loss_prev = loss
        
    print ('LS iterations done: ', i+1)
    
    return x

def tls(y, A, x0, initial_lr, n_iter, norm_estimate, lam_a, lam_y):    
    x = x0.copy()
    
    loss_prev = np.inf
    
    lr = initial_lr*(lam_y / lam_a)*(norm_estimate**4)
    
    for i in range(n_iter):
        # Update sensing vectors
        A_updated = tls_update_a(y, A, x, norm_estimate, lam_a, lam_y)
        # Update signal
        x -= (lr/(norm_estimate**2))*x_update_grad(y, A_updated, x)
        
        # Evaluate loss
        data_loss = np.linalg.norm((y - np.abs(A_updated@x)**2))**2
        a_loss = np.linalg.norm(A - A_updated)**2
        loss = (1/ y.shape[0]) * (lam_y*data_loss + (lam_a)*a_loss)
        
        # Stop if loss change is small
        loss_diff = np.abs(loss - loss_prev)
        if (loss_diff < 1e-6):
            break
        loss_prev = loss
        
    print ('TLS iterations done: ', i+1)
    
    return x, A_updated

###############################################################################
if __name__ == "__main__":
    np.random.seed(0)
    
    # LS and TLS maximum iterations and step size
    ls_iter = 5000
    ls_lr = 0.02
    tls_iter = ls_iter
    tls_lr = 0.5
    
    # If False, Gaussian measurement model is used
    # If True, CDP measurement model is used
    use_CDP = False
    
    # Signal dimension
    N = 100
    # Number of measurements
    M = int(16*N)
    
    # Sensing vector SNR
    a_SNR = 10
    # Measurement SNR
    y_SNR = 40
    
    print ('N: ' + str(N) + ', M: ' + str(M) + 
           ', a_SNR: ' + str(a_SNR) + ', y_SNR: ' + str(y_SNR))

    # Create random signal
    x = np.array(np.random.normal(size=[N,1]) + 
                 1j*np.random.normal(size=[N,1]), dtype='complex64')

    # Create Gaussian or CDP sensing vectors
    if (use_CDP):
        print ('Using coded diffraction measurement model')
        A = make_CDP_operator(N, int(M / N), x.dtype, 'octanary')
    else:
        print ('Using Gaussian measurement model') 
        A = np.array(np.random.normal(size=[M, N]) + 
                     1j*np.random.normal(size=[M, N]), dtype=x.dtype)
    # Create sensing vector errors
    E = np.array(np.random.normal(size=A.shape) + 
                 1j*np.random.normal(size=A.shape), dtype=A.dtype)
    
    # Obtain clean measurements
    y = np.abs(A @ x)**2
    # Create measurement errors
    e = np.random.normal(size=y.shape)
    
    # Scale sensing vector and measurement errors according to SNR
    Delta = create_error(A, E, a_SNR)
    eta = create_error(y, e, y_SNR)
    
    # Perturb sensing vectors and measurements
    A_measured = A + Delta
    y_measured = y + eta
    
    # Original clean sensing vectors and measurements are no longer required
    del A, y, E, e, Delta, eta
        
    ###
    ### Compare LS and TLS ###
    ###
    
    # Estimate signal norm and scale unit-norm spectral initialization
    norm_estimate = norm_estimate(y_measured)
    x0 = spectral_initialization(y_measured, A_measured, N, 50)
    x0 = x0 * norm_estimate
    print ('Initialization distance: ' + str(c_distance(x, x0)
                                             /np.linalg.norm(x)) + '\n')
    
    # Use LS
    x_ls = ls(y_measured, A_measured, x0, ls_lr, ls_iter, norm_estimate)
    print ('LS distance: ' + str(c_distance(x, x_ls)/
                                 np.linalg.norm(x)) + '\n')
    
    # Use TLS
    lam_a = 1 / N
    lam_y = 1 / (norm_estimate**4)
    x_tls, A_updated = tls(y_measured, A_measured, x0, tls_lr, tls_iter, 
                       norm_estimate, lam_a, lam_y)
    print ('TLS distance: ', c_distance(x, x_tls)/np.linalg.norm(x))
