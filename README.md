# Total least squares phase retrieval

Repository for 'Total least squares phase retrieval'

We address the phase retrieval problem with errors in the sensing vectors. 
A number of recent methods for phase retrieval are based on least squares (LS) formulations which
assume errors in the quadratic measurements. 
We extend this approach to handle errors in the sensing vectors by adopting the total least squares
(TLS) framework familiar from linear inverse problems with operator errors.

## Code
Run ```tls_ls_comparison.py``` to compare TLS and LS.

In ```tls_ls_comparison.py``` you can specify the signal to recover, the number of measurements, the
amount of sensing vector and measurement error, the measurement model and other algorithm parameters.
