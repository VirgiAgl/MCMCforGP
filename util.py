import numpy as np


def eigen_decomposition(omega, num_eigen):
    vals, vecs = np.linalg.eig(omega)
    diag = np.diag(np.sqrt(np.real(vals)))
    vecs = np.real(vecs)

    X_complete = np.dot(vecs, diag)  # Full X matrix
    vals = vals[0:num_eigen]
    vecs = vecs[:, 0:num_eigen]
    diag = diag[0:num_eigen, 0:num_eigen]
    print('this is diag', diag)
    X_eigen = np.dot(vecs, diag)
    return X_eigen, diag
