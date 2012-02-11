cimport cython
cimport numpy as np
from libc.stdlib cimport malloc, free
from cextratrees cimport ET_problem, ET_problem_destroy


cdef class Problem:

    cdef ET_problem *_prob

    def __dealloc__(self):
        if self._prob:
            ET_problem_destroy(self._prob)
            free(self._prob)


cdef Problem problem_factory(ET_problem *prob):
    cdef Problem instance = Problem.__new__(Problem)
    instance._prob = prob
    return instance


@cython.boundscheck(False)
@cython.wraparound(False)
def convert_to_problem(np.ndarray[np.float64_t, ndim=2] X,
                       np.ndarray[np.float64_t, ndim=1] y):

    if y.shape[0] != X.shape[0]:
        raise ValueError('y.shape[0] != X.shape[0]')

    cdef size_t n_features, n_samples
    cdef int i, j

    cdef ET_problem *cprob = <ET_problem *> malloc(sizeof(ET_problem))
    
    n_features = X.shape[1]
    n_samples  = X.shape[0]

    cprob.n_features = n_features
    cprob.n_samples = n_samples

    cprob.vectors = <float *> malloc(sizeof(float) * n_features * n_samples)
    cprob.labels = <double *> malloc(sizeof(double) * n_samples)

    for j in xrange(n_features):
        for i in xrange(n_samples):
            cprob.vectors[j * n_samples + i] = X[i, j]

    for i in xrange(n_samples):
        cprob.labels[i] = y[i]

    return problem_factory(cprob)
