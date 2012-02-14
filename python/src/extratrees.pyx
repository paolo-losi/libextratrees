import numpy
cimport cython
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc cimport math
from cextratrees cimport (ET_problem, ET_problem_destroy,
                          ET_forest, ET_forest_destroy, ET_forest_build,
                          ET_forest_predict, ET_params)


cdef class Problem:

    cdef ET_problem *_prob

    def __dealloc__(self):
        if self._prob:
            ET_problem_destroy(self._prob)
            free(self._prob)

    def _train(self, number_of_features_tested=None,
                     number_of_trees=100,
                     regression=False,
                     min_split_size=1,
                     select_features_with_replacement=False):
        cdef ET_params params
        cdef ET_forest *cforest
        cdef ET_problem *cprob = self._prob

        if number_of_features_tested is None:
            if regression:
                number_of_features_tested = cprob.n_features
            else:
                number_of_features_tested = math.ceil(
                                                math.sqrt(cprob.n_features))

        params.number_of_features_tested = number_of_features_tested
        params.number_of_trees = number_of_trees
        params.regression = regression
        params.min_split_size = min_split_size
        params.select_features_with_replacement = \
                                             select_features_with_replacement

        cforest = ET_forest_build(cprob, &params)
        return forest_factory(cforest)


cdef Problem problem_factory(ET_problem *prob):
    cdef Problem instance = Problem.__new__(Problem)
    instance._prob = prob
    return instance


cdef class Forest:

    cdef ET_forest *_forest

    def __dealloc__(self):
        if self._forest:
            ET_forest_destroy(self._forest)
            free(self._forest)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict(self, np.ndarray[np.float64_t, ndim=2] X):
        cdef np.ndarray[np.float64_t, ndim=1] y
        cdef float *vector
        cdef int sample_idx, feature_idx

        y = numpy.empty(shape=(X.shape[0],), dtype=numpy.float64)
        vector = <float *> malloc(sizeof(float) * X.shape[1])
        if not vector:
            raise MemoryError()

        for sample_idx in xrange(X.shape[0]):
            for feature_idx in xrange(X.shape[1]):
                vector[feature_idx] = X[sample_idx, feature_idx]

            y[sample_idx] = ET_forest_predict(self._forest, vector)

        return y


cdef Forest forest_factory(ET_forest *forest):
    cdef Forest instance = Forest.__new__(Forest)
    instance._forest = forest
    return instance


# TODO use cython typed memoryviews to avoid copying
@cython.boundscheck(False)
@cython.wraparound(False)
def convert_to_problem(np.ndarray[np.float64_t, ndim=2] X not None,
                       np.ndarray[np.float64_t, ndim=1] y not None):

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
    if not cprob.labels or not cprob.vectors:
        raise MemoryError()

    for j in xrange(n_features):
        for i in xrange(n_samples):
            cprob.vectors[j * n_samples + i] = X[i, j]

    for i in xrange(n_samples):
        cprob.labels[i] = y[i]

    return problem_factory(cprob)


@cython.boundscheck(False)
@cython.wraparound(False)
def convert_from_problem(Problem prob not None):
    cdef int i, j
    cdef ET_problem *cprob = prob._prob
    cdef np.ndarray[np.float64_t, ndim=2] X
    cdef np.ndarray[np.float64_t, ndim=1] y

    X = numpy.empty(shape=(cprob.n_samples, cprob.n_features), dtype=np.float64)
    y = numpy.empty(shape=(cprob.n_samples,), dtype=np.float64)

    for j in xrange(cprob.n_features):
        for i in xrange(cprob.n_samples):
            X[i, j] = cprob.vectors[j * cprob.n_samples + i]

    for i in xrange(cprob.n_samples):
        y[i] = cprob.labels[i]

    return X, y


def train(X, y, **params):
    return convert_to_problem(X, y)._train(**params) 

