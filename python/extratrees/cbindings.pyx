import numpy
cimport cython
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint32_t
from libc cimport math
from libcpp cimport bool
from cextratrees cimport (ET_problem, ET_problem_destroy,
                          ET_forest, ET_forest_destroy, ET_forest_build,
                          ET_forest_predict, ET_forest_predict_regression,
                          ET_forest_predict_class_majority,
                          ET_forest_predict_probability,
                          ET_forest_neighbors, ET_params,
                          ET_forest_predict_class_bayes,
                          class_probability_vec, class_probability,
                          neighbour_weight, neighbour_weight_vec,
                          double_vec, ET_forest_feature_importance)


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


cdef double _p_simple(ET_forest *f, float *v, uint32_t curtail, bool _):
    return ET_forest_predict(f, v)

cdef double _p_regression(ET_forest *f, float *v, uint32_t curtail, bool _):
    return ET_forest_predict_regression(f, v, curtail)

cdef double _p_class_majority(ET_forest *f, float *v, uint32_t curtail, bool _):
    return ET_forest_predict_class_majority(f, v, curtail)

cdef double _p_cl_bayes(ET_forest *f, float *v, uint32_t curtail, bool smooth):
    return ET_forest_predict_class_bayes(f, v, curtail, smooth)


cdef class Forest:

    cdef ET_forest *_forest

    def __dealloc__(self):
        if self._forest:
            ET_forest_destroy(self._forest)
            free(self._forest)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict(self, np.ndarray[np.float32_t, ndim=2] X, bytes mode=None,
                curtail=1, smooth=False):
        cdef np.ndarray[np.float64_t, ndim=1] y
        cdef float *vector
        cdef int sample_idx, feature_idx
        cdef uint32_t _curtail = curtail
        cdef bool _smooth = smooth
        cdef double (*predict_f)(ET_forest *f, float *v, uint32_t c, bool s)

        if mode is None:
            predict_f = _p_simple
        elif mode == 'regression':
            predict_f = _p_regression
        elif mode == 'classify_majority':
            predict_f = _p_class_majority
        elif mode == 'classify_bayes':
            predict_f = _p_cl_bayes
        else:
            raise ValueError('unsupported predict mode: %r' % mode)

        if mode is None and curtail != 1:
            raise ValueError('curtail not supported for simple prediction')

        if mode != 'classify_bayes' and smooth == True:
            raise ValueError('smooth supported only for "classify_bayes" mode')

        y = numpy.empty(shape=(X.shape[0],), dtype=numpy.float64)
        vector = <float *> malloc(sizeof(float) * X.shape[1])
        if not vector:
            raise MemoryError()

        for sample_idx in xrange(X.shape[0]):
            for feature_idx in xrange(X.shape[1]):
                vector[feature_idx] = X[sample_idx, feature_idx]

            y[sample_idx] = predict_f(self._forest, vector, _curtail, _smooth)

        free(vector)
        return y

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict_proba(self, np.ndarray[np.float32_t, ndim=2] X,
                      curtail=1, smooth=False):
        cdef float *vector
        cdef int sample_idx, feature_idx
        cdef uint32_t _curtail = curtail
        cdef bool _smooth = smooth
        cdef class_probability_vec *cpv
        cdef class_probability *cp
        cdef np.ndarray[np.float64_t, ndim=1] classes = None
        cdef np.ndarray[np.float64_t, ndim=2] probas = None

        vector = <float *> malloc(sizeof(float) * X.shape[1])
        if not vector:
            raise MemoryError()

        for sample_idx in xrange(X.shape[0]):
            for feature_idx in xrange(X.shape[1]):
                vector[feature_idx] = X[sample_idx, feature_idx]

            cpv = ET_forest_predict_probability(self._forest, vector,
                                                _curtail, _smooth)
            if not cpv:
                raise MemoryError()

            if classes is None:
                shape = (X.shape[0], cpv.n)
                probas  = numpy.empty(shape=shape, dtype=numpy.float64)
                classes = numpy.empty(shape=(cpv.n,), dtype=numpy.float64)
                for i in xrange(cpv.n):
                    cp = &cpv.a[i]
                    classes[i] = cp.label
                    probas[sample_idx, i]  = cp.probability
            else:
                for i in xrange(cpv.n):
                    cp = &cpv.a[i]
                    probas[sample_idx, i]  = cp.probability

            free(cpv.a)
            free(cpv)

        free(vector)
        return classes, probas

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def neighbors(self, np.ndarray[np.float32_t, ndim=2] X, curtail=1):
        cdef float *vector
        cdef int sample_idx, feature_idx
        cdef uint32_t _curtail = curtail
        cdef neighbour_weight_vec *nwv
        cdef neighbour_weight *nw
        cdef np.ndarray[np.float64_t, ndim=2] adiacency

        adiacency = numpy.zeros(shape=(X.shape[0], self._forest.n_samples),
                                dtype=numpy.float64)

        vector = <float *> malloc(sizeof(float) * X.shape[1])
        if not vector:
            raise MemoryError()

        for sample_idx in xrange(X.shape[0]):
            for feature_idx in xrange(X.shape[1]):
                vector[feature_idx] = X[sample_idx, feature_idx]

            nwv = ET_forest_neighbors(self._forest, vector, _curtail)
            if not nwv:
                raise MemoryError()

            for i in xrange(nwv.n):
                nw = &nwv.a[i]
                adiacency[sample_idx, nw.key] = nw.weight

            free(nwv.a)
            free(nwv)

        free(vector)
        return adiacency

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def feature_importance(self):
        cdef uint32_t n_features = self._forest.n_features
        cdef double_vec *c_feat_imp = ET_forest_feature_importance(self._forest)
        cdef np.ndarray[np.float64_t, ndim=1] feat_imp

        if c_feat_imp == NULL:
            raise MemoryError()

        feat_imp = numpy.empty(shape=(n_features,))
        for i in xrange(n_features):
            feat_imp[i] = c_feat_imp.a[i]

        free(c_feat_imp.a)
        free(c_feat_imp)

        return feat_imp


cdef Forest forest_factory(ET_forest *forest):
    cdef Forest instance = Forest.__new__(Forest)
    instance._forest = forest
    return instance


# TODO use cython typed memoryviews to avoid copying
@cython.boundscheck(False)
@cython.wraparound(False)
def convert_to_problem(np.ndarray[np.float32_t, ndim=2] X not None,
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
    cdef np.ndarray[np.float32_t, ndim=2] X
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


