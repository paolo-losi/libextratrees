import numpy
cimport cython
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint32_t
from libc cimport math
from libcpp cimport bool
from cextratrees cimport (ET_problem, ET_problem_destroy, ET_load_libsvm_file,
                          ET_forest, ET_forest_destroy, ET_forest_build,
                          ET_forest_predict, ET_forest_predict_regression,
                          ET_forest_predict_class_majority,
                          ET_forest_predict_probability,
                          ET_forest_neighbors, ET_params,
                          ET_forest_predict_class_bayes,
                          class_probability_vec, class_probability,
                          ET_forest_feature_importance, uchar_vec,
                          ET_forest_dump, ET_forest_load, ET_tree, tree_vec,
                          ET_tree_dump, ET_tree_load)


cdef class Problem:

    cdef ET_problem *_prob
    cdef _X

    def __dealloc__(self):
        if self._prob:
            if self._X is not None:
                free(self._prob.labels)
                free(self._prob)
            else:
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


cdef Problem problem_factory(ET_problem *prob,
                           np.ndarray[np.float32_t, ndim=2, mode='fortran'] X):
    cdef Problem instance = Problem.__new__(Problem)
    instance._prob = prob
    instance._X = X
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
    def predict(self, np.ndarray[np.float32_t, ndim=2] X not None,
                bytes mode=None, curtail=1, smooth=False):
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
    def predict_proba(self, np.ndarray[np.float32_t, ndim=2] X not None,
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
    def neighbors(self, np.ndarray[np.float32_t, ndim=2] X not None, curtail=1):
        cdef float *vector
        cdef double *weights
        cdef int sample_idx, feature_idx
        cdef uint32_t _curtail = curtail
        cdef np.ndarray[np.float64_t, ndim=2] adiacency

        adiacency = numpy.empty(shape=(X.shape[0], self._forest.n_samples),
                                dtype=numpy.float64)

        vector = <float *> malloc(sizeof(float) * X.shape[1])
        if not vector:
            raise MemoryError()

        for sample_idx in xrange(X.shape[0]):
            for feature_idx in xrange(X.shape[1]):
                vector[feature_idx] = X[sample_idx, feature_idx]

            weights = ET_forest_neighbors(self._forest, vector, _curtail)
            if not weights:
                raise MemoryError()

            for feature_idx in xrange(X.shape[1]):
                adiacency[sample_idx, feature_idx] = weights[feature_idx]

            free(weights)

        free(vector)
        return adiacency

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def feature_importance(self, curtail=1):
        cdef uint32_t _curtail = curtail
        cdef uint32_t n_features = self._forest.n_features
        cdef np.ndarray[np.float64_t, ndim=1] feat_imp
        cdef double *c_feat_imp = \
                           ET_forest_feature_importance(self._forest, _curtail)

        if c_feat_imp == NULL:
            raise MemoryError()

        feat_imp = numpy.empty(shape=(n_features,))
        for i in xrange(n_features):
            feat_imp[i] = c_feat_imp[i]

        free(c_feat_imp)
        return feat_imp

    def __reduce__(self):
        cdef uchar_vec buffer
        cdef bytes pickle_data
        cdef char *cstring
        buffer.n, buffer.m, buffer.a = 0, 0, NULL
        ET_forest_dump(self._forest, &buffer, False);
        cstring = <char *> buffer.a
        try:
            pickle_data = cstring[:buffer.n]
        finally:
            free(buffer.a)

        return (forest_unpickler, (pickle_data,), None,
                ForestIterator(self), None)

    def extend(self, l):
        for e in l:
            self.append(e)

    cpdef append(self, bytes pickle_data):
        cdef tree_vec *trees = &self._forest.trees
        cdef unsigned char *buffer = pickle_data

        cdef ET_tree tree = ET_tree_load(&buffer)
        trees.a[trees.n] = tree
        trees.n += 1


cdef Forest forest_factory(ET_forest *forest):
    cdef Forest instance = Forest.__new__(Forest)
    instance._forest = forest
    return instance


def forest_unpickler(bytes pickle_data):
    cdef unsigned char *buffer = pickle_data
    cdef ET_forest *cforest = ET_forest_load(&buffer)
    if not cforest:
        raise MemoryError()
    return forest_factory(cforest)


cdef class ForestIterator:

    cdef uint32_t i
    cdef Forest _forest

    def __cinit__(self, Forest forest):
        self._forest = forest
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef uchar_vec buffer
        cdef bytes pickle_data
        cdef char *cstring
        buffer.n, buffer.m, buffer.a = 0, 0, NULL

        if self.i >= self._forest._forest.trees.n:
            raise StopIteration()

        cdef ET_tree tree = self._forest._forest.trees.a[self.i]
        ET_tree_dump(tree, &buffer)

        cstring = <char *> buffer.a
        try:
            pickle_data = cstring[:buffer.n]
        finally:
            free(buffer.a)

        self.i += 1
        return pickle_data


@cython.boundscheck(False)
@cython.wraparound(False)
def convert_to_problem(
    np.ndarray[np.float32_t, ndim=2, mode='fortran'] X not None,
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

    cprob.vectors = <float *> np.PyArray_DATA(X)
    cprob.labels = <double *> malloc(sizeof(double) * n_samples)
    if not cprob.labels or not cprob.vectors:
        raise MemoryError()

    for i in xrange(n_samples):
        cprob.labels[i] = y[i]

    return problem_factory(cprob, X)


@cython.boundscheck(False)
@cython.wraparound(False)
def convert_from_problem(Problem prob not None):
    cdef int i, j
    cdef ET_problem *cprob = prob._prob
    cdef np.ndarray[np.float32_t, ndim=2] X
    cdef np.ndarray[np.float64_t, ndim=1] y

    if prob._X is None:
        X = numpy.empty(shape=(cprob.n_samples, cprob.n_features),
                        dtype=numpy.float32)
        for j in xrange(cprob.n_features):
            for i in xrange(cprob.n_samples):
                X[i, j] = cprob.vectors[j * cprob.n_samples + i]
    else:
        X = prob._X

    y = numpy.empty(shape=(cprob.n_samples,),
                    dtype=numpy.float64)

    for i in xrange(cprob.n_samples):
        y[i] = cprob.labels[i]

    return X, y


def train(X, y, **params):
    return convert_to_problem(X, y)._train(**params)


def load(bytes fname):
    cdef ET_problem *cprob = ET_load_libsvm_file(fname)

    if not cprob:
        raise MemoryError()

    return problem_factory(cprob, None)
