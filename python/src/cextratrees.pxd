from libcpp cimport bool
from libc.stdint cimport uint32_t


cdef extern from "extratrees.h":
    
    ctypedef struct ET_problem:
        float *vectors
        double *labels
        uint32_t n_features
        uint32_t n_samples

    cdef void ET_problem_destroy(ET_problem *prob)

    ctypedef struct ET_forest:
        pass

    cdef void ET_forest_destroy(ET_forest *forest)
    cdef ET_forest *ET_forest_build(ET_problem *problem, ET_params *parmas)
    cdef double ET_forest_predict(ET_forest *forest, float *vector)
    cdef double ET_forest_predict_regression(ET_forest *forest,
                                             float *vector,
                                             uint32_t curtail_min_size)
    cdef double ET_forest_predict_class_majority(ET_forest *forest,
                                             float *vector,
                                             uint32_t curtail_min_size)
    cdef double ET_forest_predict_class_bayes(ET_forest *forest,
                                             float *vector,
                                             uint32_t curtail_min_size,
                                             bool smooth)


    ctypedef struct ET_params:
        uint32_t number_of_features_tested
        uint32_t number_of_trees
        bool regression
        uint32_t min_split_size
        bool select_features_with_replacement
