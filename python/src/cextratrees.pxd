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
        uint32_t n_features
        uint32_t n_samples

    cdef void ET_forest_destroy(ET_forest *forest)
    cdef ET_forest *ET_forest_build(ET_problem *problem, ET_params *parmas)

    ctypedef struct ET_params:
        
        uint32_t number_of_features_tested
        uint32_t number_of_trees
        bool regression
        uint32_t min_split_size
        bool select_features_with_replacement

    ctypedef struct class_probability:
        double label
        double probability

    ctypedef struct class_probability_vec:
        size_t n, m
        class_probability *a

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
    cdef class_probability_vec *ET_forest_predict_probability(ET_forest *forest,
                                             float *vector,
                                             uint32_t curtail_min_size,
                                             bool smooth)

    ctypedef struct neighbour_weight:
        uint32_t key
        double weight

    ctypedef struct neighbour_weight_vec:
        size_t n, m
        neighbour_weight *a

    cdef neighbour_weight_vec *ET_forest_neighbors(ET_forest *forest,
                                             float *vector,
                                             uint32_t curtail_min_size)

    ctypedef struct double_vec:
        size_t n, m
        double *a

    cdef double_vec *ET_forest_feature_importance(ET_forest *forest) 
        

    
