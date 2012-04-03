from libcpp cimport bool
from libc.stdint cimport uint32_t


cdef extern from "extratrees.h":
    
    ctypedef struct ET_problem:
        float *vectors
        double *labels
        uint32_t n_features
        uint32_t n_samples

    cdef void ET_problem_destroy(ET_problem *prob)
    cdef ET_problem *ET_load_libsvm_file(char *fname)

    ctypedef struct ET_base_node:
        pass
    
    ctypedef ET_base_node *ET_tree

    ctypedef struct tree_vec:
        size_t n, m
        ET_tree *a

    ctypedef struct ET_forest:
        uint32_t n_features
        uint32_t n_samples
        tree_vec trees

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
    cdef double *ET_forest_neighbors(ET_forest *forest,
                                             float *vector,
                                             uint32_t curtail_min_size)
    cdef double *ET_forest_feature_importance(ET_forest *forest,
                                             uint32_t curtail_min_size)

    ctypedef struct uchar_vec:
        size_t n, m
        unsigned char *a

    cdef void ET_tree_dump(ET_tree tree, uchar_vec *buffer)
    cdef void ET_forest_dump(ET_forest *forest, uchar_vec *buffer,
                                                bool with_trees)
    
    cdef ET_tree ET_tree_load(unsigned char **bufferp)
    cdef ET_forest *ET_forest_load(unsigned char **bufferp)
