from libc.stdint cimport uint32_t


cdef extern from "extratrees.h":
    
    ctypedef struct ET_problem:
        float *vectors
        double *labels
        uint32_t n_features
        uint32_t n_samples

    cdef void ET_problem_destroy(ET_problem *prob)
