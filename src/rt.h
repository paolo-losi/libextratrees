# ifndef RT_H
# define RT_H

# include <stdint.h>
# include <stdio.h>
# include "kvec.h"

typedef kvec_t(double) double_vec;
typedef kvec_t(int) int_vec;

typedef struct {
    float *vectors;
    double *labels;
    uint32_t n_features;
    uint32_t n_samples;
} rt_problem;


rt_problem *rt_load_libsvm_file(char *fname);

void rt_print_problem(FILE *f, rt_problem *prob);

# endif 
