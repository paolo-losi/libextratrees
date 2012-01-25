# ifndef EXTRA_TREES_H
# define EXTRA_TREES_H

# include <stdint.h>
# include <stdio.h>
# include <stdbool.h>
# include "kvec.h"

typedef kvec_t(uint32_t) uint_vec;

typedef struct {
    float *vectors;
    double *labels;
    uint32_t n_features;
    uint32_t n_samples;
} ET_problem;


ET_problem *ET_load_libsvm_file(char *fname);

void ET_print_problem(FILE *f, ET_problem *prob);

# endif 
