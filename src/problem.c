#include <stdio.h>
#include <stdint.h>

#include "rt.h"
#include "problem.h"


void rt_print_problem(rt_problem *prob) {
    printf("problem: samples=%d features=%d\n", prob->n_samples,
                                                prob->n_features);
    for(uint32_t s = 0; s < prob->n_samples; s++) {
        printf("sample %d. label=%f features=", s, prob->labels[s]);
        for(uint32_t fid = 0; fid < prob->n_features; fid++) {
            if (fid) printf(" ");
            printf("%f", PROB_GET(prob, s, fid));
        }
        printf("\n");
    }
}
