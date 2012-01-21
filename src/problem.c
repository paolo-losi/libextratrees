#include <stdio.h>
#include <stdint.h>

#include "rt.h"
#include "problem.h"


void rt_print_problem(FILE *fout, rt_problem *prob) {
    fprintf(fout, "problem: samples=%d features=%d\n", prob->n_samples,
                                                       prob->n_features);
    for(uint32_t s = 0; s < prob->n_samples; s++) {
        fprintf(fout, "sample %d. label=%f features=", s, prob->labels[s]);
        for(uint32_t fid = 0; fid < prob->n_features; fid++) {
            if (fid) fprintf(fout, " ");
            fprintf(fout, "%f", PROB_GET(prob, s, fid));
        }
        fprintf(fout, "\n");
    }
}
