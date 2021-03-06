#include <stdio.h>
#include <stdint.h>

#include "extratrees.h"
#include "problem.h"


void ET_problem_print(ET_problem *prob, FILE *fout) {
    fprintf(fout, "problem: samples=%d features=%d\n", prob->n_samples,
                                                       prob->n_features);
    for(uint32_t s = 0; s < prob->n_samples; s++) {
        fprintf(fout, "sample %d. label=%g features=", s, prob->labels[s]);
        for(uint32_t fid = 0; fid < prob->n_features; fid++) {
            if (fid) fprintf(fout, " ");
            fprintf(fout, "%g", PROB_GET(prob, s, fid));
        }
        fprintf(fout, "\n");
    }
}

void ET_problem_destroy(ET_problem *prob) {
    if (prob->labels)  free(prob->labels);
    if (prob->vectors) free(prob->vectors);
}
