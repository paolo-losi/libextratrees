#ifndef __TEST_H__
#define __TEST_H__

#include <stdio.h>

#define test_header() fprintf(stderr, ">>> test: %s\n", __func__)

#define init_problem(p, vectors, labels)                                       \
    do {                                                                       \
        (p)->vectors = vectors;                                                \
        (p)->labels = labels;                                                  \
        (p)->n_samples  = sizeof(labels)  / sizeof(*labels);                   \
        (p)->n_features = sizeof(vectors) / sizeof(*vectors) / (p)->n_samples; \
    } while(0);

#endif
