#ifndef __TEST_H__
#define __TEST_H__

#include <stdio.h>

#define test_header() fprintf(stderr, ">>> test: %s\n", __func__)

#define problem_init(p, _vectors, _labels)                                     \
    do {                                                                       \
        (p)->vectors = _vectors;                                               \
        (p)->labels = _labels;                                                 \
        (p)->n_samples  = sizeof(_labels) /sizeof(*_labels);                   \
        (p)->n_features = sizeof(_vectors)/sizeof(*_vectors) / (p)->n_samples; \
    } while(0);

#endif
