#include <stdint.h>
#include "simplerandom.h"

// pick a random integer between [0, max_val)
uint32_t random_int(SimpleRandomKISS2_t *rnd, uint32_t max_val) {
    uint32_t i;
    do {
        i = simplerandom_kiss2_next(rnd);
    } while (i >= UINT32_MAX / max_val * max_val);
    return i % max_val;
}

// pick a random double between [0., 1.)
double random_double(SimpleRandomKISS2_t *rnd) {
    uint32_t i = simplerandom_kiss2_next(rnd);
    return i * 2.3283064365386963e-10;
}
