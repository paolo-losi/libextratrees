#ifndef RT_UTIL_H_
#define RT_UTIL_H_

#include "simplerandom.h"

uint32_t random_int(SimpleRandomKISS2_t *rnd, uint32_t max_val); 
double random_double(SimpleRandomKISS2_t *rnd);

#define UNUSED(x) (void)(x)

#endif
