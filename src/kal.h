#ifndef KAL_H
#define KAL_H

#include <stddef.h>
#include "kvec.h"

#define kal_getp(v, k, elmp)                          \
    do { elmp = NULL;                                 \
         for (size_t i = 0; i < kv_size((v)); i++) {  \
            if (kv_A((v), i).key == (k)) {            \
                elmp = &kv_A((v), i);                 \
                break;                                \
            }                                         \
         }                                            \
    } while(0);

#endif
