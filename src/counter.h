#ifndef COUNTER_H
#define COUNTER_H

#include <stdint.h>
#include <stdlib.h>
#include "kvec.h"
#include "kal.h"
#include "log.h"

#ifndef ET_CLASS_COUNTER
#define ET_CLASS_COUNTER
typedef struct ET_class_counter_struct ET_class_counter;
#endif

typedef struct {
    double key;
    uint32_t count;
} class_counter_elm;


struct ET_class_counter_struct {
    size_t n, m;
    class_counter_elm *a;
};

#define ET_class_counter_init(cc) kv_init(cc)
#define ET_class_counter_clear(cc) kv_clear(cc)
#define ET_class_counter_destroy(cc) kv_destroy(cc)

inline ET_class_counter *ET_class_counter_new(void) {
    ET_class_counter *cc = NULL;
    cc = malloc(sizeof(ET_class_counter));
    check_mem(cc);
    kv_init(*cc);

    exit:
    return cc;
}

inline void ET_class_counter_incr(ET_class_counter *cc, double label) {
    class_counter_elm *cce;
    kal_getp(*cc, label, cce);
    if(cce == NULL) {
        kv_push(class_counter_elm, *cc, ((class_counter_elm) {label, 1}));
    } else {
        cce->count += 1;
    };
}

inline void ET_class_counter_incr_n(ET_class_counter *cc, double label,
                                                          uint32_t count) {
    class_counter_elm *cce;
    kal_getp(*cc, label, cce);
    if(cce == NULL) {
        kv_push(class_counter_elm, *cc, ((class_counter_elm) {label, count}));
    } else {
        cce->count += count;
    };
}

int compare_on_label(const void *a, const void *b);

inline void ET_class_counter_sort(ET_class_counter *cc) {
    qsort((void *) cc->a, kv_size(*cc), sizeof(class_counter_elm),
                                        &compare_on_label);
}

inline uint32_t ET_class_counter_total(ET_class_counter *cc) {
    double tot = 0;
    for(size_t i = 0; i < kv_size(*cc); i++) {
        tot += kv_A(*cc, i).count;
    }
    return tot;
}

#endif
