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

inline ET_class_counter *ET_class_counter_new() {
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

static int compare_on_label(const void *a, const void *b) {
    class_counter_elm *ea = (class_counter_elm *) a;
    class_counter_elm *eb = (class_counter_elm *) b;
    if (ea < eb) { return +1; } else
    if (ea > eb) { return -1; } else
    { return 0; }
}

inline void ET_class_counter_sort(ET_class_counter *cc) {
    qsort((void *) cc->a, kv_size(*cc), sizeof(class_counter_elm),
                                        &compare_on_label);
}

#endif
