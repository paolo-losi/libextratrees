#include "counter.h"

int compare_on_label(const void *a, const void *b) {
    class_counter_elm *ea = (class_counter_elm *) a;
    class_counter_elm *eb = (class_counter_elm *) b;
    if (ea->key < eb->key) { return -1; } else
    if (ea->key > eb->key) { return +1; } else
    { return 0; }
}

