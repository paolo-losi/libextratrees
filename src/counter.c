#include "counter.h"

extern inline ET_class_counter *ET_class_counter_new(void);
extern inline void ET_class_counter_incr(ET_class_counter *cc, double label);
extern inline void ET_class_counter_incr_n(ET_class_counter *cc, double label,
                                                               uint32_t count);
extern inline void ET_class_counter_sort(ET_class_counter *cc);
