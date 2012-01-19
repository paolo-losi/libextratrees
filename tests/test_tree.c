#include "rt.h"
#include "test.h"

static double test_vecs[] = { 1., 3.,
                              2., 4.,
                              1., 6. };

static double test_labs[] = { 2., 3., 4. };

static rt_problem prob = (rt_problem) {
   .vectors = test_vecs,
   .labels  = test_labs,
   .n_features = 3,
   .n_samples  = 2,
};

void test_leaf() {
    test_header();
    rt_print_problem(&prob);
}

int main() {
    test_leaf();
    return 0;
}
