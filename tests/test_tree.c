#include "rt.h"
#include "tree.h"
#include "test.h"
#include "log.h"


void test_leaf() {
    test_header();

    rt_problem prob;
    tree_builder tb;
    int_vec samples_idx;
    rt_base_node *node;

    float vectors[] = { 1., 3., 2.,
                        4., 1., 6.,
                        1., 1., 1.,
                        4., 6., 5.,
                        7., 8., 0. };
    double labels[] = { 2., 2., 2. };

    problem_init(&prob, vectors, labels);
    check_mem(! tree_builder_init(&tb, &prob));

    EXTRA_TREE_DEFAULT_REGR_PARAMS(tb);

    rt_print_problem(stderr, &prob);

    kv_init(samples_idx);
    for(uint32_t i = 0; i < prob.n_samples; i++) {
        kv_push(int, samples_idx, i);
    }

    node = split_problem(&tb, &samples_idx);

    exit:
    kv_destroy(samples_idx);
    tree_builder_destroy(&tb);
    if (node) tree_destroy(node);
}


void test_split() {
    test_header();

    rt_problem prob;
    tree_builder tb;
    int_vec samples_idx;
    rt_base_node *node;

    float vectors[] = { 1., 3., 2., 3.,
                        4., 1., 6., 2.,
                        1., 1., 1., 1.,
                        4., 6., 5., 7.,
                        8., 6., 3., 3.,
                        2., 2., 2., 2. };
    double labels[] = { 2., 2., 1., 3. };

    problem_init(&prob, vectors, labels);
    check_mem(! tree_builder_init(&tb, &prob));

    EXTRA_TREE_DEFAULT_REGR_PARAMS(tb);

    rt_print_problem(stderr, &prob);

    kv_init(samples_idx);
    for(uint32_t i = 0; i < prob.n_samples; i++) {
        kv_push(int, samples_idx, i);
    }

    node = split_problem(&tb, &samples_idx);

    exit:
    kv_destroy(samples_idx);
    tree_builder_destroy(&tb);
    if (node) tree_destroy(node);
}


int main() {
    test_leaf();
    test_split();
    return 0;
}
