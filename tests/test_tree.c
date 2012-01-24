#include "rt.h"
#include "tree.h"
#include "test.h"
#include "log.h"


void test_leaf() {
    test_header();

    rt_problem prob;
    rt_params params;
    rt_tree tree;
    tree_builder tb;

    float vectors[] = { 1., 3., 2.,
                        4., 1., 6.,
                        1., 1., 1.,
                        4., 6., 5.,
                        7., 8., 0. };
    double labels[] = { 2., 2., 2. };

    problem_init(&prob, vectors, labels);
    rt_print_problem(stderr, &prob);

    EXTRA_TREE_DEFAULT_REGR_PARAMS(prob, params);

    check_mem( !tree_builder_init(&tb, &prob, &params, 0) );
    tree = build_tree(&tb);

    exit:
    tree_builder_destroy(&tb);
    tree_destroy(tree);
}


void test_split() {
    test_header();

    rt_problem prob;
    rt_params params;
    rt_tree tree;
    tree_builder tb;

    float vectors[] = { 1., 3., 2., 3., 0.,
                        4., 1., 6., 2., 1.,
                        1., 1., 1., 1., 1.,
                        4., 6., 5., 7., 5.,
                        8., 6., 3., 3., 1.,
                        2., 2., 2., 3., 2. };
    double labels[] = { 2., 2., 1., 3., 1. };

    problem_init(&prob, vectors, labels);
    rt_print_problem(stderr, &prob);

    EXTRA_TREE_DEFAULT_REGR_PARAMS(prob, params);

    check_mem( !tree_builder_init(&tb, &prob, &params, 0) );
    tree = build_tree(&tb);

    exit:
    tree_builder_destroy(&tb);
    tree_destroy(tree);
}


int main() {
    test_leaf();
    test_split();
    return 0;
}
