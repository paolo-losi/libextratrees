#include "extratrees.h"
#include "train.h"
#include "test.h"
#include "log.h"


float big_vectors[] = { 1., 3., 2., 3., 0.,
                        4., 1., 6., 2., 1.,
                        1., 1., 1., 1., 1.,
                        4., 6., 5., 7., 5.,
                        8., 6., 3., 3., 1.,
                        2., 2., 2., 3., 2. };
double big_labels[] = { 2., 2., 1., 3., 1. };


float small_vectors[] = { 1., 3., 2.,
                          4., 1., 6.,
                          1., 1., 1.,
                          4., 6., 5.,
                          7., 8., 0. };
double small_labels[] = { 2., 2., 2. };


void test_leaf() {
    test_header();

    ET_problem prob;
    ET_params params;
    ET_tree tree;
    tree_builder tb;
    uint32_t seed[] = {0, 0, 0, 0};

    problem_init(&prob, small_vectors, small_labels);
    ET_problem_print(&prob, stderr);

    EXTRA_TREE_DEFAULT_REGR_PARAMS(prob, params);

    check_mem( !tree_builder_init(&tb, &prob, &params, seed) );
    tree = build_tree(&tb);

    exit:
    tree_builder_destroy(&tb);
    tree_destroy(tree);
}


void test_split_regression() {
    test_header();

    ET_problem prob;
    ET_params params;
    ET_tree tree;
    tree_builder tb;
    uint32_t seed[] = {0, 0, 0, 0};

    problem_init(&prob, big_vectors, big_labels);
    ET_problem_print(&prob, stderr);

    EXTRA_TREE_DEFAULT_REGR_PARAMS(prob, params);

    check_mem( !tree_builder_init(&tb, &prob, &params, seed) );
    tree = build_tree(&tb);

    exit:
    tree_builder_destroy(&tb);
    tree_destroy(tree);
}


void test_split_classification() {
    test_header();

    ET_problem prob;
    ET_params params;
    ET_tree tree;
    tree_builder tb;
    uint32_t seed[] = {0, 0, 0, 0};

    problem_init(&prob, big_vectors, big_labels);
    ET_problem_print(&prob, stderr);

    EXTRA_TREE_DEFAULT_CLASSIF_PARAMS(prob, params);

    check_mem( !tree_builder_init(&tb, &prob, &params, seed) );
    tree = build_tree(&tb);

    exit:
    tree_builder_destroy(&tb);
    tree_destroy(tree);
}


void test_forest() {
    test_header();

    ET_problem prob;
    ET_params params;
    ET_forest *forest;

    problem_init(&prob, big_vectors, big_labels);
    ET_problem_print(&prob, stderr);

    EXTRA_TREE_DEFAULT_REGR_PARAMS(prob, params);
    params.number_of_trees = 10;
    params.number_of_features_tested = 1;
    params.select_features_with_replacement = true;

    forest = ET_forest_build(&prob, &params);
    ET_forest_destroy(forest);
    free(forest);
}


int main() {
    test_leaf();
    test_split_classification();
    test_split_regression();
    test_forest();
    return 0;
}
