#include "rt.h"
#include "tree.h"
#include "test.h"
#include "log.h"


void test_leaf() {
    test_header();

    rt_problem prob;
    tree_builder tb;
    int_vec samples_idx;

    double vectors[] = { 1., 3.,
                         2., 4.,
                         1., 6.};
    double labels[]  = { 2., 2., 2. };

    init_problem(&prob, vectors, labels);
    tb.prob = &prob;
    simplerandom_kiss2_seed(&tb.rand_state, 0, 0, 0, 0);

    tb.features_deck = malloc(sizeof(double) * prob.n_features);
    check_mem(tb.features_deck);
    for(uint32_t i = 0; i < prob.n_features; i++) {
        tb.features_deck[i] = i;
    }

    kv_init(samples_idx);
    for(uint32_t i = 0; i < prob.n_samples; i++) {
        kv_push(int, samples_idx, i);
    }
    tb.params.number_of_features_tested = 2;
    tb.params.number_of_trees = 1;
    tb.params.regression = 1;
    tb.params.min_split_size = 1;
    tb.params.select_features_with_replacement = 1;

    rt_print_problem(stderr, &prob);

    split_problem(&tb, &samples_idx);

    exit:
    free(tb.features_deck);
}


int main() {
    test_leaf();
    return 0;
}
