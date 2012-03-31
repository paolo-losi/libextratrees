#include "extratrees.h"
#include "util.h"
#include "log.h"
#include "counter.h"


static int compute_class_frequency(ET_forest *forest) {
    ET_class_counter *cc = NULL;
    cc = ET_class_counter_new();
    check_mem(cc);

    for(size_t i = 0; i < forest->n_samples; i++) {
        ET_class_counter_incr(cc, forest->labels[i]);
    }

    ET_class_counter_sort(cc);

    forest->class_frequency = cc;
    return 0;

    exit:
    return -1;
}


// --- tree prediction utils ---

typedef struct {
    node_processor f;
    void *data;
} filter_leaves_data;

static void filter_leaves(ET_base_node *node, filter_leaves_data *fld) {
    if IS_LEAF(node) {
        fld->f(node, fld->data);
    }
}

static void tree_lookup(ET_tree tree, float *vector, uint32_t curtail_min_size,
                        node_processor f, void *data) {

    ET_base_node *node = tree;
    filter_leaves_data fld;

    while(1) {
        switch(node->type) {
            case ET_SPLIT_NODE: {
                ET_base_node *next_node = NULL;
                ET_split_node *split = CAST_SPLIT(node);
                next_node = (vector[split->feature_id] <= split->threshold) ?
                             split->lower_node : split->higher_node;
                if (next_node->n_samples < curtail_min_size)
                    goto curtail;
                node = next_node;
                break;
            }
            case ET_LEAF_NODE: {
                f(node, data);
                return;
            }
            default:
                sentinel("unexpected split type %c", node->type);
        }
    }

    curtail:
    fld = (filter_leaves_data) {f, data};
    tree_navigate(node, (node_processor) filter_leaves, &fld);

    exit:
    return;
}


// --- tree prediction ---

// * neighbors

static void append_neighbors(ET_base_node *node, uint_vec *neighbors) {
    uint_vec *new_neighbors = &CAST_LEAF(node)->indexes;
    kv_extend(uint32_t, *neighbors, *new_neighbors);
}

static uint_vec *tree_neighbors(ET_tree tree, float *vector,
                                uint32_t curtail_min_size) {
    uint_vec *sample_idxs = NULL;

    sample_idxs = malloc(sizeof(uint_vec));
    check_mem(sample_idxs);
    kv_init(*sample_idxs);

    tree_lookup(tree, vector, curtail_min_size,
                (node_processor) append_neighbors, sample_idxs);

    exit:
    return sample_idxs;
}

// * regression

typedef struct {
    double sum;
    uint32_t count;
    double *labels;
} sum_count;

static void regression_node_processor(ET_base_node *node, sum_count *sc) {
    ET_leaf_node *lf = CAST_LEAF(node);

    if (lf->constant) {
        uint32_t first_sample_idx = kv_A(lf->indexes, 0);
        double label = sc->labels[first_sample_idx];
        sc->sum += label * node->n_samples;
    } else {
        for(size_t i = 0; i < kv_size(lf->indexes); i++) {
            uint32_t sample_idx = kv_A(lf->indexes, i);
            double label = sc->labels[sample_idx];
            sc->sum += label;
        }
    }
    sc->count += node->n_samples;
}

static double tree_regression(ET_tree tree, float *vector,
                              uint32_t curtail_min_size, double *labels) {
    sum_count sc = {0, 0, labels};
    tree_lookup(tree, vector, curtail_min_size,
                (node_processor) regression_node_processor, &sc);
    return sc.sum / (double) sc.count;
}

// * classification

typedef struct {
    ET_class_counter *class_counter;
    double *labels;
} class_freq_labels;

static void class_freq_node_processor(ET_base_node *node,
                                      class_freq_labels *cfl) {
    ET_leaf_node *ln = CAST_LEAF(node);

    if (ln->constant) {
        uint32_t first_sample_idx = kv_A(ln->indexes, 0);
        double class = cfl->labels[first_sample_idx];
        ET_class_counter_incr_n(cfl->class_counter, class, node->n_samples);
    } else {
        for(size_t i=0; i < kv_size(ln->indexes); i++) {
            uint32_t sample_idx = kv_A(ln->indexes, i);
            double class = cfl->labels[sample_idx];
            ET_class_counter_incr(cfl->class_counter, class);
        }
    }
}

static ET_class_counter *tree_classification(ET_tree tree, float *vector,
                                             uint32_t curtail_min_size,
                                             double *labels) {
    ET_class_counter *cc = NULL;
    cc = ET_class_counter_new();
    check_mem(cc);

    class_freq_labels cfl = {cc, labels};
    tree_lookup(tree, vector, curtail_min_size,
                (node_processor) class_freq_node_processor, &cfl);

    exit:
    return cc;
}

// --- forest prediction ---


uint_vec **ET_forest_neighbors_detail(ET_forest *forest, float *vector,
                                         uint32_t curtail_min_size) {
    uint32_t n_trees = kv_size(forest->trees);
    uint_vec **neighbors_array = NULL;

    neighbors_array = malloc(sizeof(uint_vec *) * n_trees);
    check_mem(neighbors_array);

    for(uint32_t i = 0; i < n_trees; i++) {
        ET_tree t = kv_A(forest->trees, i);
        neighbors_array[i] = tree_neighbors(t, vector, curtail_min_size);
    }

    exit:
    return neighbors_array;
}


neighbour_weight_vec *ET_forest_neighbors(ET_forest *forest, float *vector,
                                          uint32_t curtail_min_size) {
    uint_vec **neigh_detail;
    neighbour_weight_vec *nwvec;
    size_t n_trees = kv_size(forest->trees);

    nwvec = malloc(sizeof(neighbour_weight_vec));
    check_mem(nwvec);
    kv_init(*nwvec);

    neigh_detail = ET_forest_neighbors_detail(forest, vector, curtail_min_size);
    check_mem(neigh_detail);

    for(size_t i = 0; i < n_trees; i++) {
        uint_vec *tree_neighs = neigh_detail[i];
        double incr = 1.0 / (double) (kv_size(*tree_neighs) * n_trees);

        for(size_t j = 0; j < kv_size(*tree_neighs); j++) {
            neighbour_weight *nw = NULL;
            uint32_t sample_idx = kv_A(*tree_neighs, j);

            kal_getp(*nwvec, sample_idx, nw);
            if (nw == NULL) {
                kv_push(neighbour_weight, *nwvec,
                        ((neighbour_weight) { sample_idx, incr }));
            } else {
                nw->weight += incr;
            }
        }
        kv_destroy(*tree_neighs);
        free(tree_neighs);
    }
    free(neigh_detail);

    exit:
    return nwvec;
}


double ET_forest_predict_class_majority(ET_forest *forest,
                                        float *vector,
                                        uint32_t curtail_min_size) {
    double best_class = 0;
    ET_class_counter tree_vote_counter;
    ET_class_counter_init(tree_vote_counter);
    SimpleRandomKISS2_t rand_state;

    simplerandom_kiss2_seed(&rand_state, 0, 1, 2, 3);

    for(size_t i = 0; i < kv_size(forest->trees); i++) {
        ET_class_counter *cc = NULL;
        ET_tree tree = kv_A(forest->trees, i);

        cc = tree_classification(tree, vector, curtail_min_size,
                                 forest->labels);
        check_mem(cc);

        // compute class vote for tree
        int32_t most_frequent_count = -1;
        double_vec best_classes;
        kv_init(best_classes);

        log_debug(" --- tree count # %zu", i);
        for(size_t k=0; k < kv_size(*cc); k++) {
            class_counter_elm *ce = &(kv_A(*cc, k));
            log_debug("class: %g count: %d", ce->key, ce->count);
            if (most_frequent_count < (int32_t) ce->count) {
                most_frequent_count = ce->count;
                kv_clear(best_classes);
                kv_push(double, best_classes, ce->key);
            } else if (most_frequent_count == (int32_t) ce->count) {
                kv_push(double, best_classes, ce->key);
            }
        }
        ET_class_counter_destroy(*cc);
        free(cc);

        // in case of tie, choose class randomly
        uint32_t best_count = kv_size(best_classes);
        uint32_t best_idx = best_count == 1 ? 0 : random_int(&rand_state,
                                                             best_count);
        double tree_best_class = kv_A(best_classes, best_idx);

        ET_class_counter_incr(&tree_vote_counter, tree_best_class);
        kv_destroy(best_classes);
    }

    // voting
    int32_t best_count = -1;

    log_debug(" --- global count");
    for(size_t i=0; i < kv_size(tree_vote_counter); i++) {
        class_counter_elm *ce = &(kv_A(tree_vote_counter, i));
        log_debug("class: %g count: %d", ce->key, ce->count);
        if (best_count < (int32_t) ce->count) {
            best_count = ce->count;
            best_class = ce->key;
        }
    }

    exit:
    kv_destroy(tree_vote_counter);
    return best_class;
}


double ET_forest_predict_regression(ET_forest *forest,
                                    float *vector,
                                    uint32_t curtail_min_size) {
    double y, sum = 0;
    uint32_t n_trees = kv_size(forest->trees);

    for(size_t i = 0; i < n_trees; i++) {
        ET_tree tree = kv_A(forest->trees, i);

        y = tree_regression(tree, vector, curtail_min_size, forest->labels);
        log_debug("tree #%zu regression prediction = %g", i, y);
        sum += y;
    }

    return sum / (double) n_trees;
}


class_probability_vec *ET_forest_predict_probability(ET_forest *forest,
                                                     float *vector,
                                                     uint32_t curtail_min_size,
                                                     bool smooth) {
    bool error = true;
    class_probability_vec *prob_vec = NULL;
    neighbour_weight_vec *nwvec = NULL;
    double n_trees = kv_size(forest->trees);

    prob_vec = malloc(sizeof(class_probability_vec));
    check_mem(prob_vec);
    kv_init(*prob_vec);

    if (forest->class_frequency == NULL) {
        compute_class_frequency(forest);
    }

    for(size_t i = 0; i < kv_size(*forest->class_frequency); i++) {
        double label = kv_A(*forest->class_frequency, i).key;
        kv_push(class_probability, *prob_vec, ((class_probability) {label, 0}));
    }

    for(size_t i = 0; i < n_trees; i++) {
        ET_class_counter *cc = NULL;
        ET_tree tree = kv_A(forest->trees, i);

        cc = tree_classification(tree, vector, curtail_min_size,
                                 forest->labels);
        check_mem(cc);

        double total = ET_class_counter_total(cc);

        for(size_t k = 0; k < kv_size(*cc); k++) {
            class_counter_elm *ce = &(kv_A(*cc, k));
            for(size_t j = 0; j < kv_size(*prob_vec); j++) {
                class_probability *cp = &kv_A(*prob_vec, j);
                if (cp->label == ce->key) {
                    cp->probability += ce->count / total / n_trees;
                    break;
                }
            }
        }
        ET_class_counter_destroy(*cc);
        free(cc);
    }

    if (smooth) {
        double n_samples = forest->n_samples;

        for(size_t i = 0; i < kv_size(*prob_vec); i++) {
            double unsmoothed_prob, prior_prob;
            class_probability *cp = &kv_A(*prob_vec, i);
            unsmoothed_prob = cp->probability;
            prior_prob = kv_A(*forest->class_frequency, i).count / n_samples;

            cp->probability = (1 - 1 / n_samples) * unsmoothed_prob +
                              (1 / n_samples) * prior_prob;
        }
    }

    error = false;

    exit:
    if (error && prob_vec != NULL) {
        kv_destroy(*prob_vec);
        free(prob_vec);
    }
    if (nwvec != NULL) {
        kv_destroy(*nwvec);
        free(nwvec);
    }
    return prob_vec;
}


double ET_forest_predict_class_bayes(ET_forest *forest, float *vector,
                                     uint32_t curtail_min_size, bool smooth) {
    double best_label = 0;
    double best_probability = -1;
    class_probability_vec *cpv = NULL;

    cpv = ET_forest_predict_probability(forest, vector,
                                        curtail_min_size, smooth);
    check_mem(cpv);

    for(size_t i = 0; i < kv_size(*cpv); i++) {
        class_probability *cp = &kv_A(*cpv, i);
        if (cp->probability > best_probability) {
            best_probability = cp->probability;
            best_label = cp->label;
        }
    }

    exit:
    if (cpv != NULL) {
        kv_destroy(*cpv);
        free(cpv);
    }
    return best_label;
}


double ET_forest_predict(ET_forest *forest, float *vector) {
    if (forest->params.regression) {
        return ET_forest_predict_regression(forest, vector, 1);
    } else {
        return ET_forest_predict_class_majority(forest, vector, 1);
    }
}


// --- feature importance ---

void node_diversity(ET_base_node *node, double_vec *diversity_reduction) {
    if (IS_SPLIT(node)) {
        double curr_reduction;
        ET_split_node *sn = CAST_SPLIT(node);
        curr_reduction = node->diversity - sn->higher_node->diversity
                                         - sn->lower_node->diversity;
        kv_A(*diversity_reduction, sn->feature_id) += curr_reduction;
    }
}


double_vec *ET_forest_feature_importance(ET_forest *forest) {
    double_vec *diversity_reduction = NULL;

    diversity_reduction = malloc(sizeof(double_vec));
    check_mem(diversity_reduction);

    kv_init(*diversity_reduction);
    kv_resize(double, *diversity_reduction, forest->n_features);
    for(uint32_t i = 0; i < forest->n_features; i++) {
        kv_A(*diversity_reduction, i) = 0;
    }

    for(uint32_t i = 0; i < forest->params.number_of_trees; i++) {
        tree_navigate(kv_A(forest->trees, i),
                      (node_processor) node_diversity,
                      diversity_reduction);
    }

    for(uint32_t i = 0; i < forest->n_features; i++) {
        ET_tree tree = kv_A(forest->trees, 0);
        double den = forest->params.number_of_trees * tree->diversity;
        kv_A(*diversity_reduction, i) /= den;
    }

    exit:
    return diversity_reduction;
}
