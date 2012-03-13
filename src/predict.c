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


static void append_neighbors(ET_base_node *node, uint_vec *neighbors) {
    if IS_LEAF(node) {
        uint_vec *new_neighbors = &CAST_LEAF(node)->indexes;
        kv_extend(uint32_t, *neighbors, *new_neighbors);
    }
}


static uint_vec *tree_neighbors(ET_tree tree, float *vector,
                                uint32_t curtail_min_size) {
    ET_base_node *node = tree;
    uint_vec *sample_idxs = NULL;

    sample_idxs = malloc(sizeof(uint_vec));
    check_mem(sample_idxs);
    kv_init(*sample_idxs);

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
                kv_copy(uint32_t, *sample_idxs, CAST_LEAF(node)->indexes);
                return sample_idxs;
            }
            default:
                sentinel("unexpected split type %c", node->type);
        }
    }

    curtail:
    tree_navigate(node, (node_processor) append_neighbors, sample_idxs);
    return sample_idxs;

    exit:
    //sentinel
    return NULL;

}


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
    uint_vec **neigh_detail;
    ET_class_counter class_counter_global, class_counter_tree;

    ET_class_counter_init(class_counter_global);
    ET_class_counter_init(class_counter_tree);
    neigh_detail = ET_forest_neighbors_detail(forest, vector, curtail_min_size);
    check_mem(neigh_detail);

    for(size_t i=0; i < kv_size(forest->trees); i++) {
        double most_frequent_class = 0;
        int32_t most_frequent_count = -1;
        uint_vec *tree_neighs = neigh_detail[i];

        log_debug(" --- tree count # %zu", i);
        kv_clear(class_counter_tree);

        // count class freq in tree
        for(size_t j=0; j < kv_size(*tree_neighs); j++) {
            uint32_t sample_idx = kv_A(*tree_neighs, j);
            double class = forest->labels[sample_idx];
            ET_class_counter_incr(&class_counter_tree, class);
        }

        // look for most frequent class in tree
        for(size_t k=0; k < kv_size(class_counter_tree); k++) {
            class_counter_elm *ce = &(kv_A(class_counter_tree, k));
            log_debug("class: %g count: %d", ce->key, ce->count);
            if (most_frequent_count < (int32_t) ce->count) {
                most_frequent_count = ce->count;
                most_frequent_class = ce->key;
            } else if (most_frequent_count == (int32_t) ce->count) {
                log_warn("labels in leaf are balanced");
            }
        }

        // record most frequent class for tree
        ET_class_counter_incr(&class_counter_global, most_frequent_class);

        kv_destroy(*tree_neighs);
    }

    free(neigh_detail);
    {
        double best_class = 0;
        int32_t best_count = -1;

        log_debug(" --- global count");
        for(size_t i=0; i < kv_size(class_counter_global); i++) {
            class_counter_elm *ce4 = &(kv_A(class_counter_global, i));
            log_debug("class: %g count: %d", ce4->key, ce4->count);
            if (best_count < (int32_t) ce4->count) {
                best_count = ce4->count;
                best_class = ce4->key;
            }
        }
        return best_class;
    }

    exit:
    return 0.0;
}


double ET_forest_predict_regression(ET_forest *forest,
                                    float *vector,
                                    uint32_t curtail_min_size) {
    neighbour_weight_vec *nwvec = NULL;
    double y = 0.0;

    nwvec = ET_forest_neighbors(forest, vector, curtail_min_size);
    for(size_t i = 0; i < kv_size(*nwvec); i++) {
        neighbour_weight nw = kv_A(*nwvec, i);
        double weight = nw.weight;
        uint32_t sample_idx = nw.key;
        log_debug("sample_idx: %d weight: %g", sample_idx, weight);
        double val = forest->labels[sample_idx];

        y += val * weight;
    }

    kv_destroy(*nwvec);
    free(nwvec);

    return y;
}


class_probability_vec *ET_forest_predict_probability(ET_forest *forest,
                                                     float *vector,
                                                     uint32_t curtail_min_size,
                                                     bool smooth) {
    class_probability_vec *prob_vec = NULL;
    neighbour_weight_vec *nwvec = NULL;

    prob_vec = malloc(sizeof(class_probability_vec));
    check_mem(prob_vec);
    kv_init(*prob_vec);

    if (!forest->class_frequency) {
        compute_class_frequency(forest);
    }

    for(size_t i = 0; i < kv_size(*forest->class_frequency); i++) {
        double label = kv_A(*forest->class_frequency, i).key;
        kv_push(class_probability, *prob_vec, ((class_probability) {label, 0}));
    }

    nwvec = ET_forest_neighbors(forest, vector, curtail_min_size);
    check_mem(nwvec);

    for(size_t i = 0; i < kv_size(*nwvec); i++) {
        uint32_t sample_idx = kv_A(*nwvec, i).key;
        double weight       = kv_A(*nwvec, i).weight;
        double label = forest->labels[sample_idx];

        for(size_t j = 0; j < kv_size(*prob_vec); j++) {
            class_probability *cp = &kv_A(*prob_vec, i);
            if (cp->label == label) {
                cp->probability += weight;
                break;
            }
        }
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

    free(nwvec);
    return prob_vec;

    exit:
    if (prob_vec == NULL) free(prob_vec);
    if (nwvec == NULL) free(nwvec);
    return prob_vec;

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
