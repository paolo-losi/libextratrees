#include "extratrees.h"
#include "util.h"
#include "log.h"
#include "kal.h"


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
    kvec_t(class_counter_elm) class_counter_global, class_counter_tree;

    kv_init(class_counter_global);
    kv_init(class_counter_tree);
    neigh_detail = ET_forest_neighbors_detail(forest, vector, curtail_min_size);
    check_mem(neigh_detail);

    for(size_t i=0; i < kv_size(forest->trees); i++) {
        class_counter_elm *ce2;
        double most_frequent_class;
        int32_t most_frequent_count = -1;
        uint_vec *tree_neighs = neigh_detail[i];

        log_debug(" --- tree count # %zu", i);
        kv_clear(class_counter_tree);
        for(size_t j=0; j < kv_size(*tree_neighs); j++) {
            class_counter_elm *ce1;
            uint32_t sample_idx = kv_A(*tree_neighs, j);
            double class = forest->labels[sample_idx];

            // count class freq in tree
            kal_getp(class_counter_tree, class, ce1);
            if (ce1 == NULL) {
                kv_push(class_counter_elm, class_counter_tree,
                        ((class_counter_elm) {class, 1}));
            } else {
                ce1->count += 1;
            }

        }

        // look for most frequent class in tree
        for(size_t k=0; k < kv_size(class_counter_tree); k++) {
            class_counter_elm *ce3 = &(kv_A(class_counter_tree, k));
            log_debug("class: %g count: %d", ce3->key, ce3->count);
            if (most_frequent_count < (int32_t) ce3->count) {
                most_frequent_count = ce3->count;
                most_frequent_class = ce3->key;
            } else if (most_frequent_count == (int32_t) ce3->count) {
                log_warn("labels in leaf are balanced");
            }
        }

        // record most frequent class for tree
        kal_getp(class_counter_global, most_frequent_class, ce2);
        if (ce2 == NULL) {
            kv_push(class_counter_elm, class_counter_global,
                    ((class_counter_elm) {most_frequent_class, 1}));

        } else {
            ce2->count += 1;
        }

        kv_destroy(*tree_neighs);
    }

    free(neigh_detail);
    {
        double best_class;
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
    double den = 0.0;
    double num = 0.0;
    
    nwvec = ET_forest_neighbors(forest, vector, curtail_min_size);
    for(size_t i = 0; i < kv_size(*nwvec); i++) {
        neighbour_weight nw = kv_A(*nwvec, i);
        double weight = nw.weight;
        uint32_t sample_idx = nw.key;
        log_debug("sample_idx: %d weight: %g", sample_idx, weight);
        double val = forest->labels[sample_idx];

        num += val * weight;
        den += weight;
    }

    kv_destroy(*nwvec);
    free(nwvec);

    return num / den;
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
