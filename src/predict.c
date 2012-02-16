#include "extratrees.h"
#include "util.h"
#include "log.h"


static double predict_tree(ET_tree tree, float *vector) {
    ET_base_node *node = tree;

    while(1) {
        switch(node->type) {
            case ET_SPLIT_NODE: {
                ET_split_node *split = CAST_SPLIT(node);
                node = (vector[split->feature_id] <= split->threshold) ?
                       split->lower_node : split->higher_node;
                break;
            }    
            case ET_LEAF_NODE: {
                double_vec *labels = &CAST_LEAF(node)->labels;

                //TODO classification FIXME
                double mean = 0.0;
                for(uint32_t i = 0; i < kv_size(*labels); i++) 
                    mean += kv_A(*labels, i);

                mean /= kv_size(*labels);
                return mean;
            }
            default:
                sentinel("unexpected split type %c", node->type);
        }
    }

    exit:
    //sentinel
    return 0.0;
}

double ET_forest_predict(ET_forest *forest, float *vector) {
    //TODO classification FIXME
    double mean = 0.0;
    uint32_t n_trees = kv_size(forest->trees);

    for(uint32_t i = 0; i < n_trees; i++) {
        ET_tree t = kv_A(forest->trees, i);
        mean += predict_tree(t, vector);
    }

    return mean / n_trees;
}

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


