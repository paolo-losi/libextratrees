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
