#include "extratrees.h"
#include "train.h"
#include "log.h"


static double predict_tree(ET_tree tree, double *vector) {
    ET_base_node *node = tree;

    while(1) {
        switch(node->type) {
            case SPLIT_NODE: {
                ET_split_node *split = CAST_SPLIT(node);
                node = (vector[split->feature_id] <= split->threshold) ?
                       split->lower_node : split->higher_node;
                break;
            }    
            case LEAF_NODE: {
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

double ET_predict(ET_forest *forest, double *vector) {
    //TODO classification FIXME
    double mean = 0.0;
    uint32_t n_trees = kv_size(forest->trees);

    for(uint32_t i = 0; i < n_trees; i++) {
        ET_tree t = kv_A(forest->trees, i);
        mean += predict_tree(t, vector);
    }

    return mean / n_trees;
}
