#include <float.h>
#include <math.h>

#include "rt.h"
#include "tree.h"
#include "util.h"
#include "problem.h"
#include "log.h"


rt_leaf_node *new_leaf_node(double_vec *labels) {
    rt_leaf_node *ln = NULL;
    ln = malloc(sizeof(rt_leaf_node));
    check_mem(ln);
    ln->base.type = LEAF_NODE;
    kv_init(ln->labels);
    kv_copy(double, ln->labels, *labels);

    exit:
    return ln;
}


min_max get_feature_min_max(rt_problem *prob, int_vec *sample_idxs,
                            uint32_t fid) {

    min_max mm = {DBL_MAX, -DBL_MAX};

    for (size_t s = 0; s < kv_size(*sample_idxs); s++) {
        int val = PROB_GET(prob, s, fid);
        if (val > mm.max) mm.max = val;
        if (val < mm.min) mm.min = val;
    }

    return mm;
}


double classification_diversity(rt_problem *prob, int_vec *sample_idxs,
                                uint32_t feature_idx, double threshold) {
    // FIXME implement it!
    return 0.0;
}


double regression_diversity(rt_problem *prob, int_vec *sample_idxs,
                            uint32_t feature_idx, double threshold) {

    double lowers_mean = 0, highers_mean = 0;
    uint32_t lowers_count = 0, highers_count = 0;
    double diversity = 0;
    size_t sample_size = kv_size(*sample_idxs);

    for(size_t s = 0; s < sample_size; s++) {
        double feat_val = PROB_GET(prob, s, feature_idx);
        double label = prob->labels[s];
        if (feat_val <= threshold) {
            lowers_mean += label;
            lowers_count++;
        } else {
            highers_mean += label;
            highers_count++;
        }
    }
    lowers_mean /= lowers_count;
    highers_mean /= highers_count;

    for(size_t s = 0; s < sample_size; s++) {
        double feat_val = PROB_GET(prob, s, feature_idx);
        double label = prob->labels[s];
        diversity += (feat_val <= threshold) ? pow(label - lowers_mean,  2) :
                                               pow(label - highers_mean, 2) ;
    }
    return diversity;
}


rt_base_node *split_problem(tree_builder *tb, int_vec *sample_idxs) {

    double_vec labels;
    int labels_are_constant = 1;
    rt_base_node *node;
    int split_found = 0;
    double best_threshold;
    uint32_t best_feature_idx;
    rt_problem *prob = tb->prob;

    log_debug(">>> split_problem. n samples: %zu", kv_size(*sample_idxs));

    // TODO is that really necessary to calculate labels upfront?

    // initialize labels vector
    kv_init(labels);
    kv_resize(double, labels, kv_size(*sample_idxs));
    kv_size(labels) = kv_size(*sample_idxs);
    for(size_t i=0; i < kv_size(*sample_idxs); i++) {
        kv_A(labels, i) = prob->labels[kv_A(*sample_idxs, i)];
    }

    // TODO this does not guarantee that leaf size is always >= min_split_size
    // check if min_split_size is reached
    if(kv_size(*sample_idxs) < (size_t) tb->params.min_split_size) {
        node = (rt_base_node *) new_leaf_node(&labels);
        goto exit;
    }

    // check if labels are constant
    {
        double first_label = kv_A(labels, 0);
        for(size_t i=1; i < kv_size(labels); i++) {
            if(first_label != kv_A(labels, i)) {
                labels_are_constant = 0;
                break;
            }
        }
    }

    // if labels are constant return leaf node
    if(labels_are_constant) {
        log_debug("labels are constant. generating leaf node ...");
        node = (rt_base_node *) new_leaf_node(&labels);
        goto exit;
        
    }

    {
        double best_diversity = DBL_MAX;
        uint32_t nb_features_tested = 0;
        uint32_t nb_features_to_test = tb->params.number_of_features_tested;
        int with_replacement = tb->params.select_features_with_replacement;
        
        // select best split
        do {
            min_max mm;
            uint32_t feature_idx;
            double threshold, diversity;

            // select random feature
            if (with_replacement) {
                feature_idx = random_int(&tb->rand_state, prob->n_features);
            } else {
                uint32_t deck_idx, end_idx, *deck;

                deck = tb->features_deck;
                deck_idx = random_int(&tb->rand_state,
                                      prob->n_features-nb_features_tested);
                feature_idx = deck[deck_idx];
                end_idx = prob->n_features - nb_features_tested - 1;
                deck[deck_idx] = deck[end_idx];
                deck[end_idx] = deck_idx;
                nb_features_tested++;
            }

            // select random threshold in (min, max)
            mm = get_feature_min_max(prob, sample_idxs, feature_idx);
            if (mm.min == mm.max) {
                // FIXME are the following two lines correct?
                // the alternative would be counting all non-constant features
                // and selecting a percentage (e.g 50%, 200%) of non-constant
                // features when params.select_features_with_replacement == 1.
                if (with_replacement)
                    nb_features_to_test--; 
                continue;
            } else split_found = 1;
            do {
                double delta = mm.max - mm.min;
                threshold = mm.min + random_double(&tb->rand_state) * delta;
            } while (threshold == mm.min);

            // evaluate split diversity
            diversity = (tb->params.regression) ? 
                regression_diversity(prob, sample_idxs,
                                     feature_idx, threshold) : 
                classification_diversity(prob, sample_idxs,
                                         feature_idx, threshold);

            if (diversity < best_diversity) {
                best_threshold = threshold;
                best_feature_idx = feature_idx;
            }

            if (diversity == 0)
                break;

            nb_features_to_test--;

        } while (nb_features_to_test && 
                                    (with_replacement ||
                                     nb_features_tested < prob->n_features));
    }

    if (split_found) {
        // let's build a split node ...
        rt_base_node *higher_node, *lower_node;
        rt_split_node *sn;
        int_vec lower_idxs, higher_idxs;
        kv_init(lower_idxs); kv_init(higher_idxs);

        // divide samples on threshold
        for(size_t i = 0; i < kv_size(*sample_idxs); i++) {
            double val;
            uint32_t sample_idx = kv_A(*sample_idxs, i);
            val = PROB_GET(prob, sample_idx, best_feature_idx);
            if (val <= best_threshold) 
                kv_push(int, lower_idxs, sample_idx);
            else
                kv_push(int, higher_idxs, sample_idx);
        }

        // recursively calculate sub nodes
        higher_node = split_problem(tb, &higher_idxs);
        lower_node  = split_problem(tb, &lower_idxs);
        if (higher_node == NULL || lower_node == NULL) return NULL;

        sn = malloc(sizeof(rt_split_node));
        check_mem(sn);
        sn->base.type = SPLIT_NODE;
        sn->feature_id = best_feature_idx;
        sn->feature_val = best_threshold;
        sn->lower_node = lower_node;
        sn->higher_node = higher_node;
        node = (rt_base_node *) sn;

        kv_destroy(lower_idxs);
        kv_destroy(higher_idxs);
    } else {
        node = (rt_base_node *) new_leaf_node(&labels);
    }

    exit:
    kv_destroy(labels);
    return node;

}
