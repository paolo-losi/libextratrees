#include <float.h>
#include <math.h>

#include "rt.h"
#include "tree.h"
#include "util.h"
#include "problem.h"
#include "log.h"


#define FOR_SAMPLE_IDX_IN(sample_idxs, body)                                \
    for (size_t i = 0; i < kv_size((sample_idxs)); i++) {                   \
        int sample_idx = kv_A((sample_idxs), i);                            \
        do { body } while(0); }                                             


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


typedef struct {
    double min, max;
} min_max;


min_max get_feature_min_max(rt_problem *prob, int_vec *sample_idxs,
                            uint32_t fid) {

    min_max mm = {DBL_MAX, -DBL_MAX};

    FOR_SAMPLE_IDX_IN(*sample_idxs, {
        int val = PROB_GET(prob, sample_idx, fid);
        if (val > mm.max) mm.max = val;
        if (val < mm.min) mm.min = val;
    });

    return mm;
}


void split_on_threshold(rt_problem *prob, uint32_t feature_idx,
                                          double threshold,
                                          int_vec *sample_idxs, 
                                          int_vec *higher_idxs,
                                          int_vec *lower_idxs) {

    kv_clear(*higher_idxs);
    kv_clear(*lower_idxs);

    FOR_SAMPLE_IDX_IN(*sample_idxs, {
        double val = PROB_GET(prob, sample_idx, feature_idx);
        if (val <= threshold) {
            log_debug("sample_idx: %d, val: %g -> lower", sample_idx, val);
            kv_push(int, *lower_idxs, sample_idx);
        } else {
            log_debug("sample_idx: %d, val: %g -> higher", sample_idx, val);
            kv_push(int, *higher_idxs, sample_idx);
        }
    });
}


double classification_diversity(rt_problem *prob, int_vec *sample_idxs) {
    // FIXME implement it!
    return 0.0;
}


double regression_diversity(rt_problem *prob, int_vec *sample_idxs) {

    double mean = 0;
    uint32_t count = 0;
    double diversity = 0;

    FOR_SAMPLE_IDX_IN(*sample_idxs, {
        double label = prob->labels[sample_idx];
        mean += label;
        count++;
    });
    mean  /= count;

    FOR_SAMPLE_IDX_IN(*sample_idxs, {
        double label = prob->labels[sample_idx];
        diversity += pow(label - mean,  2);
    })
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

    double higher_diversity, lower_diversity;
    int_vec lower_idxs, higher_idxs, best_lower_idxs, best_higher_idxs;
    kv_init(lower_idxs);      kv_init(higher_idxs);
    kv_init(best_lower_idxs); kv_init(best_higher_idxs);

    // TODO #typedef diversity_f
    double (*diversity_f)(rt_problem *prob, int_vec *sample_idxs) = 
                    (tb->params.regression) ? regression_diversity :
                                              classification_diversity;

    log_debug(">>>>> split_problem. n samples: %zu", kv_size(*sample_idxs));

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
    if(kv_size(*sample_idxs) <= (size_t) tb->params.min_split_size) {
        log_debug("min size (%d) reached. current sample size: %zu", 
                                                    tb->params.min_split_size,
                                                    kv_size(*sample_idxs));
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
        
        log_debug("number of features to test: %d", nb_features_to_test);

        // select best split
        do {
            min_max mm;
            uint32_t feature_idx;
            double threshold, diversity;

            log_debug("--- new loop cycle ---");

            // select random feature
            if (with_replacement) {
                feature_idx = random_int(&tb->rand_state, prob->n_features);
                log_debug("selected feature WITH replacement");
            } else {
                uint32_t deck_idx, end_idx, *deck;

                deck = tb->features_deck;
                deck_idx = random_int(&tb->rand_state,
                                      prob->n_features-nb_features_tested);
                feature_idx = deck[deck_idx];
                end_idx = prob->n_features - nb_features_tested - 1;
                deck[deck_idx] = deck[end_idx];
                deck[end_idx] = feature_idx;
                nb_features_tested++;
                log_debug("selected feature WITHOUT replacement: %d",
                                                          nb_features_tested);
            }
            log_debug("feature index: %d", feature_idx);

            // select random threshold in (min, max)
            mm = get_feature_min_max(prob, sample_idxs, feature_idx);
            log_debug("values - min: %g max: %g", mm.min, mm.max);
            if (mm.min == mm.max) {
                // FIXME are the following two lines correct?
                // the alternative would be counting all non-constant features
                // and selecting a percentage (e.g 50%, 200%) of non-constant
                // features when params.select_features_with_replacement == 1.
                if (with_replacement)
                    nb_features_to_test--;
                log_debug("constant feature");
                continue;
            } else split_found = 1;
            do {
                double delta = mm.max - mm.min;
                threshold = mm.min + random_double(&tb->rand_state) * delta;
            } while (threshold == mm.min);

            log_debug("threshold: %g", threshold);

            // evaluate split diversity
            split_on_threshold(prob, feature_idx, threshold,
                               sample_idxs, &higher_idxs, &lower_idxs);
            higher_diversity = diversity_f(prob, &higher_idxs);
            lower_diversity  = diversity_f(prob, &lower_idxs);

            diversity = higher_diversity + lower_diversity;

            log_debug("%s diversity: %g", tb->params.regression?"regr":"class",
                                          diversity);
                                          
                                                                
            if (diversity < best_diversity) {
                log_debug("diversity is new best");
                best_threshold = threshold;
                best_diversity = diversity;
                best_feature_idx = feature_idx;
                kv_copy(int, best_higher_idxs, higher_idxs);
                kv_copy(int, best_lower_idxs,  lower_idxs);
            }

            if (diversity == 0) {
                log_debug("diversity == 0");
                break;
            }

            nb_features_to_test--;

        } while (nb_features_to_test && 
                                    (with_replacement ||
                                     nb_features_tested < prob->n_features));
    }

    if (split_found) {
        // let's build a split node ...
        log_debug("split found. feature_idx: %d, threshold: %g",                                                                best_feature_idx,
                                                best_threshold);
        rt_base_node *higher_node, *lower_node;
        rt_split_node *sn;

        // recursively calculate sub nodes
        higher_node = split_problem(tb, &best_higher_idxs);
        lower_node  = split_problem(tb, &best_lower_idxs);
        if (higher_node == NULL || lower_node == NULL) return NULL;

        sn = malloc(sizeof(rt_split_node));
        check_mem(sn);
        sn->base.type = SPLIT_NODE;
        sn->feature_id = best_feature_idx;
        sn->feature_val = best_threshold;
        sn->lower_node = lower_node;
        sn->higher_node = higher_node;
        node = (rt_base_node *) sn;
    } else {
        node = (rt_base_node *) new_leaf_node(&labels);
    }

    exit:
    kv_destroy(lower_idxs);
    kv_destroy(higher_idxs);
    kv_destroy(best_lower_idxs);
    kv_destroy(best_higher_idxs);
    kv_destroy(labels);
    return node;
}


//TODO remove recursion
void tree_destroy(rt_base_node *node) {
    rt_split_node *sn = NULL;
    rt_leaf_node  *ln = NULL;

    switch (node->type) {
        case LEAF_NODE:
            ln = (rt_leaf_node *) node;
            kv_destroy(ln->labels);
            free(node);
            break;
        case SPLIT_NODE:
            sn = (rt_split_node *) node;
            tree_destroy((rt_base_node *) sn->higher_node);
            tree_destroy((rt_base_node *) sn->lower_node);
            free(node);
            break;
        default:
            sentinel("unexpected split node type")
    }

    exit:
    return;
}


int tree_builder_init(tree_builder *tb, rt_problem *prob) {
    tb->prob = prob;
    simplerandom_kiss2_seed(&tb->rand_state, 0, 0, 0, 0);

    tb->features_deck = NULL;
    tb->features_deck = malloc(sizeof(uint32_t) * prob->n_features);
    check_mem(tb->features_deck);
    for(uint32_t i = 0; i < prob->n_features; i++) {
        tb->features_deck[i] = i;
    }

    // default parameters
    EXTRA_TREE_DEFAULT_CLASS_PARAMS(*tb);
    return 0;

    exit:
    return -1;
}


void tree_builder_destroy(tree_builder *tb) {
    if (tb->features_deck) free(tb->features_deck);
}
