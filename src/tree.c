#include <float.h>
#include <math.h>

#include "rt.h"
#include "tree.h"
#include "util.h"
#include "problem.h"
#include "log.h"


#define FOR_SAMPLE_IDX_IN(sample_idxs, body)                                \
    for (size_t i = 0; i < kv_size((sample_idxs)); i++) {                   \
        uint32_t sample_idx = kv_A((sample_idxs), i);                       \
        do { body } while(0); }                                             


typedef struct {
    rt_base_node *node;
    int_vec higher_idxs;
    int_vec lower_idxs;
} builder_stack_node;


rt_leaf_node *new_leaf_node(rt_problem *prob, int_vec *sample_idxs) {
    rt_leaf_node *ln = NULL;
    ln = malloc(sizeof(rt_leaf_node));
    check_mem(ln);

    ln->base.type = LEAF_NODE;

    kv_init(ln->labels);
    FOR_SAMPLE_IDX_IN(*sample_idxs, {
        double label = prob->labels[sample_idx];
        kv_push(double, ln->labels, label);
    });

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
        float val = PROB_GET(prob, sample_idx, fid);
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
            kv_push(uint32_t, *lower_idxs, sample_idx);
        } else {
            log_debug("sample_idx: %d, val: %g -> higher", sample_idx, val);
            kv_push(uint32_t, *higher_idxs, sample_idx);
        }
    });
}


double classification_diversity(rt_problem *prob, int_vec *sample_idxs) {
    // FIXME implement it!
    UNUSED(prob);
    UNUSED(sample_idxs);
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


void split_problem(tree_builder *tb, int_vec *sample_idxs,
                   builder_stack_node *stack_node) {

    bool labels_are_constant = true;
    rt_base_node *node = NULL;
    bool split_found = false;
    double best_threshold = 0;       // initialized to silence compiler warn
    uint32_t best_feature_idx = 0.0; // initialized to silence compiler warn 
    rt_problem *prob = tb->prob;

    double higher_diversity, lower_diversity;
    int_vec lower_idxs, higher_idxs;
    kv_init(lower_idxs); kv_init(higher_idxs);

    log_debug(">>>>> split_problem. n samples: %zu", kv_size(*sample_idxs));

    // NOTE: this does not guarantee that leaf size is always >= min_split_size
    // check if min_split_size is reached
    if(kv_size(*sample_idxs) <= (size_t) tb->params.min_split_size) {
        log_debug("min size (%d) reached. current sample size: %zu", 
                                                    tb->params.min_split_size,
                                                    kv_size(*sample_idxs));
        node = (rt_base_node *) new_leaf_node(prob, sample_idxs);
        goto exit;
    }

    // check if labels are constant
    {
        double first_label = 0;
        FOR_SAMPLE_IDX_IN(*sample_idxs, {
            double label = prob->labels[sample_idx];
            if (i == 0) {
                first_label = label;
            } else if (first_label != label) {
                labels_are_constant = false;
                break;
            }
        });
    }

    // if labels are constant return leaf node
    if(labels_are_constant) {
        log_debug("labels are constant. generating leaf node ...");
        node = (rt_base_node *) new_leaf_node(prob, sample_idxs);
        goto exit;
    }

    {
        double best_diversity = DBL_MAX;
        uint32_t nb_features_tested = 0;
        uint32_t nb_features_to_test = tb->params.number_of_features_tested;
        bool with_replacement = tb->params.select_features_with_replacement;
        
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
            } else split_found = true;
            do {
                double delta = mm.max - mm.min;
                threshold = mm.min + random_double(&tb->rand_state) * delta;
            } while (threshold == mm.min);

            log_debug("threshold: %g", threshold);

            // evaluate split diversity
            split_on_threshold(prob, feature_idx, threshold,
                               sample_idxs, &higher_idxs, &lower_idxs);
            higher_diversity = tb->diversity_f(prob, &higher_idxs);
            lower_diversity  = tb->diversity_f(prob, &lower_idxs);

            diversity = higher_diversity + lower_diversity;

            log_debug("%s diversity: %g", tb->params.regression?"regr":"class",
                                          diversity);
                                          
                                                                
            if (diversity < best_diversity) {
                log_debug("diversity is new best");
                best_threshold = threshold;
                best_feature_idx = feature_idx;
                best_diversity = diversity;
                kv_copy(uint32_t, stack_node->higher_idxs, higher_idxs);
                kv_copy(uint32_t, stack_node->lower_idxs,  lower_idxs);
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
        rt_split_node *sn;

        sn = malloc(sizeof(rt_split_node));
        check_mem(sn);
        sn->base.type = SPLIT_NODE;
        sn->feature_id = best_feature_idx;
        sn->feature_val = best_threshold;
        sn->lower_node = NULL;
        sn->higher_node = NULL;
        node = (rt_base_node *) sn;
    } else {
        node = (rt_base_node *) new_leaf_node(prob, sample_idxs);
    }

    exit:
    kv_destroy(lower_idxs);
    kv_destroy(higher_idxs);
    stack_node->node = node;
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


int tree_builder_init(tree_builder *tb, rt_problem *prob, rt_params *params) {
    tb->prob = prob;
    simplerandom_kiss2_seed(&tb->rand_state, 0, 0, 0, 0);

    tb->features_deck = NULL;
    tb->features_deck = malloc(sizeof(uint32_t) * prob->n_features);
    check_mem(tb->features_deck);
    for(uint32_t i = 0; i < prob->n_features; i++) {
        tb->features_deck[i] = i;
    }

    tb->params = *params;
    tb->diversity_f = (tb->params.regression) ? regression_diversity :
                                                classification_diversity;

    return 0;

    exit:
    return -1;
}


void tree_builder_destroy(tree_builder *tb) {
    if (tb->features_deck) free(tb->features_deck);
}


rt_tree *build_tree(rt_problem *prob, rt_params *params) {
    rt_tree *tree = NULL;
    tree_builder tb;
    builder_stack_node *curr_snode;
    kvec_t(builder_stack_node) stack;

    // general initialization
    kv_init(stack);
    kv_resize(builder_stack_node, stack, prob->n_samples);
    check_mem(! tree_builder_init(&tb, prob, params) );

    {
        int_vec sample_idxs;
        kv_init(sample_idxs);
        kv_range(uint32_t, sample_idxs, prob->n_samples);

        // stack initialization
        curr_snode = ( kv_pushp(builder_stack_node, stack) );
        kv_init(curr_snode->higher_idxs);
        kv_init(curr_snode->lower_idxs);
        split_problem(&tb, &sample_idxs, curr_snode);
        check_mem(curr_snode->node);
        kv_destroy(sample_idxs);
    }

    while (kv_size(stack) > 0) {
        bool link_to_parent_required = false;
        int_vec *curr_sample_idxs = NULL;
        curr_snode = &kv_last(stack);

        if (IS_SPLIT(curr_snode->node)) {
            rt_split_node *sn = CAST_SPLIT(curr_snode->node);

            if (sn->higher_node == NULL) {
                curr_sample_idxs = &curr_snode->higher_idxs;
            } else if (sn->lower_node == NULL) {
                curr_sample_idxs = &curr_snode->lower_idxs;
            } else {
                link_to_parent_required = true;
            }
        } else {
            // node is a LEAF_NODE
            link_to_parent_required = true;
        }

        if (link_to_parent_required) {
            rt_split_node *sn = NULL;

            UNUSED(kv_pop(stack));
            if (kv_size(stack) == 0) {
                kv_destroy(curr_snode->higher_idxs);
                kv_destroy(curr_snode->lower_idxs);
                tree = curr_snode->node;
                break;
            }

            check(IS_SPLIT(kv_last(stack).node), "unexpected NON leaf node");
            sn = CAST_SPLIT(kv_last(stack).node);

            if (sn->higher_node == NULL) {
                sn->higher_node = curr_snode->node;
            } else if (sn->lower_node == NULL) {
                sn->lower_node  = curr_snode->node;
            } else {
                sentinel("unexpected split node state in stack");
            }
            kv_destroy(curr_snode->higher_idxs);
            kv_destroy(curr_snode->lower_idxs);

        } else {
            curr_snode = ( kv_pushp(builder_stack_node, stack) );
            kv_init(curr_snode->higher_idxs);
            kv_init(curr_snode->lower_idxs);
            split_problem(&tb, curr_sample_idxs, curr_snode);
            check_mem(curr_snode->node);
        }
    }

    exit:
    tree_builder_destroy(&tb);
    kv_destroy(stack);
    return tree;
}
