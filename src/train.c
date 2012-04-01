#include <float.h>
#include <math.h>

#include "extratrees.h"
#include "train.h"
#include "util.h"
#include "problem.h"
#include "log.h"
#include "counter.h"


#define FOR_SAMPLE_IDX_IN(sample_idxs, body)                                \
    for (size_t i = 0; i < kv_size((sample_idxs)); i++) {                   \
        uint32_t sample_idx = kv_A((sample_idxs), i);                       \
        do { body } while(0); }


typedef struct {
    ET_base_node *node;
    uint_vec higher_idxs;
    uint_vec lower_idxs;
    double higher_diversity;
    double lower_diversity;
} builder_stack_node;


ET_leaf_node *new_leaf_node(uint_vec *sample_idxs, bool constant) {
    ET_leaf_node *ln = NULL;
    ln = malloc(sizeof(ET_leaf_node));
    check_mem(ln);

    ln->base.type = ET_LEAF_NODE;

    kv_init(ln->indexes);
    kv_copy(uint32_t, ln->indexes, *sample_idxs);
    ln->constant = constant;
    ln->base.n_samples = kv_size(*sample_idxs);

    exit:
    return ln;
}


typedef struct {
    double min, max;
} min_max;


min_max get_feature_min_max(ET_problem *prob, uint_vec *sample_idxs,
                            uint32_t fid) {

    min_max mm = {DBL_MAX, -DBL_MAX};

    FOR_SAMPLE_IDX_IN(*sample_idxs, {
        float val = PROB_GET(prob, sample_idx, fid);
        if (val > mm.max) mm.max = val;
        if (val < mm.min) mm.min = val;
    });

    return mm;
}


void split_on_threshold(ET_problem *prob, uint32_t feature_idx,
                                          double threshold,
                                          uint_vec *sample_idxs,
                                          uint_vec *higher_idxs,
                                          uint_vec *lower_idxs) {

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


double classification_diversity(ET_problem *prob, uint_vec *sample_idxs) {
    double n_samples = kv_size(*sample_idxs);
    double gini_diversity = 0.0;
    ET_class_counter class_counter;

    ET_class_counter_init(class_counter);


    FOR_SAMPLE_IDX_IN(*sample_idxs, {
        double label = prob->labels[sample_idx];
        ET_class_counter_incr(&class_counter, label);
    });

    log_debug("class counter:");
    for(size_t i = 0; i < kv_size(class_counter); i++) {
        double class;
        uint32_t count;
        class = kv_A(class_counter, i).key;
        count = kv_A(class_counter, i).count;

        log_debug("    > class: %g count:%d", class, count);

        gini_diversity += count * (1.0 - count / n_samples);
    }
    log_debug("gini index: %g", gini_diversity / n_samples);

    ET_class_counter_destroy(class_counter);
    return gini_diversity;
}


double regression_diversity(ET_problem *prob, uint_vec *sample_idxs) {

    double mean = 0;
    uint32_t count = 0;
    double diversity = 0;

    FOR_SAMPLE_IDX_IN(*sample_idxs, {
        double label = prob->labels[sample_idx];
        mean += label;
        count++;
    });
    mean /= count;

    FOR_SAMPLE_IDX_IN(*sample_idxs, {
        double label = prob->labels[sample_idx];
        diversity += pow(label - mean,  2);
    })
    return diversity;
}


void split_problem(tree_builder *tb, uint_vec *sample_idxs,
                   builder_stack_node *stack_node) {

    bool labels_are_constant = true;
    ET_base_node *node = NULL;
    bool split_found = false;
    double best_threshold = 0;       // initialized to silence compiler warn
    uint32_t best_feature_idx = 0.0; // initialized to silence compiler warn
    ET_problem *prob = tb->prob;

    double higher_diversity, lower_diversity;
    uint_vec lower_idxs, higher_idxs;
    kv_init(lower_idxs); kv_init(higher_idxs);

    log_debug(">>>>> split_problem. n samples: %zu", kv_size(*sample_idxs));

    // check if min_split_size is reached
    // NOTE: this does not guarantee that leaf size is always >= min_split_size
    if(kv_size(*sample_idxs) < (size_t) tb->params.min_split_size) {
        log_debug("min_split_size (%d) NOT reached. sample size: %zu",
                                                    tb->params.min_split_size,
                                                    kv_size(*sample_idxs));
        node = (ET_base_node *) new_leaf_node(sample_idxs, false);
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
        node = (ET_base_node *) new_leaf_node(sample_idxs, true);
        goto exit;
    }

    {
        double best_diversity = DBL_MAX;
        uint32_t nb_features_tested = 0;
        uint32_t nb_features_to_test = tb->params.number_of_features_tested;
        bool with_replacement = tb->params.select_features_with_replacement;

        log_debug("number of features to test: %d", nb_features_to_test);

        // select best split
        while (true) {
            min_max mm;
            uint32_t feature_idx;
            double threshold, diversity;

            log_debug("--- new loop cycle ---");

            // select random feature
            if (with_replacement) {
                feature_idx = random_int(&tb->rand_state, prob->n_features);
            } else {
                uint32_t deck_idx, end_idx, *deck;

                deck = tb->features_deck;
                deck_idx = random_int(&tb->rand_state,
                                      prob->n_features - nb_features_tested);
                feature_idx = deck[deck_idx];
                end_idx = prob->n_features - nb_features_tested - 1;
                deck[deck_idx] = deck[end_idx];
                deck[end_idx] = feature_idx;
            }
            nb_features_tested++;
            log_debug("number of feature selected %s replacement: %d",
                    with_replacement ? "WITH" : "WITHOUT", nb_features_tested);
            log_debug("feature index: %d", feature_idx);

            // select random threshold in (min, max)
            mm = get_feature_min_max(prob, sample_idxs, feature_idx);
            log_debug("values - min: %g max: %g", mm.min, mm.max);
            if (mm.min == mm.max) {
                log_debug("constant feature");
                continue;
            } else split_found = true;

            double delta = mm.max - mm.min;
            threshold = mm.min + random_double(&tb->rand_state) * delta;

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
                stack_node->higher_diversity = higher_diversity;
                stack_node->lower_diversity = lower_diversity;
            }

            if (diversity == 0) {
                log_debug("diversity == 0");
                break;
            }

            nb_features_to_test--;

        if (nb_features_to_test <= 0) { break; }
        if (with_replacement) {
            if (nb_features_tested >= 10 * prob->n_features) { break; }
        } else {
            if (nb_features_tested >= prob->n_features) { break; }
        }

        }
    }

    if (split_found) {
        // let's build a split node ...
        log_debug("split found. feature_idx: %d, threshold: %g",                                                                best_feature_idx,
                                                best_threshold);
        ET_split_node *sn;

        sn = malloc(sizeof(ET_split_node));
        check_mem(sn);
        sn->base.type = ET_SPLIT_NODE;
        sn->base.n_samples = kv_size(*sample_idxs);
        sn->feature_id = best_feature_idx;
        sn->threshold = best_threshold;
        sn->lower_node = NULL;
        sn->higher_node = NULL;
        node = (ET_base_node *) sn;
    } else {
        log_debug("split NOT found. building leaf node ...");
        node = (ET_base_node *) new_leaf_node(sample_idxs, false);
    }

    exit:
    kv_destroy(lower_idxs);
    kv_destroy(higher_idxs);
    stack_node->node = node;
}


//TODO remove recursion
void tree_destroy(ET_base_node *node) {
    ET_split_node *sn = NULL;
    ET_leaf_node  *ln = NULL;

    switch (node->type) {
        case ET_LEAF_NODE:
            ln = (ET_leaf_node *) node;
            kv_destroy(ln->indexes);
            free(node);
            break;
        case ET_SPLIT_NODE:
            sn = (ET_split_node *) node;
            tree_destroy((ET_base_node *) sn->higher_node);
            tree_destroy((ET_base_node *) sn->lower_node);
            free(node);
            break;
        default:
            sentinel("unexpected split node type")
    }

    exit:
    return;
}


int tree_builder_init(tree_builder *tb, ET_problem *prob,
                      ET_params *params, uint32_t *seed) {
    tb->prob = prob;
    simplerandom_kiss2_seed(&tb->rand_state, seed[2], seed[3],
                                             seed[1], seed[0]);

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


ET_tree build_tree(tree_builder *tb) {
    ET_tree tree = NULL;
    kvec_t(builder_stack_node) stack;
    builder_stack_node *curr_snode;

    // general initialization
    kv_init(stack);
    kv_resize(builder_stack_node, stack, tb->prob->n_samples);

    {
        double diversity = -1;
        uint_vec sample_idxs;
        kv_init(sample_idxs);
        kv_range(uint32_t, sample_idxs, tb->prob->n_samples);

        // stack initialization
        curr_snode = ( kv_pushp(builder_stack_node, stack) );
        kv_init(curr_snode->higher_idxs);
        kv_init(curr_snode->lower_idxs);

        diversity = tb->diversity_f(tb->prob, &sample_idxs);
        log_debug("node diversity for next split: %g", diversity);
        split_problem(tb, &sample_idxs, curr_snode);
        check_mem(curr_snode->node);
        curr_snode->node->diversity = diversity;
        kv_destroy(sample_idxs);
    }

    while (kv_size(stack) > 0) {
        bool link_to_parent_required = false;
        uint_vec *curr_sample_idxs = NULL;
        double curr_diversity = -1;
        curr_snode = &kv_last(stack);

        if (IS_SPLIT(curr_snode->node)) {
            ET_split_node *sn = CAST_SPLIT(curr_snode->node);

            if (sn->higher_node == NULL) {
                curr_sample_idxs = &curr_snode->higher_idxs;
                curr_diversity = curr_snode->higher_diversity;
            } else if (sn->lower_node == NULL) {
                curr_sample_idxs = &curr_snode->lower_idxs;
                curr_diversity = curr_snode->lower_diversity;
            } else {
                link_to_parent_required = true;
            }
        } else {
            // node is a ET_LEAF_NODE
            link_to_parent_required = true;
        }

        if (link_to_parent_required) {
            ET_split_node *sn = NULL;

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
            log_debug("node diversity for next split: %g", curr_diversity);
            split_problem(tb, curr_sample_idxs, curr_snode);
            check_mem(curr_snode->node);
            curr_snode->node->diversity = curr_diversity;
        }
    }

    exit:
    kv_destroy(stack);
    return tree;
}


ET_forest *ET_forest_build(ET_problem *prob, ET_params *params) {
    ET_forest *forest = NULL;
    ET_tree tree = NULL;
    tree_builder tb;
    // random seed obtained from mersenne twister invocation
    uint32_t seed[4] = {3346013320, 826458053, 1844335739, 274945865};

    forest = malloc(sizeof(ET_forest));
    check_mem(forest);
    forest->params = *params;
    forest->n_samples  = prob->n_samples;
    forest->n_features = prob->n_features;
    forest->labels = malloc(prob->n_samples * sizeof(double));
    forest->class_frequency = NULL;
    check_mem(forest->labels);
    memcpy(forest->labels, prob->labels, prob->n_samples * sizeof(double));

    kv_init(forest->trees);
    check_mem(! tree_builder_init(&tb, prob, params, seed) );

    for(uint32_t i = 0; i < params->number_of_trees; i++) {
        log_debug("***** building tree # %d *****", i);
        tree = build_tree(&tb);
        check_mem(tree);
        kv_push(ET_tree, forest->trees, tree);
    }

    exit:
    tree_builder_destroy(&tb);
    return forest;
}


void ET_forest_destroy(ET_forest *forest) {
    for(uint32_t i = 0; i < kv_size(forest->trees); i++) {
        ET_tree t = kv_A(forest->trees, i);
        tree_destroy(t);
    }
    kv_destroy(forest->trees);
    free(forest->labels);
    if(forest->class_frequency) {
        ET_class_counter_destroy(*forest->class_frequency);
        free(forest->class_frequency);
    }
}
