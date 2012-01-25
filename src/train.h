#ifndef ET_TRAIN_H
#define ET_TRAIN_H

#include "simplerandom.h"
#include "kvec.h"


// --- params ---

typedef struct ET_params {
    uint32_t number_of_features_tested;
    uint32_t number_of_trees;
    bool regression;
    uint32_t min_split_size;
    bool select_features_with_replacement;
} ET_params;


// --- tree ---

#define LEAF_NODE 'L'
#define SPLIT_NODE 'S'

typedef struct ET_base_node {
    char type;
} ET_base_node;

typedef struct ET_split_node {
    ET_base_node base;
    uint32_t feature_id;
    double feature_val;
    ET_base_node *lower_node, *higher_node;
} ET_split_node;

typedef struct ET_leaf_node {
    ET_base_node base;
    kvec_t(double) labels;
    //TODO kvec_t(int) sample_idx;
} ET_leaf_node;

typedef ET_base_node *ET_tree;


#define IS_LEAF(n)  ((n)->type == LEAF_NODE)
#define IS_SPLIT(n) ((n)->type == SPLIT_NODE)

#define CAST_LEAF(n)  ((ET_leaf_node *)  (n))
#define CAST_SPLIT(n) ((ET_split_node *) (n))


// --- forest ---

typedef struct {
    kvec_t(ET_tree) trees;
    ET_params params;
} ET_forest;


// --- builder ---

typedef double (*diversity_function) (ET_problem *prob, uint_vec *sample_idxs);

typedef struct tree_builder {
    ET_problem *prob;
    ET_params params;
    SimpleRandomKISS2_t rand_state;
    uint32_t *features_deck;
    diversity_function diversity_f;
} tree_builder;


# define EXTRA_TREE_DEFAULT_CLASS_PARAMS(prob, params) do {              \
    (params).number_of_features_tested = ceil(sqrt((prob).n_features));  \
    (params).number_of_trees           = 100;                            \
    (params).regression                = false;                          \
    (params).min_split_size            = 1;                              \
    (params).select_features_with_replacement = false;                   \
    } while(0)

# define EXTRA_TREE_DEFAULT_REGR_PARAMS(prob, params) do {               \
    (params).number_of_features_tested = (prob).n_features;              \
    (params).number_of_trees           = 100;                            \
    (params).regression                = true;                           \
    (params).min_split_size            = 1;                              \
    (params).select_features_with_replacement = false;                   \
    } while(0)


// --- utils ---



ET_tree build_tree(tree_builder *tb);
ET_forest *build_forest(ET_problem *prob, ET_params *params);
int tree_builder_init(tree_builder *tb, ET_problem *prob, ET_params *params,
                      uint32_t *seed);
void tree_builder_destroy(tree_builder *tb);
void tree_destroy(ET_base_node *bn);

#endif
