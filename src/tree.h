#include "simplerandom.h"
#include "kvec.h"


// --- tree ---

#define LEAF_NODE 'L'
#define SPLIT_NODE 'S'

typedef struct rt_base_node {
    char type;
} rt_base_node;

typedef struct rt_split_node {
    rt_base_node base;
    uint32_t feature_id;
    double feature_val;
    rt_base_node *lower_node, *higher_node;
} rt_split_node;

typedef struct rt_leaf_node {
    rt_base_node base;
    kvec_t(double) labels;
    //TODO kvec_t(int) sample_idx;
} rt_leaf_node;


// --- params ---

typedef struct rt_params {
    uint32_t number_of_features_tested;
    uint32_t number_of_trees;
    int regression;
    uint32_t min_split_size;
    int select_features_with_replacement;
} rt_params;


// --- builder ---

typedef struct tree_builder {
    rt_problem *prob;
    rt_params params;
    SimpleRandomKISS2_t rand_state;
    uint32_t *features_deck;
    //TODO the following vals should help optimize constant feature detection
    //double *max_vals;
    //double *min_vals;
} tree_builder;


# define EXTRA_TREE_DEFAULT_CLASS_PARAMS(tb) do {                              \
    (tb).params.number_of_features_tested = ceil(sqrt((tb).prob->n_features)); \
    (tb).params.number_of_trees           = 100;                               \
    (tb).params.regression                = 0;                                 \
    (tb).params.min_split_size            = 1;                                 \
    (tb).params.select_features_with_replacement = 0;                          \
    } while(0)

# define EXTRA_TREE_DEFAULT_REGR_PARAMS(tb) do {                               \
    (tb).params.number_of_features_tested = (tb).prob->n_features;             \
    (tb).params.number_of_trees           = 100;                               \
    (tb).params.regression                = 1;                                 \
    (tb).params.min_split_size            = 1;                                 \
    (tb).params.select_features_with_replacement = 0;                          \
    } while(0)


// --- utils ---

typedef struct {
    double min, max;
} min_max;


rt_base_node *split_problem(tree_builder *tb, int_vec *sample_idxs);
int tree_builder_init(tree_builder *tb, rt_problem *prob);
void tree_builder_destroy(tree_builder *tb);
void tree_destroy(rt_base_node *bn);
