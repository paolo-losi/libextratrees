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

typedef rt_base_node rt_tree;


#define IS_LEAF(n)  ((n)->type == LEAF_NODE)
#define IS_SPLIT(n) ((n)->type == SPLIT_NODE)

#define CAST_LEAF(n)  ((rt_leaf_node *)  (n))
#define CAST_SPLIT(n) ((rt_split_node *) (n))


// --- params ---

typedef struct rt_params {
    uint32_t number_of_features_tested;
    uint32_t number_of_trees;
    bool regression;
    uint32_t min_split_size;
    bool select_features_with_replacement;
} rt_params;


// --- builder ---

typedef double (*diversity_function) (rt_problem *prob, uint_vec *sample_idxs);

typedef struct tree_builder {
    rt_problem *prob;
    rt_params params;
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



rt_base_node *build_tree(rt_problem *prob, rt_params *params);
int tree_builder_init(tree_builder *tb, rt_problem *prob, rt_params *params);
void tree_builder_destroy(tree_builder *tb);
void tree_destroy(rt_base_node *bn);

