# ifndef EXTRA_TREES_H
# define EXTRA_TREES_H

# include <stdint.h>
# include <stdio.h>
# include <stdbool.h>
# include <math.h>
# include "kvec.h"

typedef kvec_t(uint32_t) uint_vec;
typedef kvec_t(double) double_vec;
typedef kvec_t(unsigned char) uchar_vec;

// --- problem ---

typedef struct {
    float *vectors;
    double *labels;
    uint32_t n_features;
    uint32_t n_samples;
} ET_problem;


// --- params ---

typedef struct ET_params {
    uint32_t number_of_features_tested;
    uint32_t number_of_trees;
    bool regression;
    uint32_t min_split_size;
    bool select_features_with_replacement;
} ET_params;


# define EXTRA_TREE_DEFAULT_CLASSIF_PARAMS(prob, params) do {            \
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


// --- tree ---

#define ET_LEAF_NODE 'L'
#define ET_SPLIT_NODE 'S'

typedef struct ET_base_node {
    char type;
    uint32_t n_samples;
    double diversity;
} ET_base_node;

typedef struct ET_split_node {
    ET_base_node base;
    float threshold;
    uint32_t feature_id;
    ET_base_node *lower_node, *higher_node;
} ET_split_node;

typedef struct ET_leaf_node {
    ET_base_node base;
    uint_vec indexes;
} ET_leaf_node;

typedef ET_base_node *ET_tree;

// --- forest ---

typedef struct {
    kvec_t(ET_tree) trees;
    ET_params params;
    uint32_t n_features;
    uint32_t n_samples;
    double *labels;
} ET_forest;


// --- predict types ---

typedef struct {
    uint32_t key;
    double weight;
} neighbour_weight;

typedef kvec_t(neighbour_weight) neighbour_weight_vec;


// --- functions ---

ET_problem *ET_load_libsvm_file(char *fname);
ET_forest *ET_forest_build(ET_problem *prob, ET_params *params);
double_vec *ET_forest_feature_importance(ET_forest *forest);
void ET_forest_destroy(ET_forest *forest);
void ET_forest_dump(ET_forest *forest, uchar_vec *buffer);
void ET_problem_print(ET_problem *prob, FILE *f);
void ET_problem_destroy(ET_problem *prob);

double ET_forest_predict(ET_forest *forest, float *vector);
double ET_forest_predict_regression(ET_forest *forest, float *v,
                                    uint32_t curtail_min_size);
# endif 
