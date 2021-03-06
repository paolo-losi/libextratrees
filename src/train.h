#ifndef ET_TRAIN_H
#define ET_TRAIN_H

#include "simplerandom.h"
#include "kvec.h"


// --- builder ---

typedef double (*diversity_function) (ET_problem *prob, uint_vec *sample_idxs);

typedef struct tree_builder {
    ET_problem *prob;
    ET_params params;
    SimpleRandomKISS2_t rand_state;
    uint32_t *features_deck;
    diversity_function diversity_f;
} tree_builder;


// --- utils ---

ET_tree build_tree(tree_builder *tb);
int tree_builder_init(tree_builder *tb, ET_problem *prob, ET_params *params,
                      uint32_t *seed);
void tree_builder_destroy(tree_builder *tb);
void tree_destroy(ET_base_node *bn);

#endif
