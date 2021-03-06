#ifndef ET_UTIL_H
#define ET_UTIL_H

#include "simplerandom.h"
#include "extratrees.h"

uint32_t random_int(SimpleRandomKISS2_t *rnd, uint32_t max_val); 
double random_double(SimpleRandomKISS2_t *rnd);

#define IS_LEAF(n)  ((n)->type == ET_LEAF_NODE)
#define IS_SPLIT(n) ((n)->type == ET_SPLIT_NODE)

#define CAST_LEAF(n)  ((ET_leaf_node *)  (n))
#define CAST_SPLIT(n) ((ET_split_node *) (n))

#define UNUSED(x) (void)(x)

#ifdef TEST
#define STATIC
#else
#define STATIC static
#endif


typedef struct {
    ET_split_node *node;
    bool higher_visited;
    bool lower_visited;
} visit_stack_node;

typedef void (*node_processor)(struct ET_base_node *node, void *data);

void visit_stack_node_init(visit_stack_node *stack_node, ET_base_node *node);
void tree_navigate(ET_tree tree, node_processor f, void *data);

#endif
