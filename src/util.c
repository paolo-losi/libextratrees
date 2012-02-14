#include <stdint.h>
#include "simplerandom.h"
#include "util.h"
#include "log.h"

// pick a random integer between [0, max_val)
uint32_t random_int(SimpleRandomKISS2_t *rnd, uint32_t max_val) {
    uint32_t i;
    do {
        i = simplerandom_kiss2_next(rnd);
    } while (i >= UINT32_MAX / max_val * max_val);
    return i % max_val;
}

// pick a random double between [0., 1.)
double random_double(SimpleRandomKISS2_t *rnd) {
    uint32_t i = simplerandom_kiss2_next(rnd);
    return i * 2.3283064365386963e-10;
}

void visit_stack_node_init(visit_stack_node *stack_node, ET_base_node *node) {
        stack_node->node = CAST_SPLIT(node);
        stack_node->higher_visited = false;
        stack_node->lower_visited = false;
}

void tree_navigate(ET_tree tree, void (*f) (ET_base_node* node, void *data),
                                 void *data) {
    kvec_t(visit_stack_node) stack;
    visit_stack_node *stack_node;

    kv_init(stack);

    f(tree, data);
    if(IS_SPLIT(tree)) {
        stack_node = (kv_pushp(visit_stack_node, stack));
        visit_stack_node_init(stack_node, tree);
    }

    while (kv_size(stack)) {
        stack_node = &kv_last(stack);
        ET_base_node *next_node = NULL;
        if (!stack_node->higher_visited) {
            next_node = stack_node->node->higher_node;
            stack_node->higher_visited = true;
        } else if (!stack_node->lower_visited) {
            next_node = stack_node->node->lower_node;
            stack_node->lower_visited = true;
        }

        if (next_node) {
            f(next_node, data);
            if(IS_SPLIT(next_node)) {
                visit_stack_node *next_stack_node;
                next_stack_node = (kv_pushp(visit_stack_node, stack));
                visit_stack_node_init(next_stack_node, next_node);
            }
        } else {
            UNUSED(kv_pop(stack));
        }
    }
    kv_destroy(stack);
}
