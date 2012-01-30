#include "extratrees.h"
#include "util.h"
#include "log.h"


// --- utils ---

// compile time check for double and float storage
enum{ foobar_d = 1/(sizeof(uint64_t) == sizeof(double)) };
enum{ foobar_f = 1/(sizeof(uint32_t) == sizeof(float)) };


static inline void dump_char(char c, uchar_vec *buffer) {
    kv_push(unsigned char, *buffer, (unsigned char) c);
}

static inline char load_char(unsigned char **bufferp) {
    unsigned char *buffer = *bufferp;
    char c = (char) *buffer;
    *bufferp += 1;
    return c;
}

static void dump_float(float f, uchar_vec *buffer) {
    const unsigned char *s = (unsigned char *) &f;
    for(int i = 0; i < 4; i++) {
        kv_push(unsigned char, *buffer, *(s + i));
    }
}

static float load_float(unsigned char **bufferp) {
    unsigned char *buffer = *bufferp;
    float f;
    memcpy(&f, buffer, 4);
    *bufferp += 4;
    return f;
}

static void dump_double(double d, uchar_vec *buffer) {
    const unsigned char *s = (unsigned char *) &d;
    for(int i = 0; i < 8; i++) {
        kv_push(unsigned char, *buffer, *(s + i));
    }
}

static double load_double(unsigned char **bufferp) {
    unsigned char *buffer = *bufferp;
    double d;
    memcpy(&d, buffer, 8);
    *bufferp += 8;
    return d;
}

static void dump_uint32(uint32_t ivalue, uchar_vec *buffer) {
    for(int i = 0; i < 4; i++) {
        kv_push(unsigned char, *buffer, ivalue >> 8 * i);
    }
}

static uint32_t load_uint32(unsigned char **bufferp) {
    unsigned char *buffer = *bufferp;
    uint32_t i;
    memcpy(&i, buffer, 4);
    *bufferp += 4;
    return i;
}

static void dump_double_vec(double_vec *dv, uchar_vec *buffer) {
    uint32_t size = (uint32_t) kv_size(*dv);

    dump_uint32(size, buffer);
    for(uint32_t i = 0; i < size; i++) {
        dump_double(kv_A(*dv, i), buffer);
    }
}

static void load_double_vec(double_vec *dv, unsigned char **bufferp) {
    uint32_t size = load_uint32(bufferp);
    //TODO verify the state of dv;
    for(uint32_t i = 0; i < size; i++) {
        kv_push(double, *dv, load_double(bufferp));
    }
}


// --- dump ---

static void node_dump(ET_base_node *node, uchar_vec *buffer) {
    dump_char(node->type, buffer);
    switch(node->type) {
        case ET_LEAF_NODE: {
            ET_leaf_node *ln = CAST_LEAF(node);
            dump_double_vec(&ln->labels, buffer);
        }
        case ET_SPLIT_NODE: {
            ET_split_node *sn = CAST_SPLIT(node);
            dump_uint32(sn->feature_id, buffer);
            dump_float(sn->threshold, buffer);
        }
    }
}


// can we make tree navigation generic?

typedef struct {
    ET_split_node *node;
    bool higher_visited;
    bool lower_visited;
} dump_stack_node;


static dump_stack_node *dump_stack_node_new(ET_base_node *node) {
        dump_stack_node *stack_node = NULL;
        stack_node = malloc(sizeof(dump_stack_node));
        check_mem(stack_node);
        stack_node->node = CAST_SPLIT(node);
        stack_node->higher_visited = false;
        stack_node->lower_visited = false;
        exit:
        return stack_node;
}


int ET_tree_dump(ET_tree tree, uchar_vec *buffer) {
    kvec_t(dump_stack_node *) stack;
    dump_stack_node *stack_node;

    kv_init(stack);

    node_dump(tree, buffer);
    if(IS_SPLIT(tree)) {
        stack_node = dump_stack_node_new(tree);
        check_mem(stack_node);
        kv_push(dump_stack_node *, stack, stack_node);
    }

    while (kv_size(stack)) {
        stack_node = kv_last(stack);
        ET_base_node *next_node = NULL;
        if (!stack_node->higher_visited) {
            next_node = stack_node->node->higher_node;
            stack_node->higher_visited = true;
        } else if (!stack_node->lower_visited) {
            next_node = stack_node->node->lower_node;
            stack_node->lower_visited = true;
        }

        if (next_node) {
            node_dump(next_node, buffer);
            if(IS_SPLIT(next_node)) {
                dump_stack_node *next_stack_node = NULL;
                next_stack_node = dump_stack_node_new(next_node);
                check_mem(next_stack_node);
                kv_push(dump_stack_node *, stack, next_stack_node);
            }
        } else {
            UNUSED(kv_pop(stack));
            free(stack_node);
        }
    }
    kv_destroy(stack);
    return 0;

    exit:
    kv_destroy(stack);
    return -1;
}


int ET_forest_dump(ET_forest *forest, uchar_vec *buffer) {
    uint32_t size = (uint32_t) kv_size(forest->trees);

    //FIXME dump parameters
    //FIXME handle endianess
    dump_uint32(size, buffer);
    for(uint32_t i = 0; i < size; i++) {
        int ret = ET_tree_dump(kv_A(forest->trees, i), buffer);
        check_mem(!ret);
    }
    return 0;
    exit:
    return -1;
}
