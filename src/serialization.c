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

/*
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
        double val = load_double(bufferp);
        kv_push(double, *dv, val);
    }
}
*/

static void dump_uint_vec(uint_vec *uiv, uchar_vec *buffer) {
    uint32_t size = (uint32_t) kv_size(*uiv);

    dump_uint32(size, buffer);
    for(uint32_t i = 0; i < size; i++) {
        dump_uint32(kv_A(*uiv, i), buffer);
    }
}

static void load_uint_vec(uint_vec *uiv, unsigned char **bufferp) {
    uint32_t size = load_uint32(bufferp);
    //TODO verify the state of dv;
    for(uint32_t i = 0; i < size; i++) {
        double val = load_uint32(bufferp);
        kv_push(uint32_t, *uiv, val);
    }
}

static void dump_data(void *data, size_t n, uchar_vec *buffer) {
    size_t buffer_size = kv_size(*buffer);
    kv_resize(unsigned char, *buffer, buffer_size + n);
    memcpy(&kv_A(*buffer, buffer_size), data, n);
    kv_size(*buffer) = buffer_size + n;
}

static void load_data(void *data, size_t n, unsigned char **bufferp) {
    memcpy(data, *bufferp, n);
    *bufferp += n;
}


// --- dump / load node ---

static void node_dump(ET_base_node *node, uchar_vec *buffer) {
    dump_char(node->type, buffer);
    dump_uint32(node->n_samples, buffer);
    dump_double(node->diversity, buffer);
    switch(node->type) {
        case ET_LEAF_NODE: {
            ET_leaf_node *ln = CAST_LEAF(node);
            dump_char((char) ln->constant, buffer);
            dump_uint_vec(&ln->indexes, buffer);
            break;
        }
        case ET_SPLIT_NODE: {
            ET_split_node *sn = CAST_SPLIT(node);
            dump_uint32(sn->feature_id, buffer);
            dump_float(sn->threshold, buffer);
            break;
        }
    }
}

static ET_base_node *node_load(unsigned char **bufferp) {
    char type;
    double diversity;
    uint32_t n_samples;
    ET_base_node *node = NULL;

    type = load_char(bufferp);
    n_samples = load_uint32(bufferp);
    diversity = load_double(bufferp);

    switch(type) {
        case ET_LEAF_NODE: {
            ET_leaf_node *ln = malloc(sizeof(ET_leaf_node));
            check_mem(ln);
            ln->constant = (bool) load_char(bufferp);
            kv_init(ln->indexes);
            load_uint_vec(&ln->indexes, bufferp);
            node = (ET_base_node *) ln;
            break;
        }

        case ET_SPLIT_NODE: {
            ET_split_node *sn = malloc(sizeof(ET_split_node));
            check_mem(sn);
            sn->feature_id = load_uint32(bufferp);
            sn->threshold = load_float(bufferp);
            sn->higher_node = NULL;
            sn->lower_node = NULL;
            node = (ET_base_node *) sn;
            break;
        }

        default:
            sentinel("unexpected node type: %x", type);
    }
    node->type = type;
    node->n_samples = n_samples;
    node->diversity = diversity;

    exit:
    return node;
    
}


// --- dump tree ---


void ET_tree_dump(ET_tree tree, uchar_vec *buffer) {
    tree_navigate(tree, (node_processor) node_dump, buffer);
}


// --- load tree ---

ET_tree ET_tree_load(unsigned char **bufferp) {
    kvec_t(ET_base_node *) stack;
    ET_base_node *curr_node = NULL, *child_node = NULL;

    kv_init(stack);

    curr_node = node_load(bufferp);
    check_mem(curr_node);
    kv_push(ET_base_node *, stack, curr_node);

    while (kv_size(stack)) {
        curr_node = kv_last(stack);

        if (child_node) {
            ET_split_node *sn = CAST_SPLIT(curr_node);
            if (!sn->higher_node) {
                sn->higher_node = child_node;
            } else {
                sn->lower_node = child_node;
            }
            child_node = NULL;
        }

        if(IS_LEAF(curr_node) || (CAST_SPLIT(curr_node)->higher_node != NULL &&
                                  CAST_SPLIT(curr_node)->lower_node != NULL)) {
            child_node = kv_pop(stack);
        } else {
            curr_node = node_load(bufferp);
            check_mem(curr_node);
            kv_push(ET_base_node *, stack, curr_node);
        }
    }

    exit:
    kv_destroy(stack);
    return child_node;
}

// --- dump / load forest ---


//FIXME handle endianess
void ET_forest_dump(ET_forest *forest, uchar_vec *buffer) {
    uint32_t size = (uint32_t) kv_size(forest->trees);

    dump_data(&forest->params, sizeof(ET_params), buffer);
    dump_double(forest->n_features, buffer);
    dump_double(forest->n_samples, buffer);
    
    for(uint32_t i = 0; i < forest->n_samples; i++) {
        dump_double(forest->labels[i], buffer);
    }

    dump_uint32(size, buffer);
    
    for(uint32_t i = 0; i < size; i++) {
        ET_tree_dump(kv_A(forest->trees, i), buffer);
    }
}

ET_forest *ET_forest_load(unsigned char **bufferp) {
    uint32_t n_trees;
    ET_forest *forest = NULL;

    forest = malloc(sizeof(ET_forest));
    check_mem(forest);
    kv_init(forest->trees);

    load_data(&forest->params, sizeof(ET_params), bufferp);
    forest->n_features = load_double(bufferp);
    forest->n_samples = load_double(bufferp);

    forest->labels = malloc(sizeof(double) * forest->n_samples);
    check_mem(forest->labels);

    forest->class_frequency = NULL;

    for(size_t i = 0; i < forest->n_samples; i++) {
        forest->labels[i] = load_double(bufferp);
    }

    n_trees = load_uint32(bufferp);
    for(uint32_t i = 0; i < n_trees; i++) {
        kv_push(ET_tree, forest->trees, ET_tree_load(bufferp));
    }

    exit:
    return forest;

}
