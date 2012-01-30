#include "extratrees.h"
#include "train.h"
#include "test.h"
#include "log.h"
#include "serialization.c"


float big_vectors[] = { 1., 3., 2., 3., 0.,
                        4., 1., 6., 2., 1.,
                        1., 1., 1., 1., 1.,
                        4., 6., 5., 7., 5.,
                        8., 6., 3., 3., 1.,
                        2., 2., 2., 3., 2. };
double big_labels[] = { 2., 2., 1., 3., 1. };


void test_double_float() {
    double d = 2.2354e-10;
    float f = d;
    uint32_t i = 7821334;
    uchar_vec b;
    unsigned char *buffer = NULL;

    kv_init(b);
    dump_double(d, &b);
    buffer = b.a;
    fprintf(stderr, "%g == %g - ", d, load_double(&buffer));
    fprintf(stderr, "buffer used: %zu\n", buffer - b.a);

    kv_clear(b);
    dump_float(f, &b);
    buffer = b.a;
    fprintf(stderr, "%g == %g - ", d, load_float(&buffer));
    fprintf(stderr, "buffer used: %zu\n", buffer - b.a);

    kv_clear(b);
    dump_uint32(i, &b);
    buffer = b.a;
    fprintf(stderr, "%d == %d - ", i, load_uint32(&buffer));
    fprintf(stderr, "buffer used: %zu\n", buffer - b.a);
}


void test_forest_serialization() {
    test_header();

    ET_problem prob;
    ET_params params;
    ET_forest *forest;
    uchar_vec buffer;

    kv_init(buffer);
    problem_init(&prob, big_vectors, big_labels);

    EXTRA_TREE_DEFAULT_REGR_PARAMS(prob, params);
    params.number_of_trees = 100;
    params.number_of_features_tested = 1;
    params.select_features_with_replacement = true;

    forest = build_forest(&prob, &params);
    ET_forest_dump(forest, &buffer);
    fprintf(stderr, "forest dump: %zu bytes\n", kv_size(buffer));

    kv_destroy(buffer);
    ET_forest_destroy(forest);
    free(forest);
}


int main() {
    test_forest_serialization();
    test_double_float();
    return 0;
}
