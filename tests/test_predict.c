#include "extratrees.h"
#include "test.h"


float vectors[] = { 2, 2, 2, 3, 3, 3, 4, 4, 4 ,
                    4, 4, 4, 4, 4, 4, 4, 4, 4 ,
                    1, 2, 3, 1, 2, 3, 1, 2, 3 };
double labels[] = { 2, 2, 2, 1, 1, 1, 0, 0, 0 };


void test_predict() {
    test_header();

    ET_problem prob;
    ET_params params;
    ET_forest *forest;
    double vector1[3] = {3, 4, 4};
    double vector2[3] = {2, 1, 1};
    double prediction1, prediction2;

    problem_init(&prob, vectors, labels);
    ET_print_problem(stderr, &prob);

    EXTRA_TREE_DEFAULT_REGR_PARAMS(prob, params);
    params.number_of_trees = 100;
    params.number_of_features_tested = 2;
    params.select_features_with_replacement = false;

    forest = build_forest(&prob, &params);

    prediction1 = ET_predict(forest, vector1);
    prediction2 = ET_predict(forest, vector2);
    fprintf(stderr, "prediction vector1: %g\n", prediction1);
    fprintf(stderr, "prediction vector2: %g\n", prediction2);

    ET_forest_destroy(forest);
    free(forest);
}


int main() {
    test_predict();
    return 0;
}