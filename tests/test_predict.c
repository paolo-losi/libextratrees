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
    float vector1[3] = {3, 4, 4};
    float vector2[3] = {2, 1, 1};
    double prediction1, prediction2, prediction3, prediction4;

    problem_init(&prob, vectors, labels);
    ET_problem_print(&prob, stderr);

    EXTRA_TREE_DEFAULT_REGR_PARAMS(prob, params);
    params.number_of_trees = 100;
    params.number_of_features_tested = 2;
    params.select_features_with_replacement = false;

    forest = ET_forest_build(&prob, &params);

    prediction1 = ET_forest_predict(forest, vector1);
    fprintf(stderr, "prediction vector1: %g\n", prediction1);

    prediction2 = ET_forest_predict(forest, vector2);
    fprintf(stderr, "prediction vector2: %g\n", prediction2);

    prediction3 = ET_forest_predict_regression(forest, vector1, 5);
    fprintf(stderr, "prediction vector1 (curtail=5): %g\n", prediction3);

    prediction4 = ET_forest_predict_regression(forest, vector2, 5);
    fprintf(stderr, "prediction vector2 (curtail=5): %g\n", prediction4);

    ET_forest_destroy(forest);
    free(forest);
}


int main() {
    test_predict();
    return 0;
}
