#include "extratrees.h"
#include "test.h"


float vectors[] = { 2, 2, 2, 3, 3, 3, 4, 4, 4 ,
                    2, 3, 2, 3, 2, 3, 4, 5, 4 ,
                    4, 4, 4, 4, 4, 4, 4, 4, 4 ,
                    1, 2, 3, 1, 2, 3, 1, 2, 3 };
double labels[] = { 2, 2, 2, 1, 1, 1, 0, 0, 0 };


void test_importance_regression() {
    test_header();

    ET_problem prob;
    ET_params params;
    ET_forest *forest;
    double *feature_importance;

    problem_init(&prob, vectors, labels);
    ET_problem_print(&prob, stderr);

    EXTRA_TREE_DEFAULT_REGR_PARAMS(prob, params);
    params.number_of_trees = 100;
    params.number_of_features_tested = 2;
    params.select_features_with_replacement = false;

    forest = ET_forest_build(&prob, &params);
    feature_importance = ET_forest_feature_importance(forest, 1);
    for(uint32_t i = 0; i < forest->n_features; i++) {
        fprintf(stderr, "feature #%d -> importance: %g\n",
            i, feature_importance[i]);
    }

    ET_forest_destroy(forest);
    free(feature_importance);
    free(forest);
}


void test_importance_classification() {
    test_header();

    ET_problem prob;
    ET_params params;
    ET_forest *forest;
    double *feature_importance;

    problem_init(&prob, vectors, labels);
    ET_problem_print(&prob, stderr);

    EXTRA_TREE_DEFAULT_CLASSIF_PARAMS(prob, params);
    params.number_of_trees = 100;
    params.number_of_features_tested = 2;
    params.select_features_with_replacement = false;

    forest = ET_forest_build(&prob, &params);
    feature_importance = ET_forest_feature_importance(forest, 1);
    for(uint32_t i = 0; i < forest->n_features; i++) {
        fprintf(stderr, "feature #%d -> importance: %g\n",
            i, feature_importance[i]);
    }

    ET_forest_destroy(forest);
    free(feature_importance);
    free(forest);
}


int main() {
    test_importance_regression();
    test_importance_classification();
    return 0;
}
