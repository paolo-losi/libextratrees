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
    float vector3[3] = {2.1, 1, 1};
    double prediction;
    double *neighbor_weights;
    class_probability_vec *cpv;

    problem_init(&prob, vectors, labels);
    ET_problem_print(&prob, stderr);

    EXTRA_TREE_DEFAULT_REGR_PARAMS(prob, params);
    params.number_of_trees = 100;
    params.number_of_features_tested = 2;
    params.select_features_with_replacement = false;

    forest = ET_forest_build(&prob, &params);

    prediction = ET_forest_predict(forest, vector1);
    fprintf(stderr, "reg prediction vector1: %g\n", prediction);

    prediction = ET_forest_predict(forest, vector2);
    fprintf(stderr, "reg prediction vector2: %g\n", prediction);

    prediction = ET_forest_predict_regression(forest, vector1, 5);
    fprintf(stderr, "reg prediction vector1 (curtail=5): %g\n", prediction);

    prediction = ET_forest_predict_regression(forest, vector2, 5);
    fprintf(stderr, "reg prediction vector2 (curtail=5): %g\n", prediction);

    prediction = ET_forest_predict_quantile(forest, vector1, .5, 2);
    fprintf(stderr, "reg quantile vector1 (curtail=2): %g\n", prediction);

    prediction = ET_forest_predict_quantile(forest, vector2, .5, 2);
    fprintf(stderr, "reg quantile vector2 (curtail=2): %g\n", prediction);

    prediction = ET_forest_predict_class_majority(forest, vector2, 1);
    fprintf(stderr, "class prediction vector2: %g\n", prediction);

    prediction = ET_forest_predict_class_majority(forest, vector3, 4);
    fprintf(stderr, "class prediction vector3 (curtail=4): %g\n", prediction);

    for(int smooth = 0; smooth <= 1; smooth++) {
        cpv = ET_forest_predict_probability(forest, vector3, 0, smooth);

        fprintf(stderr, "class probability vector3. smooth: %d\n", smooth);
        for(size_t i = 0; i < kv_size(*cpv); i++) {
            class_probability *cp = &kv_A(*cpv, i);
            fprintf(stderr, "    class %g -> %g\n", cp->label, cp->probability);
        }

        kv_destroy(*cpv);
        free(cpv);
    }

    prediction = ET_forest_predict_class_bayes(forest, vector3, 1, false);
    fprintf(stderr, "class prediction vector3 (bayes): %g\n", prediction);

    neighbor_weights = ET_forest_neighbors(forest, vector3, 1);
    fprintf(stderr, "neighbor weights for vector3:\n");
    for(size_t i = 0; i < forest->n_samples; i++) {
        fprintf(stderr, "  - sample_idx: %zd. weight: %g\n",
                i, neighbor_weights[i]);
    }

    ET_forest_destroy(forest);
    free(forest);
    free(neighbor_weights);
}


int main() {
    test_predict();
    return 0;
}
