#include "rt.h"
#include "test.h"


void test_leaf() {
    test_header();

    rt_problem prob;
    double vectors[] = { 1., 3.,
                         2., 4.,
                         1., 6.};
    double labels[]  = { 2., 3., 4. };

    init_problem(&prob, vectors, labels);
    rt_print_problem(&prob);
}


int main() {
    test_leaf();
    return 0;
}
