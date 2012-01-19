#include "stdio.h"

#include "rt.h"
#include "load.h"
#include "test.h"

void load_simple_file() {
    test_header();
    rt_problem *prob;
    prob = rt_load_libsvm_file("test1.libsvm");
    rt_print_problem(prob);
}

void load_unexistent_file() {
    test_header();
    rt_problem *prob;
    prob = rt_load_libsvm_file("foo");
}

int main() {
    load_simple_file();
    load_unexistent_file();
    return 0;
}
