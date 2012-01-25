#include "stdio.h"

#include "extratrees.h"
#include "load.h"
#include "test.h"

void load_simple_file() {
    test_header();
    ET_problem *prob;
    prob = ET_load_libsvm_file("test1.libsvm");
    ET_print_problem(stderr, prob);
}

void load_unexistent_file() {
    test_header();
    ET_problem *prob;
    prob = ET_load_libsvm_file("foo");
}

int main() {
    load_simple_file();
    load_unexistent_file();
    return 0;
}
