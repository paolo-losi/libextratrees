#ifndef RT_LOAD_H
#define RT_LOAD_H

#define PARSE_BUF_SIZE 4096 * 1024


typedef struct parse_state {
    void (*on_new_label) (struct parse_state *state,  double);
    void (*on_new_feature) (struct parse_state *state, int, double);
    void (*on_error) (struct parse_state *state, char *);
} parse_state;


void parse_libsvm_file(parse_state *ps, FILE *f, int bufsize); 

#endif
