#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include "rt.h"
#include "load.h"
#include "log.h"
#include "util.h"


char *new_err_str(char *msg, int line_count, char *line) {
    char *error_str;

    line[20] = '\0';
    error_str = malloc(1024);
    snprintf(error_str, 1024,
             "parsing error on line %d: %s. line=%s...",
             line_count, msg, line);
    return error_str;
}


int parse_line(parse_state *ps, char *line, int line_count) {
    int feat_id, n;
    double val;

    if (sscanf(line, "%lf%n", &val, &n) != 1) {
        ps->on_error(ps, new_err_str("label not found", line_count, line));
        return -1;
    }

    line += n;
    ps->on_new_label(ps, val);

    while(sscanf(line, "%d : %lf%n", &feat_id, &val, &n) == 2) {
        line += n;
        ps->on_new_feature(ps, feat_id, val);
    }

    while(isspace(*line)) line++;

    if (*line != '\0') {
        ps->on_error(ps, new_err_str("feature not found", line_count, line));
        return -1;
    }

    return 0;
}


void parse_libsvm_file(parse_state *ps, FILE *f, int bufsize) {
    char *buffer, *bp;
    int line_count = 0;

    rewind(f);
    bp = buffer = malloc(bufsize);
    check_mem(buffer);

    while(1) {
        int c = fgetc(f);
        if (c == '\n' || c == EOF) {
            if (bp != buffer) {
                line_count += 1;
                *bp = '\0';
                if (parse_line(ps, buffer, line_count)) break;
                bp = buffer;
            }
            if (c == EOF) break;
        } else {
            if ((bp - buffer) >= bufsize) {
                size_t newbufsize = bufsize * 4;
                buffer = realloc(buffer, newbufsize);
                check_mem(buffer);
                bp = buffer + bufsize;
                bufsize = newbufsize;
            }
            *bp++ = c;
        }
    }
    free(buffer);
    return;

    exit:
    if (buffer) free(buffer);
    ps->on_error(ps, "malloc failed");
    return;
}


// ----- size_parser -----

typedef struct size_parser {
    parse_state ops;
    char *error_msg;
    int n_samples, n_features;
} size_parser;

void size_parser_on_error(parse_state *ps, char *error_msg) {
    size_parser *sp = (size_parser *) ps;
    sp->error_msg = error_msg;
}

void size_parser_on_new_label(parse_state *ps, double val) {
    UNUSED(val);
    size_parser *sp = (size_parser *) ps;
    sp->n_samples += 1;
}

void size_parser_on_new_feature(parse_state *ps, int fid, double val) {
    UNUSED(val);
    size_parser *sp = (size_parser *) ps;
    if (fid > sp->n_features) 
        sp->n_features = fid;
}

void size_parser_init(size_parser *sp) {
    sp->ops.on_error = size_parser_on_error;
    sp->ops.on_new_label = size_parser_on_new_label;
    sp->ops.on_new_feature = size_parser_on_new_feature;
    sp->n_features = 0;
    sp->n_samples = 0;
    sp->error_msg = NULL;
}

void size_parser_destroy(size_parser *sp) {
    if(sp->error_msg) free(sp->error_msg);
}


// ----- load_parser -----

typedef struct load_parser {
    parse_state ops;
    int error_flag;
    float *vectors;
    double *labels;
    int sample_idx;
    int n_samples;
    int n_features;
} load_parser;

void load_parser_on_error(parse_state *ps, char *error_msg) {
    UNUSED(error_msg);
    load_parser *lp = (load_parser *) ps;
    lp->error_flag = 1;
}

void load_parser_on_new_label(parse_state *ps, double val) {
    load_parser *lp = (load_parser *) ps;
    lp->sample_idx += 1;
    lp->labels[lp->sample_idx] = val;
}

void load_parser_on_new_feature(parse_state *ps, int fid, double val) {
    load_parser *lp = (load_parser *) ps;
    lp->vectors[lp->sample_idx + lp->n_samples * (fid-1)] = val;
}

int load_parser_init(load_parser *lp, int n_features, int n_samples) {
    float *vectors;
    double *labels;

    lp->ops.on_error = load_parser_on_error;
    lp->ops.on_new_label = load_parser_on_new_label;
    lp->ops.on_new_feature = load_parser_on_new_feature;
    lp->error_flag = 0;
    lp->sample_idx = -1;
    lp->n_samples = n_samples;
    lp->n_features = n_features;
    
    vectors = calloc(n_samples * n_features, sizeof(float));
    check_mem(vectors);
    labels = calloc(n_samples, sizeof(double));
    check_mem(labels);

    lp->vectors = vectors;
    lp->labels = labels;
    return 0;

    exit:
    lp->error_flag = 1;
    return -1;
}

void load_parser_destroy(load_parser* lp) {
    if(lp->error_flag) {
        if(lp->vectors) free(lp->vectors);
        if(lp->labels)  free(lp->labels);
    }
}


rt_problem *rt_load_libsvm_file(char *fname) {
    FILE *f;
    size_parser *sp = NULL;
    load_parser *lp = NULL;
    rt_problem *prob = NULL;

    f = fopen(fname, "r");
    check(f, "Could not open %s.", fname);

    sp = calloc(1, sizeof(size_parser));
    check_mem(sp);
    size_parser_init(sp);
    parse_libsvm_file((parse_state *) sp, f, PARSE_BUF_SIZE);
    check(!sp->error_msg, "%s", sp->error_msg);

    lp = calloc(1, sizeof(load_parser));
    check_mem(lp);
    check(! load_parser_init(lp, sp->n_features, sp->n_samples),
          "load_parser could not allocate enough memory.");

    parse_libsvm_file((parse_state *) lp, f, PARSE_BUF_SIZE);
    check(! lp->error_flag, "Unexpected failure in loading phase.");

    prob = malloc(sizeof(rt_problem));
    check_mem(prob);

    prob->vectors = lp->vectors;
    prob->labels  = lp->labels;
    prob->n_features = sp->n_features;
    prob->n_samples  = sp->n_samples;

    exit:
    if (sp) {
        size_parser_destroy(sp);
        free(sp);
    }
    if (lp) {
        load_parser_destroy(lp);
        free(lp);
    }
    if(f) fclose(f);
    return prob;
}
