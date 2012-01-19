#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <stdio.h>
#include <errno.h>
#include <string.h>

#if defined(DEBUG)
#define log_debug(M, ...) \
  fprintf(stderr,"[DEBUG] %15s:%-3d " M "\n",__FILE__,__LINE__, ##__VA_ARGS__)
#else
#define log_debug(M, ...)
#endif

#if defined(INFO)
#define log_info(M, ...) \
  fprintf(stderr,"[INFO]  %15s:%-3d " M "\n",__FILE__,__LINE__, ##__VA_ARGS__)
#else
#define log_info(M, ...)
#endif

#define clean_errno() (errno == 0 ? "None" : strerror(errno))

#define log_error(M, ...) \
  fprintf(stderr,"[ERROR] %15s:%-3d errno=%s. " M "\n", __FILE__, __LINE__, \
          clean_errno(), ##__VA_ARGS__)

#define log_warn(M, ...) \
  fprintf(stderr,"[WARN]  %15s:%-3d errno=%s. " M "\n", __FILE__, __LINE__, \
          clean_errno(), ##__VA_ARGS__)

#define check(A, M, ...)       if(!(A)) { \
                                    log_error(M, ##__VA_ARGS__); errno=0; \
                                    goto exit; }

#define check_debug(A, M, ...) if(!(A)) { \
                                    log_debug(M, ##__VA_ARGS__); errno=0; \
                                    goto exit; }

#define check_mem(A) check((A), "Out of memory.") 

#define sentinel(M, ...)  { log_error(M, ##__VA_ARGS__); errno=0; goto exit; }

#endif
