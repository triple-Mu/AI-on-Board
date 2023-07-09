//
// Created by ubuntu on 23-5-31.
//

#ifndef _CHECKER_H_
#define _CHECKER_H_

#ifndef NDEBUG
#define CHECK(call)                                                                                                    \
    do {                                                                                                               \
        const auto ret = call;                                                                                         \
        if (!ret) {                                                                                                    \
            printf("********** Error occurred ! **********\n");                                                        \
            printf("***** File:      %s\n", __FILE__);                                                                 \
            printf("***** Line:      %d\n", __LINE__);                                                                 \
            printf("***** Error:     %s\n", #call);                                                                    \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)
#else
#define CHECK(call)                                                                                                    \
    {                                                                                                                  \
        const auto ret = call;                                                                                         \
    }
#endif
#endif  //_CHECKER_H_
