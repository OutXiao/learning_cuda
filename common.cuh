#define USE_DP

#ifdef USE_DP
typedef double real;
#else
typedef float real;
#endif

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
    }                                                 \
} while (0)


#define TIMEING_START                                   \
do                                                      \
{                                                       \
    cudaEvent_t start, stop;                            \
    CHECK(cudaEventCreate(&start));                     \
    CHECK(cudaEventCreate(&stop));                      \
    CHECK(cudaEventRecord(start));                      \
    cudaEventQuery(start);                              \
}while(0)                                               \


#define TIMEING_END                                     \
do                                                      \
{                                                       \
    CHECK(cudaEventRecord(stop));                       \
    CHECK(cudaEventSynchronize(stop));                  \
    float elapsed_time;                                 \
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));\
    printf("Time = %g ms.\n", elapsed_time);            \
    CHECK(cudaEventDestroy(start));                     \
    CHECK(cudaEventDestroy(stop));                      \
}while(0)                                               \
