#include "common.cuh"
#include "arithmetic.cuh"
#include <iostream>
using namespace std;

int main(void)
{

    test_gpu_arithmetic();
    cout << "----------" << endl;
    test_cpu_arithmetic();
    return 0;
}


