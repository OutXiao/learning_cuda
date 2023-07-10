#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.cuh"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

static int N = 10000;
const int NUM_REPEATS = 10;
const real x0 = 100.0;

__global__ void gpu_arithmetic(real* d_x, const real x0, const int N);
cudaError test_gpu_arithmetic();

void cpu_arithmetic(real* d_x, const real x0, const int N);
cudaError test_cpu_arithmetic();
