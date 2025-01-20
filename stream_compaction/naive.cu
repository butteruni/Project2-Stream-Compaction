#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int offset, int* dataA, int* dataB) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;
			if (index >= offset) {
				dataB[index] = dataA[index - offset] + dataA[index];
			}
			else {
				dataB[index] = dataA[index];
			}
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			int* dev_dataA;
            int* dev_dataB;
			cudaMalloc((void**)&dev_dataA, n * sizeof(int));
			cudaMalloc((void**)&dev_dataB, n * sizeof(int));
			cudaMemcpy(dev_dataA, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            // TODO
			for (int d = 1; d <= ilog2ceil(n); d++) {
				int offset = 1 << (d - 1);
				kernNaiveScan << <fullBlocksPerGrid, blockSize >> > (n, offset, dev_dataA, dev_dataB);
				std::swap(dev_dataA, dev_dataB);
			}
			odata[0] = 0;
			cudaMemcpy(odata + 1, dev_dataA, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            timer().endGpuTimer();
			cudaFree(dev_dataA);
			cudaFree(dev_dataB);
        }
    }
}
