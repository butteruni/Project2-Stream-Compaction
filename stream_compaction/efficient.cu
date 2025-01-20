#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
		__global__ void kernUpSweep(int n, int d, int* data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) return;
			int offset = 1 << (d + 1);
            if (index % offset != 0) return;

            data[index + offset - 1] += data[index + (1 << d) - 1];
            
		}
        __global__ void kernDownSweep(int n, int d, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;
			int offset = 1 << (d + 1);
            if (index % offset != 0) return;

			int t = data[index + (1 << d) - 1];
			data[index + (1 << d) - 1] = data[index + offset - 1];
			data[index + offset - 1] += t;
            
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            // TODO
            int dataLen = n * sizeof(int);
            int paddedN = 1 << ilog2ceil(n);
            int paddedLen = paddedN * sizeof(int);
            dim3 fullBlocksPerGrid((paddedN + blockSize - 1) / blockSize);

            int* dev_data;
            cudaMalloc((void**)&dev_data, paddedLen);
            cudaMemcpy(dev_data, idata, dataLen, cudaMemcpyHostToDevice);
            cudaMemset(dev_data + n, 0, paddedLen - dataLen);
            timer().startGpuTimer();
			// exclusive scan
            {
                // UpSweep
                for (int d = 0; d <= log2(paddedN) - 1; d++) {
                    kernUpSweep << <fullBlocksPerGrid, blockSize >> > (paddedN, d, dev_data);
                }
                cudaDeviceSynchronize();
                cudaMemset(dev_data + paddedN - 1, 0, sizeof(int));

                // DownSweep
                for (int d = log2(paddedN) - 1; d >= 0; d--) {
                    kernDownSweep << <fullBlocksPerGrid, blockSize >> > (paddedN, d, dev_data);
                }
            }
            timer().endGpuTimer();
            // Copy the result back to the host
            cudaMemcpy(odata, dev_data , dataLen, cudaMemcpyDeviceToHost);
			cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            // TODO
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			int* dev_bools;
			int* dev_indices;
			int* dev_idata;
			int* dev_odata;

			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMalloc((void**)&dev_odata, n * sizeof(int));

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata);
            scan(n, dev_indices, dev_bools);
            timer().startGpuTimer();
            Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
			
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

			int last_bool;
			cudaMemcpy(&last_bool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int last_count;
            cudaMemcpy(&last_count, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            
            cudaFree(dev_bools);
			cudaFree(dev_indices);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
            timer().endGpuTimer();
            return last_count + last_bool;
        }
    }
}
