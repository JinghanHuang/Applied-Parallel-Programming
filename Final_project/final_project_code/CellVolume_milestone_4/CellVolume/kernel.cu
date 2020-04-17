#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define WHITE 255
#define X_PIXEL_UNIT 0.065
#define Y_PIXEL_UNIT 0.065
#define Z_PIXEL_UNIT 0.29

#define X 437
#define Y 415
#define Z 63

#define FILENAME_SIZE 256
#define CHARS_SIZE 10
#define DECIMAL 10

#define BLOCK_SIZE 8

typedef unsigned int uint;
typedef unsigned char uchar;

cudaError_t bfsWithCuda(unsigned char *cell, unsigned int* volume, int *disjoint_set,
	uint *block_vol, int x_size, int y_size, int z_size, int block_num);

__global__ void simpleBfs(unsigned char* global_cell, int* disjoint_set, uint *block_vol, 
	uint* global_vol, int imageX, int imageY, int imageZ) {

	// cell: original pixel value
	__shared__ uchar cell[BLOCK_SIZE][BLOCK_SIZE];
	// dist: at which BFS layer
	__shared__ uint dist[BLOCK_SIZE][BLOCK_SIZE];
	// shared_vol: the volume for each block
	__shared__ uint shared_vol;

	int dir[4][2] = {
		{ -1,0 }, // go left:    x--
		{ 1,0 },  // go right:   x++
		{ 0,-1 }, // go up:      y--
		{ 0,1 }   // go down:    y++
	};

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tIdx = ty * blockDim.x + tx;
	int bIdx = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z;

	// copy global cell pixels to the shared memory
	if ((x < imageX) && (y < imageY)) {
		cell[tx][ty] = global_cell[z*imageX*imageY + y*imageX + x];
	}
	else {
		cell[tx][ty] = 2; // This means that the index is out of range
	}

	if (tx == 0 && ty == 0) {
		block_vol[blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x] = 0;
	}
	

	int start_x = blockIdx.x * blockDim.x;
	int start_y = blockIdx.y * blockDim.y;

	int mid_x = blockIdx.x * blockDim.x + blockDim.x / 2;
	int mid_y = blockIdx.y * blockDim.y + blockDim.y / 2;

	int end_x = (blockIdx.x + 1) * blockDim.x;
	int end_y = (blockIdx.y + 1) * blockDim.y;

	// We will choose the middle point in the block as a start point
	if ((x < imageX) && (y < imageY)) {
		if (x == mid_x && y == mid_y) {
			dist[tx][ty] = 1;
		}
		else {
			dist[tx][ty] = 0;
		}
	}

	uint level;
	uint local_vol;
	uint prevVol = -1;
	shared_vol = 1; // We initialize this to 1 because we need to count the start point
	__syncthreads();
	
	for (level = 1; (prevVol != shared_vol) ; level++) {
		local_vol = 0;
		__syncthreads();
		prevVol = shared_vol;
		// if the thread is on the level of BFS, it starts to search 
		if (dist[tx][ty] == (uint)level) {
			for (int dir_idx = 0; dir_idx < 4; dir_idx++) {

				// neighbor pixel's index
				int next_tx = tx + dir[dir_idx][0];
				int next_ty = ty + dir[dir_idx][1];
				int next_x = x + dir[dir_idx][0];
				int next_y = y + dir[dir_idx][1];

				if ((next_tx >= 0) && (next_ty >= 0) && \
					(next_tx < blockDim.x) && (next_ty < blockDim.y) && \
					(next_x < imageX) && (next_y < imageY) && \
					(cell[next_tx][next_ty] == 0) && !atomicCAS((uint *)&dist[next_tx][next_ty], (uint)0, (uint)level + 1)) {
					local_vol += 1;
				}
			}
		}
		atomicAdd(&shared_vol, local_vol);
		__syncthreads();
	}
	
	// For each block, only one thread needs to add the volume in the shared memory to the global memory
	if (tx == 0 && ty == 0) {
		block_vol[bIdx] = shared_vol;
	}
	
	// If the block is not on the membreane,
	// we can choose to connect it to other blocks by using disjointset
	if (shared_vol == BLOCK_SIZE * BLOCK_SIZE) {
		if (tx + 1 == blockDim.x && x + 1 < imageX) {
			int neighborBlock2DIdx = bIdx + 1;
			atomicCAS((int *)&disjoint_set[neighborBlock2DIdx], neighborBlock2DIdx, bIdx);
		}
		if (ty + 1 == blockDim.y && y + 1 < imageY) {
			int neighborBlock2DIdx = gridDim.x + bIdx;
			atomicCAS((int *)&disjoint_set[neighborBlock2DIdx], neighborBlock2DIdx, bIdx);
		}
	}

	// If the block in the right bottom corner, we will connect it to the its left block
	if (blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1) {
		atomicCAS((int *)&disjoint_set[bIdx], bIdx, bIdx - 1);
	}
}

int main()
{

	/* The variables are used to get filename and open file*/
	FILE *file;
	const char a[CHARS_SIZE] = "csv/im_";
	char b[CHARS_SIZE];
	const char c[CHARS_SIZE] = ".csv";
	char filename[FILENAME_SIZE];

	unsigned char* cell;
	cell = (unsigned char*)malloc((X*Y*Z)*sizeof(unsigned int));
	unsigned int* volume;
	volume = (unsigned int*)malloc(sizeof(unsigned int));
	*volume = 0;
	int i, j, k;

	uint block_num = ceil(X / (float)BLOCK_SIZE) * ceil(Y / (float)BLOCK_SIZE) * Z;

	int *disjoint_set;
	disjoint_set = (int*)malloc(block_num * sizeof(int));
	uint *block_vol;
	block_vol = (uint*)malloc(block_num * sizeof(uint));
	uint *block_vol_copy;
	block_vol_copy = (uint*)malloc(block_num * sizeof(uint));

	for (k = 0; k < Z; k++) {
		// get filename
		memset(filename, '\0', sizeof(filename));
		memset(b, '\0', sizeof(b));
		strcpy(filename, a);
		if (k / DECIMAL == 0) {
			b[0] = '0' + k;
		}
		else {
			b[0] = '0' + k / DECIMAL;
			b[1] = '0' + k % DECIMAL;
		}
		strcat(filename, b);
		strcat(filename, c);

		// read file
		file = fopen(filename, "r");
		for (i = 0; i < X; i++) {
			fscanf(file, "%d", &cell[k*X*Y + i]);
			for (j = 1; j < Y; j++) {
				fscanf(file, ",%d", &cell[k*X*Y + j*X + i]);
			}
		}
		fclose(file);
	}

	// Initialize the disjoint set
	for (uint q = 0; q < block_num; q++) {
		disjoint_set[q] = (int) q;
	}

	// Use BFS to calculate cell volume
    cudaError_t cudaStatus = bfsWithCuda(cell, volume, disjoint_set, block_vol, X, Y, Z, block_num);

	memcpy(block_vol_copy, block_vol, block_num * sizeof(uint));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	int dir[4][2] = {
		{ -1,0 }, // go left:    x--
		{ 1,0 },  // go right:   x++
		{ 0,-1 }, // go up:      y--
		{ 0,1 }   // go down:    y++
	};

	int b_y = ceil(Y / (float)BLOCK_SIZE);
	int b_x = ceil(X / (float)BLOCK_SIZE);

	// We use disjoint set to connect related blocks
	int temp = (block_num / Z);
	for (int p = 0; p < Z; p++) {
		for (int bIdx = 0; bIdx < temp; bIdx++) {
			int anc1 = disjoint_set[p * temp + bIdx];
			while (anc1 != disjoint_set[anc1]) anc1 = disjoint_set[anc1];
			if (anc1 != p * temp + bIdx)
				block_vol[anc1] += block_vol[p * temp + bIdx];
			disjoint_set[p * temp + bIdx] = anc1;
			for (int dirIdx = 0; dirIdx < 4; dirIdx++) {
				int y = bIdx / b_x;
				int x = bIdx % b_x;
				int neighborX = x + dir[dirIdx][0];
				int neighborY = y + dir[dirIdx][1];
				if (!(neighborX >= 0) || !(neighborX < b_x) || !(neighborY >= 0) || !(neighborY < b_y)) continue;
				int neighborIdx = p * temp + neighborX + b_x*neighborY;

				int anc2 = disjoint_set[neighborIdx];
				while (anc2 != disjoint_set[anc2]) anc2 = disjoint_set[anc2];
				if (anc2 != neighborIdx)
					block_vol[anc2] += block_vol[neighborIdx];
				disjoint_set[neighborIdx] = anc2;
				if ((block_vol_copy[p * temp + bIdx] == BLOCK_SIZE*BLOCK_SIZE || y == b_y - 1 || x == b_x - 1) && (block_vol_copy[neighborIdx] == BLOCK_SIZE*BLOCK_SIZE || neighborY == b_y - 1 || neighborX == b_x - 1) && anc1 != anc2) {
					disjoint_set[anc2] = anc1;
					block_vol[anc1] += block_vol[anc2];
				}
			}
		}
		int ancestor = disjoint_set[p * temp];
		while (ancestor != disjoint_set[ancestor]) ancestor = disjoint_set[ancestor];

		// add up internal pixels
		int internalBlocks = 0;
		for (j = 0; j < b_y; j++) {
			for (i = 0; i < b_x; i++) {
				int anc = disjoint_set[p * temp + j * b_x + i];
				while (anc != disjoint_set[anc]) anc = disjoint_set[anc];
				if (anc == ancestor) {
					//printf("%d ", 0);
				}
				else {
					//printf("%d ", 1);
					internalBlocks++;
				}
			}
			//printf("\n");
		}
		//printf("The %d th layer internal pixels: %d\n", p, internalBlocks*BLOCK_SIZE*BLOCK_SIZE);
		*volume += internalBlocks*BLOCK_SIZE*BLOCK_SIZE;
	}

	printf("The pixel size: %d\n", *volume);
	float volume_;
	volume_ = (float)(*volume) * X_PIXEL_UNIT * Y_PIXEL_UNIT * Z_PIXEL_UNIT;
	printf("The volume of the cell: %f fL \n", volume_);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

    cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	free(cell);
	free(volume);
	free(disjoint_set);
	free(block_vol);
	free(block_vol_copy);

	getchar();
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t bfsWithCuda(unsigned char *cell, unsigned int* volume, int *disjoint_set, uint *block_vol, int x_size, int y_size, int z_size, int block_num)
{
	unsigned char *dev_cell;
	unsigned int *dev_volume;
	int *dev_disjoint_set;
	uint *dev_block_vol;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_cell, x_size * y_size * z_size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_volume, sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_disjoint_set, block_num * sizeof(int)); // need to add z_size later
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_block_vol, block_num * sizeof(unsigned int)); // need to add z_size later
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_cell, cell, x_size * y_size * z_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_volume, volume, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_disjoint_set, disjoint_set, block_num * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemset(dev_block_vol, 0, block_num * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	dim3 dimGrid(ceil(x_size /(float) BLOCK_SIZE), ceil(y_size /(float)BLOCK_SIZE), z_size);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    // Launch a kernel on the GPU with one thread for each element.
	simpleBfs<<<dimGrid, dimBlock>>>(dev_cell, dev_disjoint_set, dev_block_vol, dev_volume, x_size, y_size, z_size);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(volume, dev_volume, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(disjoint_set, dev_disjoint_set, block_num * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(block_vol, dev_block_vol, block_num * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
    cudaFree(dev_cell);
    cudaFree(dev_volume);
	cudaFree(dev_disjoint_set);
	cudaFree(dev_block_vol);

    return cudaStatus;
}


