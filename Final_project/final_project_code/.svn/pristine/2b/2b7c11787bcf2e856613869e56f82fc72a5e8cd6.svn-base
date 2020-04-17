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

#define BLOCK_SIZE 32

typedef unsigned int uint;
typedef unsigned char uchar;

cudaError_t bfsWithCuda(unsigned char *cell, unsigned int* volume, int x_size, int y_size, int z_size);

__global__ void simpleBfs(unsigned char* cell, uchar* dist, uint* global_vol, int imageX, int imageY, int imageZ) {
	// cell: original pixel value
	//__shared__ uchar cell[X][Y];
	// dist: at which BFS layer
	//__shared__ uchar dist[X][Y];
	//__shared__ int shared_vol;

	// copy global cell pixels to the shared memory
	//cell[x][y] = global_cell[bz*imageX*imageY + y*imageX + x];
	//dist[x][y] = 0;

	int dir[4][2] = {
		{ -1,0 }, // go down:     x--
		{ 1,0 },  // go up:       x++
		{ 0,-1 }, // go left:     y--
		{ 0,1 }   // go right:    y++
	};

	int bz = blockIdx.z;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// We randomly choose a point as a start point
	if (x == 200 && y == 200) {
		dist[y * imageX + x] = 1;
	}

	uint level;
	int prevVol = -1;
	__syncthreads();

	
	for (level = 1; (prevVol != *global_vol); level++) {
		uint local_vol = 0;
		__syncthreads();
		// if the thread is on the level of BFS, it starts to search 
		if (dist[y*imageX + x] == (uchar)level) {
			prevVol = *global_vol;
			for (int dir_idx = 0; dir_idx < 4; dir_idx++) {
				// neighbor pixel's index
				int next_x = x + dir[dir_idx][0];
				int next_y = y + dir[dir_idx][1];
				// if the neighbor pixels are not visited before
				if ((next_x >= 0) && (next_y >= 0) && \
					(next_x < imageX) && (next_y < imageY) \
					&& (dist[next_y*imageX + next_x] == 0) && (cell[next_y*imageX + next_x] == 0)) {
					// visit it in the next level/loop
					dist[next_y*imageX + next_x] = level + 1;
					atomicAdd(global_vol, 1);
				}
			}
		}
		__syncthreads();
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

/*
	for (j = 0; j < Y; j++) {
		for (i = 0; i < X; i++) {
			printf("%d ", cell[j*X + i]);
		}
		printf("\n");
	}
*/

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
		//printf("%s\n", filename);
		file = fopen(filename, "r");
		for (i = 0; i < X; i++) {
			fscanf(file, "%d", &cell[k*X*Y + i]);
			for (j = 1; j < Y; j++) {
				fscanf(file, ",%d", &cell[k*X*Y + j*X + i]);
			}
		}
		fclose(file);
	}

	// Use BFS to calculate cell volume
    cudaError_t cudaStatus = bfsWithCuda(cell, volume, X, Y, Z);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	printf("%d\n", *volume);
	float volume_;
	volume_ = (float) (*volume) * X_PIXEL_UNIT * Y_PIXEL_UNIT * Z_PIXEL_UNIT;
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

	getchar();
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t bfsWithCuda(unsigned char *cell, unsigned int* volume, int x_size, int y_size, int z_size)
{
	unsigned char *dev_cell;
	unsigned int *dev_volume;
	unsigned char *dev_dist;
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

	cudaStatus = cudaMalloc((void**)&dev_dist, x_size * y_size * z_size * sizeof(unsigned char));
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

	cudaStatus = cudaMemset(dev_dist, 0, x_size * y_size * z_size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	dim3 dimGrid(ceil(x_size / BLOCK_SIZE), ceil(y_size / BLOCK_SIZE), 1);
	//dim3 dimGrid(ceil(x_size / BLOCK_SIZE), ceil(y_size / BLOCK_SIZE), z_size);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    // Launch a kernel on the GPU with one thread for each element.
	simpleBfs<<<dimGrid, dimBlock>>>(dev_cell, dev_dist, dev_volume, x_size, y_size, z_size);
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

Error:
    cudaFree(dev_cell);
    cudaFree(dev_volume);
	cudaFree(dev_dist);

    return cudaStatus;
}


