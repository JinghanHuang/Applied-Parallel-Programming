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

//#define X 437
//#define Y 415
//#define Z 63

#define FILENAME_SIZE 256
#define CHARS_SIZE 15
#define DECIMAL 10

#define BLOCK_SIZE 8

typedef unsigned int uint;
typedef unsigned char uchar;
cudaError_t SumWithCuda(unsigned char *cell, uint *block_vol, int x_size, int y_size, int z_size, int block_num);
cudaError_t bfsWithCuda(unsigned char *cell, unsigned int* volume, int *block_type, 
	uint *block_vol, int x_size, int y_size, int z_size, int block_num);

__global__ void plainSum(unsigned char* global_cell, uint *block_vol,
	int imageX, int imageY, int imageZ) {

	// cell: original pixel value
	//__shared__ uchar cell[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ uint shared_vol;


	int tx = threadIdx.x;
	int ty = threadIdx.y;
	//int tIdx = ty * blockDim.x + tx;
	int bIdx = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z;
	//shared_vol = 0;
	shared_vol = BLOCK_SIZE*BLOCK_SIZE;
	__syncthreads();
	// copy global cell pixels to the shared memory
	//cell[tx][ty] = global_cell[z*imageX*imageY + y*imageX + x];
	if (global_cell[z*imageX*imageY + y*imageX + x] != 0) {
		atomicAdd(&shared_vol, -1);
	}
	if (tx == 0 && ty == 0)
	block_vol[bIdx] = shared_vol;
}


__global__ void parallelBFS(unsigned char* global_cell, int *block_type, uint *block_vol, 
	uint* global_vol, int imageX, int imageY, int imageZ) {

	// cell: original pixel value
	__shared__ uchar cell[BLOCK_SIZE][BLOCK_SIZE];
	// dist: at which BFS layer
	__shared__ uint dist[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ uint shared_vol;

	int dir[4][2] = {
		{ -1,0 }, // go left:    x--
		{ 1,0 },  // go right:   x++
		{ 0,-1 }, // go up:      y--
		{ 0,1 }   // go down:    y++
	};

	//int bx = blockIdx.x;
	//int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tIdx = ty * blockDim.x + tx;
	int bIdx = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z;

	// copy global cell pixels to the shared memory
	shared_vol = 0;
	dist[tx][ty] = 0;

	__syncthreads();

	cell[tx][ty] = global_cell[z*imageX*imageY + y*imageX + x];
	// strategy 1
	//if ((tx == 0 && ty == 0) || (tx == 0 && ty == blockDim.y - 1) || (tx == blockDim.x - 1 && ty == 0) || (tx == blockDim.x - 1 && ty == blockDim.y - 1)) {
	// strategy 2
	//if ((ty == 0) || (ty == blockDim.y - 1)) {
	// strategy 3
		if (cell[tx][ty] == 0 && block_type[bIdx] != 0) {
			if (tx == 0 && blockIdx.x != 0 && block_type[bIdx - 1] == 0)
				dist[tx][ty] = 1;
			if (ty == 0 && blockIdx.y != 0 && block_type[bIdx - gridDim.x] == 0)
				dist[tx][ty] = 1;
			if (tx == blockDim.x - 1 && blockIdx.x != gridDim.x - 2 && block_type[bIdx + 1] == 0)
				dist[tx][ty] = 1;
			if (ty == blockDim.y - 1 && blockIdx.y != gridDim.y - 2 && block_type[bIdx + gridDim.x] == 0)
				dist[tx][ty] = 1;

			if (tx == 0 && blockIdx.x != 0 && ty == 0 && blockIdx.y != 0 && block_type[bIdx - 1 - gridDim.x] == 0)
				dist[tx][ty] = 1;
			if (tx == blockDim.x - 1 && blockIdx.x != gridDim.x - 2 && ty == blockDim.y - 1 && blockIdx.y != gridDim.y - 2 && block_type[bIdx + 1 + gridDim.x] == 0)
				dist[tx][ty] = 1;
			if (tx == 0 && blockIdx.x != 0 && ty == blockDim.y - 1 && blockIdx.y != gridDim.y - 2 && block_type[bIdx - 1 + gridDim.x] == 0)
				dist[tx][ty] = 1;
			if (tx == blockDim.x - 1 && blockIdx.x != gridDim.x - 2 && ty == 0 && blockIdx.y != 0 && block_type[bIdx + 1 - gridDim.x] == 0)
				dist[tx][ty] = 1;

			if (dist[tx][ty] == 1) {
				atomicAdd(&shared_vol, 1);
				block_type[bIdx] = 2;
			}
		}
	//}

	uint level;
	int prevVol = -1;
	
	__syncthreads();
	
	uint local_vol;
	
	for (level = 1; (prevVol != shared_vol) ; level++) {
		local_vol = 0;
		//__syncthreads();
		prevVol = shared_vol;
		// if the thread is on the level of BFS, it starts to search 
		if (dist[tx][ty] == (uint)level) {
			for (int dir_idx = 0; dir_idx < 4; dir_idx++) {
				// neighbor pixel's index
				int next_tx = tx + dir[dir_idx][0];
				int next_ty = ty + dir[dir_idx][1];
				int next_x = x + dir[dir_idx][0];
				int next_y = y + dir[dir_idx][1];
				// if the neighbor pixels are not visited before
				if ((next_tx >= 0) && (next_ty >= 0) && \
					(next_tx < blockDim.x) && (next_ty < blockDim.y) && \
					(cell[next_tx][next_ty] == 0) && !atomicCAS((uint *)&dist[next_tx][next_ty], (uint)0, (uint)level + 1)) {
					local_vol += 1;
				}
			}
			atomicAdd(&shared_vol, local_vol);
		}
		__syncthreads();
	}
	
	if (tx == 0 && ty == 0)
		atomicAdd(&global_vol[z], shared_vol);
}

int main()
{
	/* The variables are used to get filename and open file*/
	char b[CHARS_SIZE];
	//const char c[CHARS_SIZE] = ".csv";
	char filename[FILENAME_SIZE];
	char cellsize[FILENAME_SIZE];
	const char a[CHARS_SIZE] = "Segmentation/";
	const char c[CHARS_SIZE] = "/size.csv";
	const char d[CHARS_SIZE] = "/CSV/array_";
	const char e[CHARS_SIZE] = ".csv";

	int i, j, k;
	int x_size, y_size, z_size;



	/* The variables are used to get filename and open file*/
	FILE *file, *dirc, *image;
	char oneproduct[10];
	char im_name[10];

	if ((dirc = fopen("Segmentation/dir.csv", "r")) == NULL) {
		fprintf(stderr, "File dir.csv connot open\n");
		while (1) {}
		//exit(-1);
	}
	int index = 0;
	while (1) {
		clock_t t_start = clock();
		if (fgets(oneproduct, 10, dirc) == NULL) {
			if (fgets(oneproduct, 10, dirc) == NULL) {
				break;
			}
		}

		memset(im_name, '\0', sizeof(im_name));
		memset(b, '\0', sizeof(b));
		strcat(im_name, "im_");
		if (index / DECIMAL == 0) {
			b[0] = '0' + index;
		}
		else {
			b[0] = '0' + index / DECIMAL;
			b[1] = '0' + index % DECIMAL;
		}

		strcat(im_name, b);


		memset(cellsize, '\0', FILENAME_SIZE * sizeof(char));
		strcpy(cellsize, a);
		strcat(cellsize, im_name);
		strcat(cellsize, c);
		if ((image = fopen(cellsize, "r")) == NULL) {
			fprintf(stderr, "File size.csv connot open\n");
			while (1) {}
			//exit(-1);
		}

		printf("succeed in open %d th cell %s:	", index, cellsize);

		// get zyx size of image
		fscanf(image, "%d", &z_size);
		fscanf(image, ",%d", &y_size);
		fscanf(image, ",%d", &x_size);
		printf("z = %d	", z_size);
		printf("y = %d	", y_size);
		printf("x = %d\n", x_size);

		//////////////////////////////////// Basic Data Preprocessing - Begin ////////////////////////////////////
		/*
		int x_range = (ceil(x_size / (float)BLOCK_SIZE) + 2) * BLOCK_SIZE;
		int y_range = (ceil(y_size / (float)BLOCK_SIZE) + 2) * BLOCK_SIZE;

		unsigned char* cell;
		cell = (unsigned char*)malloc(z_size * y_range * x_range * sizeof(unsigned int));
		unsigned int* volume;
		volume = (unsigned int*)malloc(z_size * sizeof(unsigned int));
		memset(volume, 0, sizeof(volume));

		//*volume = 0;

		for (k = 0; k < z_size; k++) {
			// get filename
			memset(filename, '\0', sizeof(filename));
			memset(b, '\0', sizeof(b));
			strcpy(filename, a);
			strcat(filename, im_name);
			strcat(filename, d);
			if (k / DECIMAL == 0) {
				b[0] = '0' + k;
			}
			else {
				b[0] = '0' + k / DECIMAL;
				b[1] = '0' + k % DECIMAL;
			}
			strcat(filename, b);
			strcat(filename, e);

			//printf("	scan %d layer: %s\n", k, filename);

			// read file
			//printf("%s\n", filename);
			if ((file = fopen(filename, "r")) == NULL) {
				fprintf(stderr, "File array.csv connot open\n");
				while (1) {}
				//exit(-1);
			}
			for (i = 0; i < y_size; i++) {
				fscanf(file, "%d", &cell[k*x_range*y_range + (i + BLOCK_SIZE) * x_range + BLOCK_SIZE]);
				for (j = 1; j < x_size; j++) {
					fscanf(file, ",%d", &cell[k*x_range*y_range + (i + BLOCK_SIZE) * x_range + j + BLOCK_SIZE]);
				}
			}
			fclose(file);
		}

		x_size = x_range;
		y_size = y_range;
		*/
		//////////////////////////////////// Basic Data Preprocessing - End ////////////////////////////////////

		//////////////////////////////////// Optimized Data Preprocessing - Begin ////////////////////////////////////
		
		int x_len = 0;
		int y_len = 0;
		int z_min = z_size;
		int z_max = 0;
		int *x_min;
		x_min = (int *)malloc(z_size * sizeof(int));
		int *x_max;
		x_max = (int *)malloc(z_size * sizeof(int));
		int *y_min;
		y_min = (int *)malloc(z_size * sizeof(int));
		int *y_max;
		y_max = (int *)malloc(z_size * sizeof(int));

		for (int p = 0; p < z_size; p++) {
			x_max[p] = 0;
			x_min[p] = x_size;
			y_max[p] = 0;
			y_min[p] = y_size;
		}

		unsigned char* whole_cell;
		whole_cell = (unsigned char*)malloc(z_size * y_size * x_size * sizeof(unsigned int));
		unsigned int* volume;
		volume = (unsigned int*)malloc(z_size * sizeof(unsigned int));
		memset(volume, 0, sizeof(volume));

		for (k = 0; k < z_size; k++) {
			// get filename
			memset(filename, '\0', sizeof(filename));
			memset(b, '\0', sizeof(b));
			strcpy(filename, a);
			strcat(filename, im_name);
			strcat(filename, d);
			if (k / DECIMAL == 0) {
				b[0] = '0' + k;
			}
			else {
				b[0] = '0' + k / DECIMAL;
				b[1] = '0' + k % DECIMAL;
			}
			strcat(filename, b);
			strcat(filename, e);

			//printf("	scan %d layer: %s\n", k, filename);

			// read file
			//printf("%s\n", filename);
			if ((file = fopen(filename, "r")) == NULL) {
				fprintf(stderr, "File array.csv connot open\n");
				while (1) {}
				//exit(-1);
			}
			for (j = 0; j < y_size; j++) {
				fscanf(file, "%d", &whole_cell[k * x_size * y_size + j * x_size]);
				if (whole_cell[k * x_size * y_size + j * x_size] > 0) {
					if (z_min > k)
						z_min = k;
					if (z_max < k)
						z_max = k;
					if (0 < x_min[k])
						x_min[k] = 0;
					if (0 > x_max[k])
						x_max[k] = 0;
					if (j < y_min[k])
						y_min[k] = j;
					if (j > y_max[k])
						y_max[k] = j;
				}
				for (i = 1; i < x_size; i++) {
					fscanf(file, ",%d", &whole_cell[k * x_size * y_size + j * x_size + i]);
					if (whole_cell[k * x_size * y_size + j * x_size + i] > 0) {
						if (z_min > k)
							z_min = k;
						if (z_max < k)
							z_max = k;
						if (i < x_min[k])
							x_min[k] = i;
						if (i > x_max[k])
							x_max[k] = i;
						if (j < y_min[k])
							y_min[k] = j;
						if (j > y_max[k])
							y_max[k] = j;
					}
				}
			}
			if (x_max[k] - x_min[k] + 1 > x_len)
				x_len = x_max[k] - x_min[k] + 1;
			if (y_max[k] - y_min[k] + 1 > y_len)
				y_len = y_max[k] - y_min[k] + 1;
			fclose(file);
		}

		int x_range = (ceil((x_len+2) / (float)BLOCK_SIZE) + 2) * BLOCK_SIZE;
		int y_range = (ceil((y_len+2) / (float)BLOCK_SIZE) + 2) * BLOCK_SIZE;
		int z_range = z_max - z_min + 1;

		unsigned char* cell;
		cell = (unsigned char*)malloc(z_range * y_range * x_range * sizeof(unsigned int));
		for (k = 0; k < z_range; k++) {
			for (j = 0; j < y_len; j++) {
				for (i = 0; i < x_len; i++) {
					int cell_z = k + z_min;
					int cell_y = j + y_min[cell_z];
					int cell_x = i + x_min[cell_z];
					if (cell_x < x_size && cell_y < y_size)
						cell[k * y_range * x_range + (j + BLOCK_SIZE + 1) * x_range + i + BLOCK_SIZE + 1] = whole_cell[cell_z * y_size * x_size + cell_y * x_size + cell_x];
					else
						cell[k * y_range * x_range + (j + BLOCK_SIZE + 1) * x_range + i + BLOCK_SIZE + 1] = 0;
				}
			}
		}

		x_size = x_range;
		y_size = y_range;
		z_size = z_range;
		
		//////////////////////////////////// Optimized Data Preprocessing - End ////////////////////////////////////

		index++;

		uint block_num = ceil(x_size / (float)BLOCK_SIZE) * ceil(y_size / (float)BLOCK_SIZE) * z_size;

		int *disjoint_set;
		disjoint_set = (int*)malloc(block_num * sizeof(int));
		int *block_type;
		block_type = (int*)malloc(block_num * sizeof(int));
		uint *block_vol;
		block_vol = (uint*)malloc(block_num * sizeof(uint));
		//uint *block_vol_copy;
		//block_vol_copy = (uint*)malloc(block_num * sizeof(uint));


		clock_t t_cudaStart = clock();
		// Count the non-membrane pixels in each block.
		cudaError_t cudaStatus = SumWithCuda(cell, block_vol, x_size, y_size, z_size, block_num);
		clock_t t_0 = clock();


		int b_y = ceil(y_size / (float)BLOCK_SIZE);
		int b_x = ceil(x_size / (float)BLOCK_SIZE);

		for (uint q = 0; q < block_num; q++) {
			disjoint_set[q] = (int)q;
		}

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

		int temp = (block_num / z_size);
		for (int p = 0; p < z_size; p++) {
			for (int bIdx = 0; bIdx < temp; bIdx++) {
				int anc1 = disjoint_set[p * temp + bIdx];
				while (anc1 != disjoint_set[anc1]) anc1 = disjoint_set[anc1];
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
					disjoint_set[neighborIdx] = anc2;
					// union two ancescors
					if ((block_vol[p * temp + bIdx] == BLOCK_SIZE*BLOCK_SIZE) && (block_vol[neighborIdx] == BLOCK_SIZE*BLOCK_SIZE) && anc1 != anc2) {
						disjoint_set[anc2] = anc1;
						disjoint_set[neighborIdx] = anc1;
					}
				}
			}

			int ancestor = disjoint_set[p * temp];
			while (ancestor != disjoint_set[ancestor]) ancestor = disjoint_set[ancestor];

			for (j = 0; j < b_y; j++) {
				for (i = 0; i < b_x; i++) {
					int anc = disjoint_set[p * temp + j * b_x + i];
					while (anc != disjoint_set[anc]) anc = disjoint_set[anc];
					if (anc == ancestor) {
						block_type[p * temp + j * b_x + i] = 0;
					}
					else {
						block_type[p * temp + j * b_x + i] = 1;
					}
				}
			}
		}


		memset(volume, 0, sizeof(volume));
		clock_t t_1 = clock();
		cudaError_t cudaStatus1 = bfsWithCuda(cell, volume, block_type, block_vol, x_size, y_size, z_size, block_num);
		clock_t t_2 = clock();

		int mBlock = 0;
		int ePixel = 0;
		int tVolum = 0;
		for (int p = 0; p < z_size; p++) {
			int externalPixels = 0;
			int membraneBlocks = 0;
			for (j = 0; j < b_y; j++) {
				for (i = 0; i < b_x; i++) {
					//printf("%d ", block_type[p * temp + j * b_x + i]);
					if (block_type[p * temp + j * b_x + i] == 0) {
						externalPixels += block_vol[p * temp + j * b_x + i];
					}
					if (block_type[p * temp + j * b_x + i] == 2) {
						membraneBlocks++;
					}
				}
				//printf("\n");
			}
			mBlock += membraneBlocks;
			ePixel += externalPixels;
			tVolum += volume[p];

			//printf("# special volume: %d\n", volume[p]);
			//printf("# membraneBlocks: %d\n", membraneBlocks);
			//printf("# external pixels (empty block): %d\n", externalPixels);
		}
		//printf("special volume: %d\n", tVolum);
		//printf("# total membraneBlocks: %d\n", mBlock);
		//printf("# total external pixels (empty block): %d\n", ePixel);
		//printf("# total internal pixels: %d\n", Z*X*Y - ePixel - tVolum);
		clock_t t_final = clock();
		float volume_;
		volume_ = (z_size*y_size*x_size - ePixel - tVolum) * X_PIXEL_UNIT * Y_PIXEL_UNIT * Z_PIXEL_UNIT;
		printf("The # pixel is: %d \n", z_size*y_size*x_size - ePixel - tVolum);
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
		free(whole_cell);
		free(x_min);
		free(x_max);
		free(y_min);
		free(y_max);

		clock_t t_end = clock();
		printf("data import and setup time: %f ms\n", (t_cudaStart - t_start)*1.0 / CLOCKS_PER_SEC * 1000);
		printf("cuda part 1 time: %f ms\n", (t_0 - t_cudaStart)*1.0 / CLOCKS_PER_SEC * 1000);
		printf("sequential time: %f ms\n", (t_1 - t_0)*1.0 / CLOCKS_PER_SEC * 1000);
		printf("cuda part 2 time: %f ms\n", (t_2 - t_1)*1.0 / CLOCKS_PER_SEC * 1000);
		printf("final calc time: %f ms\n", (t_final - t_2)*1.0 / CLOCKS_PER_SEC * 1000);
		printf("remaining time: %f ms\n", (t_end - t_final)*1.0 / CLOCKS_PER_SEC * 1000);
		printf("total running time: %f ms\n", (t_end - t_start)*1.0 / CLOCKS_PER_SEC * 1000);
	}

	fclose(dirc);
	getchar();
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t SumWithCuda(unsigned char *cell, uint *block_vol, int x_size, int y_size, int z_size, int block_num)
{
	unsigned char *dev_cell;
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


	cudaStatus = cudaMemset(dev_block_vol, 0, block_num * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	dim3 dimGrid(ceil(x_size / (float)BLOCK_SIZE), ceil(y_size / (float)BLOCK_SIZE), z_size);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	// Launch a kernel on the GPU with one thread for each element.
	plainSum <<<dimGrid, dimBlock >> >(dev_cell, dev_block_vol, x_size, y_size, z_size);

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


	cudaStatus = cudaMemcpy(block_vol, dev_block_vol, block_num * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_cell);
	cudaFree(dev_block_vol);
	return cudaStatus;
}

cudaError_t bfsWithCuda(unsigned char *cell, unsigned int* volume, int *block_type, uint *block_vol, int x_size, int y_size, int z_size, int block_num)
{
	unsigned char *dev_cell;
	unsigned int *dev_volume;
	int *dev_block_type;
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

    cudaStatus = cudaMalloc((void**)&dev_volume, z_size * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_block_type, block_num * sizeof(int)); // need to add z_size later
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

	cudaStatus = cudaMemset(dev_volume, 0, z_size * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_block_vol, block_vol, block_num * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_block_type, block_type, block_num * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	dim3 dimGrid(ceil(x_size /(float) BLOCK_SIZE), ceil(y_size /(float)BLOCK_SIZE), z_size);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    // Launch a kernel on the GPU with one thread for each element.
	parallelBFS<<<dimGrid, dimBlock>>>(dev_cell, dev_block_type, dev_block_vol, dev_volume, x_size, y_size, z_size);
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
    cudaStatus = cudaMemcpy(volume, dev_volume, z_size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(block_type, dev_block_type, block_num * sizeof(int), cudaMemcpyDeviceToHost);
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
	cudaFree(dev_block_type);
	cudaFree(dev_block_vol);

    return cudaStatus;
}
