/************************************************************************
ECE 408 Final Project

Jinghan Huang
Chao Xu
Run Zhang

************************************************************************/

#include <stdlib.h>
#include <stdio.h>
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

#define BEFORE_STATE 0
#define MOVE_IN_STATE 1
#define STAY_STATE 2
#define MOVE_OUT_STATE 3
#define AFTER_STATE 4

/** main
 *
 */
int main()
{
	/* The variables are used to calculate time */
	clock_t start_t,finish_t;
    double total_t = 0;
    start_t = clock();

	/* The variables are used to get filename and open file*/
    FILE *file;
	char a[CHARS_SIZE]="csv/im_";
	char b[CHARS_SIZE];
	char c[CHARS_SIZE]=".csv";
	char filename[FILENAME_SIZE];

	/* The variables are used to store data in csv file */
	unsigned int cell[X][Y];
	unsigned int pixel = 0;
	int i,j,k;

	for(k = 0; k < Z; k++){
		/* get filename of each csv */
		memset(filename, '\0', sizeof(filename));
		strcpy(filename, a);
		if(k/DECIMAL == 0){
			b[0] = '0' + k/DECIMAL;
		}
		else{
			b[0] = '0' + k/DECIMAL;
			b[1] = '0' + k%DECIMAL;
		}
		strcat(filename, b);
		strcat(filename, c);

		/* read data from csv file */
		file = fopen(filename,"r");
		for(i = 0; i < X; i++){
			fscanf(file, "%d", &cell[i][0]);
			for(j = 1; j < Y; j++){
				fscanf(file, ",%d", &cell[i][j]);
			}
		}
		fclose(file);

		/* count the number of pixels */
		int state;
		int temp_count;
		int special_temp_count;
		for(i = 0; i < X; i++){
			state = BEFORE_STATE;
			temp_count = 0;
			special_temp_count = 0; // maybe only the surface of the cell is in this line

			// if we find the cell at the beginning of a line
			j = 0;
			if(cell[i][j] == WHITE){
				pixel++;
				state = MOVE_IN_STATE;
			}

			for(j = 1; j < Y; j++){
				// update state and we can know our current position
				if(state == BEFORE_STATE && cell[i][j] != cell[i][j-1]){
					state = MOVE_IN_STATE; // move in the cell
				}
				else if(state == MOVE_IN_STATE && cell[i][j] != cell[i][j-1]){
					state = STAY_STATE; // in the cell
				}
				else if(state == STAY_STATE && cell[i][j] != cell[i][j-1]){
					state = MOVE_OUT_STATE; // move out the cell
				}
				else if(state == MOVE_OUT_STATE && cell[i][j] != cell[i][j-1]){
					state = AFTER_STATE;
				}

				// count the pixels according to the state
				if(state == MOVE_IN_STATE){
					special_temp_count += 1;
				}
				if(state == MOVE_IN_STATE || state == STAY_STATE || state == MOVE_OUT_STATE){
					temp_count += 1;
				}
			}
			// add temp count to the total number of pixels
			if(state == STAY_STATE){
				pixel += special_temp_count;
			}
			if(state == AFTER_STATE){
				pixel += temp_count;
			}
		}
	}
	printf("The number of pixels: %d\n", pixel);

	/* calculate the volume */
	float volume;
	volume = pixel * X_PIXEL_UNIT * Y_PIXEL_UNIT * Z_PIXEL_UNIT;
	printf("The volume of the cell: %f fL \n", volume);

	/* timing is over */
	finish_t = clock();
	total_t = (double)(finish_t - start_t) / CLOCKS_PER_SEC;
    printf("Time used：%f seconds \n", total_t);
}

