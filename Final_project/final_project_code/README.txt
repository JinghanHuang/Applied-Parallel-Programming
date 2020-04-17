CS408 Final Project Milestone 2
---------------------------------------
Names:
Jinghan Huang
Chao Xu
Run Zhang

---------------------------------------
1. AICS-10_46_3.ome.tif
We download this cell data from the Allen Cell Explorer

2. AICS-10_46_3.ome-c8.tif
We get this file from AICS-10_46_3.ome.tif by using Fiji. If you want to know the details,
please see “Point_Cloud_Dataset.pdf”. We send it to your email.

3. AICS_Cell-feature-analysis_v1.7.csv
We download it from the Allen Cell Explorer. In this file, we can get the real volume of the cell.

4. tif-preprocessing.ipynb
We obtain information in “.tif” and store these binary images as “csv” files in the file called 'csv'.

5.main.c
The code we use to calculate the volume of the cell.
To run the code,

gcc main.c
./a.out

You will see the output like
The number of pixels: 1156524
The volume of the cell: 1417.031006 fL 
Time used：1.084740 seconds 

