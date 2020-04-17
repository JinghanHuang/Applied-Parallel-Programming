CS408 Final Project Milestone 4
---------------------------------------
Names:
Jinghan Huang
Chao Xu
Run Zhang
---------------------------------------
In CellVolume_milestone_4 file folder,
we have our parallel code in CellVolume/kernel.cu
our image data is stored in CellVolume/csv

Now we can apply our parallel code to run BFS on x-y-z point cloud dataset with multiple blocks.
Within one block, each thread will deal with one pixel and wait the BFS level to arrive there.
Then we use disjoint set to connect all the pixels outside the membrane.
And the sum of all connected block volumes would be the total volume outside the cell.

After opening project, Click "Local Windows Debugger" to run the code.

As expected, the output will be
1280832
The volume of the cell: 1569.339355 fL

PS: The real volume of the cell: 1570.106297 fL

---------------------------------------
Answers to the Questions
1. Detailed explanations of the optimization strategies

  1) We can run BFS for multiple blocks instead of one block in the previous milestone.
  2) BFS apply improves from planar x-y plane to x-y-z 3d set.
  3) We use disjoint set to connect all the blocks outside the membrane and then calculate the accurate volume of the cell.

2. The changes necessary to the code

  1) We divide the whole image into separate parts for blocks and in each block, every thread check BFS for one pixel. Then we can get the volume of one block by BFS and then sum them up later.
  2) Set dimGrid and dimBlock to (ceil(X/(float)BLOCK_SIZE),ceil(Y/(float)BLOCK_SIZE),ceil(Z/(float)BLOCK_SIZE)) and (BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE). And BFS would check the 6 dimensions of current pixel including up, down, front, back, left and right.
  3) We set parents of each block as their neighbors, which should also satisfy that they are connected without separate by membrane. And by checking all the blocks, build a disjoint set (up tree) for all the blocks outside the membrane. Then sum up the volume for blocks outside the membrane and we can get the complement of cell volume.

3. The expected impact on performance (why one expects a benefit, and how much—similar to the in-class analysis)

  1) By multiple blocks, we can deal with larger images with more pixels and also run more efficiently. Also we can decrease the global memory usage.
  2) Higher efficience than run by layers.
  3) Higher accuracy on the final results.

4. The actual (measured) benefit to performance

  1) Less global memory usage and global memory write/read time
  2) Decrease the running time
  3) Compared with the acutal result of 1570 fL, our result improved from ~2500 fL to ~1569 fL.

5. Any difficulties in realizing the implementation

  1) Because we will implement parallel method of BFS, each pixel aims to communicate each other and each block aims to communicate with each other. It is straight forward to use a large global memory. Hence, it is very challenging for us to use BFS without large global memory.
  2) Because the slices of the cell is not totally continuous, it is difficult for us to use the real 3D parallel implementation. It is difficult to determine the top and the bottom of the cell.
  3) Some threads are needed to wait for BFS process, so it is hard for us to decide when the block will end.

6. Any issues with variations in the correctness of the results (or a statement that the results are indistinguishable or identical, based on your comparison methodology from the earlier milestone)

  We use atomicCAS() to avoid access to change the same address at the same time to avoid various results within many trials, and also use atomicAdd() instead of simply adding for the same aim.

7. Explanation of any interesting aspects of performance

  1) Although our task is only a simple floodfill problem, our parallel BFS could actually apply to many general graph traversal tasks that are solvable by sequential BFS.
  2) Either large block size or small block size could boost the performance in some part. Thus, the block size could be adjusted according to the task's features.

8. Mention of any unexplained performance behavior (these are to be investigated for the final report)

  In our opinions, the expected result of cuda kernel is more accurate than we expected. There might be some error when run BFS with blocks but the result shows that it is very accurate. And we cannot explain why the answer is so accurate.