# Convolution-Operation-CUDA
A code to perform convolution operation on a matrix in cuda

## Output
On GTX 1050, i7 7700 4 core

Enter size of array:10001

Enter size of filter:3

Size of Matrix after Convolution with stride = (2) will be: 5000

Time for Convolution with 25000000 threads: 2.246000

Time for Convolution using serial:3.104000

Time for Convolution with 25000000 x 9 threads: 0.131000

Sanity check:

result[0]=12792
result_serial[0]=12792
resultoptimal[0]=12792

result[1]=14060
result_serial[1]=14060
resultoptimal[1]=14060

result[2]=20138
result_serial[2]=20138
resultoptimal[2]=20138

result[3]=19328
result_serial[3]=19328
resultoptimal[3]=19328

result[4]=20288
result_serial[4]=20288
resultoptimal[4]=20288

result[5]=14252
result_serial[5]=14252
resultoptimal[5]=14252

result[6]=16804
result_serial[6]=16804
resultoptimal[6]=16804

result[7]=20854
result_serial[7]=20854
resultoptimal[7]=20854

result[8]=24886
result_serial[8]=24886
resultoptimal[8]=24886

result[9]=21838
result_serial[9]=21838
resultoptimal[9]=21838


### A small note:
This is not the most efficient form of convolution. I have learned now (in 2019 at graduate school) that using a polynomial multiplication algorithm using Fast Fourier Transform will be much more efficient. But at the time, for the scope of this project, we just had to parallelize some operation using CUDA C.
