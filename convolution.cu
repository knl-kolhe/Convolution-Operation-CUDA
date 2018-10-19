#include "cuda_runtime.h"
#include <stdio.h>

#include <time.h>

//#define SIZE  1000

using namespace std;

__global__ void Convolution1(int *a,int *filter,int *result,int size_a,int size_filter,int size_result)

{

        int i=threadIdx.x;

        if(i<(size_result*size_result))

        {

                for(int k=0;k<size_filter;k++)

                        for(int j=0;j<size_filter;j++)

                                result[i] += a[k*size_a+(i%size_result)*2*size_a+j];

        }

}



void Convolution2(int *a,int *filter,int *result,int size_a,int size_filter,int size_result)
{
        for(int i=0;i<size_result*size_result;i++)
        {
                for(int k=0;k<size_filter;k++)

                        for(int j=0;j<size_filter;j++)

                                result[i] += a[k*size_a+(i%size_result)*2*size_a+j];
        }
}




int main()
{

        int *a,*filter,*result,*result_serial;

        int size_a,size_filter,size_result;

        clock_t t;

        double time_taken;



        x: printf("\n Enter size of array:");

        scanf("%d",&size_a);

        printf("\n Enter size of filter:");

        scanf("%d",&size_filter);

        if(size_a%2==0||size_filter%2==0)

        {

                printf("\n Enter odd numbers for sizes.");

                goto x;

        }

        if((size_a-size_filter)<0)

        {
                printf("\n Enter larger matrix size or smaller filter size.");

                goto x;

        }

        size_result=(size_a-size_filter)/2 +1;

        printf("Size of Matrix after Convolution with stride = (2) will be: %d \n",size_result);



        cudaMallocManaged(&a,size_a*size_a*sizeof(int));

        cudaMallocManaged(&filter,size_filter*size_filter*sizeof(int));

        cudaMallocManaged(&result,size_result*size_result*sizeof(int));

        cudaMallocManaged(&result_serial,size_result*size_result*sizeof(int));

        srand(0);



        for(int i=0;i<size_a*size_a;i++)

        {

                a[i]=rand()%100;

        }

        for(int i=0;i<size_filter*size_filter;i++)

        {

                filter[i]=rand()%100;

        }
        for(int i=0;i<size_result*size_result;i++)

        {

                result[i]=0;
                result_serial[i]=0;

        }


        t=clock();

        Convolution1<<<1,size_result*size_result>>>(a,filter,result,size_a,size_filter,size_result);

        cudaDeviceSynchronize();

        t=clock()-t;

        time_taken=((double)t)/CLOCKS_PER_SEC;

        printf("Time for Convolution with %d threads: %f \n",size_result*size_result,time_taken);





        t=clock();

        Convolution2(a,filter,result_serial,size_a,size_filter,size_result);

        t=clock()-t;

        time_taken=((double)t)/CLOCKS_PER_SEC;

        printf("Time for Convolution using serial:%f \n",time_taken);

        cudaFree(a);

        cudaFree(filter);

        cudaFree(result);

        cudaFree(result_serial);

        return 0;

}




/***********************OUTPUT*************************
[user10@linux-teslagpu ~]$ ./a.out

 Enter size of array:10001

 Enter size of filter:3
Size of Matrix after Convolution with stride = (2) will be: 5000 
Time for Convolution with 25000000 threads: 0.000000 
Time for Convolution using serial:1.990000 
******************************************************/
