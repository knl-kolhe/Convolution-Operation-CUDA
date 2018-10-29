#include "cuda_runtime.h"
#include <stdio.h>

#include <time.h>

using namespace std;

//without using atomic add function
__global__ void Convolution1(int *a,int *filter,int *result,int size_a,int size_filter,int size_result)

{

        int i=blockIdx.x;
		int j=blockIdx.y;

        if(i<size_result||j<size_result)
        {
                for(int k=0;k<size_filter;k++)
                        for(int l=0;l<size_filter;l++)
                                result[i*size_result+j] += filter[k*size_filter+l]*a[(2*i+k)*size_a+2*j+l];
		}

}

//using atomic add function
__global__ void Convolution3(int *a,int *filter,int *result,int size_a,int size_filter,int size_result)
{
	int i=blockIdx.x;
	int j=blockIdx.y;
	int k=threadIdx.x;
	int l=threadIdx.y;
	
    if(i<size_result||j<size_result||k<size_filter||l<size_filter)
    {
        result[i*size_result+j] = atomicAdd(&result[i*size_result+j],filter[k*size_filter+l]*a[(2*i+k)*size_a+2*j+l]);
	}
}

//serial function
void Convolution2(int *a,int *filter,int *result,int size_a,int size_filter,int size_result)
{
        for(int i=0;i<size_result;i++)
        {
                for(int j=0;j<size_result;j++)
					for(int k=0;k<size_filter;k++)
						for(int l=0;l<size_filter;l++)
                                result[i*size_result+j] += filter[k*size_filter+l]*a[(2*i+k)*size_a+2*j+l];
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
				//printf("Enter a[%d]",i);
				//scanf("%d",&a[i]);
        }

        for(int i=0;i<size_filter*size_filter;i++)
        {
                filter[i]=rand()%100;
				//printf("Enter filter[%d]",i);
				//scanf("%d",&filter[i]);
        }
        for(int i=0;i<size_result*size_result;i++)

        {

                result[i]=0;
                result_serial[i]=0;

        }

		dim3 res(size_result,size_result);
		dim3 fil(size_filter,size_filter);
        t=clock();

        Convolution1<<<res,1>>>(a,filter,result,size_a,size_filter,size_result);

        cudaDeviceSynchronize();

        t=clock()-t;

        time_taken=((double)t)/CLOCKS_PER_SEC;

        printf("Time for Convolution with %d threads: %f \n",size_result*size_result,time_taken);


		t=clock();

        Convolution3<<<res,fil>>>(a,filter,result,size_a,size_filter,size_result);

        cudaDeviceSynchronize();

        t=clock()-t;

        time_taken=((double)t)/CLOCKS_PER_SEC;

        printf("Time for Convolution with %d x %d threads: %f \n",size_result*size_result,size_filter*size_filter,time_taken);


        t=clock();

        Convolution2(a,filter,result_serial,size_a,size_filter,size_result);

        t=clock()-t;

        time_taken=((double)t)/CLOCKS_PER_SEC;

        printf("Time for Convolution using serial:%f \n",time_taken);
		
		/*
		for(int i=0;i<size_result*size_result;i++)
		{
			printf("\nresult[%d]=%d \nresult_serial[%d]=%d",i,result[i],i,result_serial[i]);
		}
		*/

        cudaFree(a);

        cudaFree(filter);

        cudaFree(result);

        cudaFree(result_serial);

        return 0;

}
