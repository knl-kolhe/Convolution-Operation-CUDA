#include "cuda_runtime.h"
#include <stdio.h>

#include <time.h>

//#define SIZE  1000

using namespace std;

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

__global__ void Convolution3(int *a,int *filter,int *result,int size_a,int size_filter,int size_result)
{
	int i=blockIdx.x;
	int j=blockIdx.y;
	int k=threadIdx.x;
	int l=threadIdx.y;
	
    if(i<size_result||j<size_result||k<size_filter||l<size_filter)
    {
        atomicAdd(&result[i*size_result+j],filter[k*size_filter+l]*a[(2*i+k)*size_a+2*j+l]);
	}
}


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

        int *a,*filter,*result,*result_serial,*result_optimal;

        int size_a,size_filter,size_result;

        clock_t t;

        double time_taken;



        x: printf("\nEnter size of array:");

        scanf("%d",&size_a);

        printf("\nEnter size of filter:");

        scanf("%d",&size_filter);

        if(size_a%2==0||size_filter%2==0)

        {

                printf("\nEnter odd numbers for sizes.");

                goto x;

        }

        if((size_a-size_filter)<0)

        {
                printf("\nEnter larger matrix size or smaller filter size.");

                goto x;

        }

        size_result=(size_a-size_filter)/2 +1;

        printf("Size of Matrix after Convolution with stride = (2) will be: %d \n",size_result);



        cudaMallocManaged(&a,size_a*size_a*sizeof(int));

        cudaMallocManaged(&filter,size_filter*size_filter*sizeof(int));

        cudaMallocManaged(&result,size_result*size_result*sizeof(int));
		
		cudaMallocManaged(&result_optimal,size_result*size_result*sizeof(int));

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
				result_optimal[i]=0;

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

        Convolution3<<<res,fil>>>(a,filter,result_optimal,size_a,size_filter,size_result);

        cudaDeviceSynchronize();

        t=clock()-t;

        time_taken=((double)t)/CLOCKS_PER_SEC;

        printf("Time for Convolution with %d x %d threads: %f \n",size_result*size_result,size_filter*size_filter,time_taken);


        t=clock();

        Convolution2(a,filter,result_serial,size_a,size_filter,size_result);

        t=clock()-t;

        time_taken=((double)t)/CLOCKS_PER_SEC;

        printf("Time for Convolution using serial:%f \n",time_taken);
		
		
		printf("\nSanity Check:");
		if(size_filter*size_filter>11)
			for(int i=0;i<10;i++)
			{
				printf("\nresult[%d]=%d \nresult_serial[%d]=%d \nresult_optimal[%d]=%d\n",i,result[i],i,result_serial[i],i,result_optimal[i]);
			}
		else
			for(int i=0;i<size_filter*size_filter;i++)
			{
				printf("\nresult[%d]=%d \nresult_serial[%d]=%d \nresult_optimal[%d]=%d\n",i,result[i],i,result_serial[i],i,result_optimal[i]);
			}
		

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

On Gtx 1050:
E:\!KUNAL\MIT\BE\HPC\MiniProject>a.exe
Enter size of array:10001

Enter size of filter:3
Size of Matrix after Convolution with stride = (2) will be: 5000
Time for Convolution with 25000000 threads: 2.210000
Time for Convolution with 25000000 x 9 threads: 0.134000
Time for Convolution using serial:3.210000

Sanity Check:
result[0]=12792
result_serial[0]=12792
result_optimal[0]=12792

result[1]=14060
result_serial[1]=14060
result_optimal[1]=14060

result[2]=20138
result_serial[2]=20138
result_optimal[2]=20138

result[3]=19328
result_serial[3]=19328
result_optimal[3]=19328

result[4]=20288
result_serial[4]=20288
result_optimal[4]=20288

result[5]=14252
result_serial[5]=14252
result_optimal[5]=14252

result[6]=16804
result_serial[6]=16804
result_optimal[6]=16804

result[7]=20854
result_serial[7]=20854
result_optimal[7]=20854

result[8]=24886
result_serial[8]=24886
result_optimal[8]=24886

******************************************************/
