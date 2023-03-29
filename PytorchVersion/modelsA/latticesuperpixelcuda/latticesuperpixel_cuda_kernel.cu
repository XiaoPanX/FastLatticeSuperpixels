#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

 template <typename scalar_t>
 __global__ void Assign_labels(const int nthreads,
  scalar_t* labels,  scalar_t* parent, scalar_t* nr_partitions,const scalar_t* features,  scalar_t* block_features,
   scalar_t* T, int nr_seeds_h,  int nr_seeds_w, const int height,
  const int width,  int step_h,  int step_w,
  const int nr_seeds_h1, const int nr_seeds_w1,  int cur_level, const int levels, const int in_channels) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
      const int spatial_dim = height * width;
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      int y_index = s / width;
      int x_index = s % width;
      int label_x=floor((float)x_index/step_w);
      if(label_x>=nr_seeds_w){
        label_x = nr_seeds_w-1;
      }
      int label_y=floor((float)y_index/step_h);
      if(label_y >= nr_seeds_h){
        label_y = nr_seeds_h-1;
      }
      int label_index=(n*levels+cur_level)*spatial_dim+s;
      labels[label_index]=label_y*nr_seeds_w + label_x;

      if(cur_level==0){
        int p_index = (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1+label_y*nr_seeds_w + label_x;//*1为申请空间大小，不带1为当前大小
        nr_partitions[p_index] = 1;
        atomicAdd(T+p_index,1);
        //T[p_index]=T[p_index]+1;
        for (int k=0;k<in_channels;k++)
        {
         int block_feature_index=(n*(levels*in_channels)+cur_level*in_channels+k)*nr_seeds_h1*nr_seeds_w1+labels[label_index];
         int feature_index=(n*in_channels+k)*spatial_dim+s;
         //caffe_gpu_atomic_add(features[feature_index],block_features+block_feature_index);
         atomicAdd(block_features+block_feature_index,static_cast<scalar_t>(features[feature_index]));
        }
      }

      if(cur_level>=1){
       int p_index = (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1+label_y*nr_seeds_w + label_x;//*1为申请空间大小，不带1为当前大小
       nr_partitions[p_index] = 0;
       int child_index=(n*levels+cur_level-1)*spatial_dim+s;
       int label_child= labels[child_index];
       int parent_index= (n*levels+cur_level-1)*nr_seeds_h1*nr_seeds_w1+label_child;
       parent[parent_index]=labels[label_index];//parent 是按行优先存的，nr_seeds_h1*nr_seeds_w1先存满一行后再另一行
       }
    //   __syncthreads();
  }
}
template <typename scalar_t>
__global__ void Compute_block_Feature(const int nthreads,
  scalar_t* labels, scalar_t* parent, scalar_t* nr_partitions,  scalar_t* block_features,
    scalar_t* T,const int nr_seeds_h1, const int nr_seeds_w1, int nr_seeds_h,  int nr_seeds_w,
    int cur_level, const int levels, const int in_channels) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nthreads) {
    const int spatial_dim = nr_seeds_h * nr_seeds_w;
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int space_dim=nr_seeds_h1*nr_seeds_w1;
    int child_index=(n*levels+cur_level)*space_dim+s;
    int parent_id=parent[child_index];
    for (int k=0;k<in_channels;k++)
    {
       int block_child_index=((n*levels+cur_level)*in_channels+k)*space_dim+s;
       int block_parent_index=((n*levels+cur_level+1)*in_channels+k)*space_dim+parent_id;
       atomicAdd(block_features+block_parent_index,static_cast<scalar_t>(block_features[block_child_index]));
       //caffe_gpu_atomic_add(block_features[block_child_index],block_features+block_parent_index);
    }
    int parent_index=(n*levels+cur_level+1)*space_dim+parent_id;
    atomicAdd(T+parent_index,static_cast<scalar_t>(T[child_index]));
    //caffe_gpu_atomic_add(T[child_index],T+parent_index);
    //caffe_gpu_atomic_add(1,nr_partitions+parent_index);
    atomicAdd(nr_partitions+parent_index,static_cast<scalar_t>(1));
 }
}

template <typename scalar_t>
__device__ bool check_split11(int a11, int a12, int a13, int a21, int a22,
                       int a23, int a31, int a32, int a33)
{

		//	if ((a22 != a21) && (a22 == a12) && (a22 == a32)) return true;
			if ((a22 != a11) && (a22 == a12) && (a22 == a21)) return true;
			if ((a22 != a31) && (a22 == a32) && (a22 == a21)) return true;

	return false;
}
template <typename scalar_t>
__device__ bool check_split10(int a11, int a12, int a13, int a21, int a22,
                       int a23, int a31, int a32, int a33)
{

		//	if ((a22 != a23) && (a22 == a12) && (a22 == a32)) return true;
			if ((a22 != a13) && (a22 == a12) && (a22 == a23)) return true;
			if ((a22 != a33) && (a22 == a32) && (a22 == a23)) return true;

	return false;
}
template <typename scalar_t>
__device__ bool check_split01(int a11, int a12, int a13, int a21, int a22,
                       int a23, int a31, int a32, int a33)
{
		//	if ((a22 != a12) && (a22 == a21) && (a22 == a23)) return true;
			if ((a22 != a11) && (a22 == a21) && (a22 == a12)) return true;
			if ((a22 != a13) && (a22 == a23) && (a22 == a12)) return true;

	return false;
}

template <typename scalar_t>
__device__ bool check_split00(int a11, int a12, int a13, int a21, int a22,
                       int a23, int a31, int a32, int a33)
{

		//	if ((a22 != a32) && (a22 == a21) && (a22 == a23)) return true;
			if ((a22 != a31) && (a22 == a21) && (a22 == a32)) return true;
			if ((a22 != a33) && (a22 == a23) && (a22 == a32)) return true;

	return false;
}
template <typename scalar_t>
__device__ scalar_t get_block_dis_CE(scalar_t* block_features,scalar_t* T,
const int nr_seeds_h1, const int nr_seeds_w1,const int levels,int level1,int level2,
const int in_channels,int label1,int label2,const int n){
  scalar_t dist =0.0;
  int T_index1 = (n*levels+level1)*nr_seeds_h1*nr_seeds_w1+label1;
  int T_index2 = (n*levels+level2)*nr_seeds_h1*nr_seeds_w1+label2;
  for (int k=0;k<in_channels;k++)
    {
       int block_index1 = ((n*levels+level1)*in_channels+k)*nr_seeds_h1*nr_seeds_w1+label1;
       int block_index2 = ((n*levels+level2)*in_channels+k)*nr_seeds_h1*nr_seeds_w1+label2;
      // dist += pow((block_features[block_index1]/(float)T[T_index1]-block_features[block_index2]/(float)T[T_index2]), 2);
      dist += abs(block_features[block_index1]/(float)T[T_index1]-block_features[block_index2]/(float)T[T_index2]);
    }
  return dist;
}

template <typename scalar_t>
__device__ scalar_t get_block_dis_CE1(scalar_t* block_features,scalar_t* T,
const int nr_seeds_h1, const int nr_seeds_w1,const int levels,int top_level,int cur_level,
const int in_channels,int label1,int label2,const int n){
  scalar_t dist =0.0;
  int T_index1 = (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1+label2;
  int T_index2 = (n*levels+top_level)*nr_seeds_h1*nr_seeds_w1+label1;
  for (int k=0;k<in_channels;k++)
    {
       int block_index1 = ((n*levels+cur_level)*in_channels+k)*nr_seeds_h1*nr_seeds_w1+label2;
       int block_index2 = ((n*levels+top_level)*in_channels+k)*nr_seeds_h1*nr_seeds_w1+label1;
       scalar_t new_feature=block_features[block_index2]-block_features[block_index1];
      // dist += pow(new_feature/(float)(T[T_index2]-1)-(block_features[block_index1]/(float)T[T_index1]), 2);
      dist += abs(new_feature/(float)(T[T_index2]-1)-(block_features[block_index1]/(float)T[T_index1]));
    }
  return dist;
}

template <typename scalar_t>
__device__ scalar_t get_pixels_dis_CE(scalar_t* block_features,scalar_t* T,const scalar_t* features,const int levels,
const int nr_seeds_h1, const int nr_seeds_w1, const int top_level,const int height, const int width,
const int in_channels,int label,int ori_superP_size,int x, int y,int n){
    scalar_t dist =0.0;
    int T_index = (n*levels+top_level)*nr_seeds_h1*nr_seeds_w1+label;
    scalar_t dis_size=0.0;
   if(ori_superP_size>1)
   {
     dis_size=scalar_t(T[T_index]-ori_superP_size)/scalar_t(ori_superP_size)*5.0;
   }

    for (int k=0;k<in_channels;k++)
    {
       int block_index = ((n*levels+top_level)*in_channels+k)*nr_seeds_h1*nr_seeds_w1+label;
       int pixel_index = (n*in_channels+k)*height*width+y*width+x;
       //dist += pow((block_features[block_index]/(float)T[T_index]-features[pixel_index]), 2);
       dist += abs(block_features[block_index]/(float)T[T_index]-features[pixel_index]);
    }
    dist=dist+dis_size;
 //  dist =110.0/25.0*10.0;
  return dist;
}


template <typename scalar_t>
__global__ void  update_pixels(const int nthreads,scalar_t* block_features,scalar_t* T, const scalar_t* features, scalar_t* labels, scalar_t* pixel_Tag,
const int nr_seeds_h1, const int nr_seeds_w1,const int levels,const int top_level,const int height, const int width,
const int in_channels){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index < nthreads) {
  const int spatial_dim = height * width;
  const int n = index / spatial_dim;
  const int s = index % spatial_dim;
  int label_index=(n*levels+top_level)*spatial_dim+s;
  int label_old = labels[label_index];
  int sub_spatial_dim = nr_seeds_h1*nr_seeds_w1;
  int T_old_index = (n*levels+top_level)*sub_spatial_dim+label_old;
  //caffe_gpu_atomic_add(-1,T+T_old_index);
  atomicAdd(T+T_old_index,static_cast<scalar_t>(-1));
//  T[T_old_index]--;
  int label_index_new=label_index+pixel_Tag[index];
  int label_new = labels[label_index_new];
  int T_new_index = (n*levels+top_level)*sub_spatial_dim+label_new;
  atomicAdd(T+T_new_index,static_cast<scalar_t>(1));
//  T[T_new_index]++;
  for (int k=0;k<in_channels;k++)
    {
       int block_old_index=((n*levels+top_level)*in_channels+k)*sub_spatial_dim+label_old;
       int pixel_index = (n*in_channels+k)*spatial_dim+s;
        atomicAdd(block_features+block_old_index,static_cast<scalar_t>(-features[pixel_index]));
      //  caffe_gpu_atomic_add((scalar_t)-features[pixel_index],block_features+block_old_index);
     //  block_features[block_old_index]=block_features[block_old_index]-features[pixel_index];
       int block_new_index = ((n*levels+top_level)*in_channels+k)*sub_spatial_dim+label_new;
     //  block_features[block_new_index]=block_features[block_new_index]+features[pixel_index];
       //caffe_gpu_atomic_add(features[pixel_index],block_features+block_new_index);
       atomicAdd(block_features+block_new_index,static_cast<scalar_t>(features[pixel_index]));
    }
  labels[label_index] = label_new;
  }
}
template <typename scalar_t>
__global__ void update_labels(const int nthreads,scalar_t* labels, scalar_t* parent,
   const int height, const int width,const int levels, int cur_level,int seeds_top_level,
   const int nr_seeds_h1, const int nr_seeds_w1){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index<nthreads) {
       const int spatial_dim = height * width;
       const int n = index / spatial_dim;
       const int s = index % spatial_dim;
       int label_index=(n*levels+cur_level)*spatial_dim+s;
       int parent_index=(n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1+labels[label_index];
       int top_index=(n*levels+seeds_top_level)*spatial_dim+s;
       labels[top_index]=parent[parent_index];
    }
}
template <typename scalar_t>
__global__ void update_pixels_X_right(const int nthreads,
  scalar_t* labels, scalar_t* parent, scalar_t* pixel_Tag,const scalar_t* features,  scalar_t* block_features, scalar_t* T,scalar_t* dis,
  const int nr_seeds_h1, const int nr_seeds_w1, const int height, const int width,
   int shift_x,int shift_y,   const int levels, const int seeds_top_level, int ori_superP_size,
  const int in_channels) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index<nthreads) {
        int spatial_dim = height * width;
        int s_width = ceil(float(width)/3.0);
        int s_height = ceil(float(height)/3.0);
        int spatial_dim1 = s_width * s_height;
        int n = index/spatial_dim1;
        int s = index%(spatial_dim1);

        int x = (s%s_width)*3+shift_x;
        int y = (s/s_width)*3+shift_y;
        //int x = (s%(width/3))*3+shift_x;
        // int y = (s/(width/3))*3+shift_y;
        int sublabel=(y)*width+(x);
        int top_level_index=(n*levels+seeds_top_level)*spatial_dim;

		if(x>0&&x<(width-1)&&y>0&&y<(height-1)){
         int labelA = labels[top_level_index+sublabel];
		 int labelB = labels[top_level_index+sublabel+1];
         if (labelA != labelB)
		 {
            int a11 = labels[top_level_index+sublabel-width-1];
            int a12 = labels[top_level_index+sublabel-width];
          //  int a13 = labels[top_level_index+sublabel-width+1];

            int a21 = labels[top_level_index+sublabel-1];
            int a22 = labels[top_level_index+sublabel];
          //  int a23 = labels[top_level_index+sublabel+1];

            int a31 = labels[top_level_index+sublabel+width-1];
            int a32 = labels[top_level_index+sublabel+width];
          //  int a33 = labels[top_level_index+sublabel+width+1];
            bool isSplit = false;
            if ((a22 != a11) && (a22 == a12) && (a22 == a21)) isSplit = true;
			if ((a22 != a31) && (a22 == a32) && (a22 == a21)) isSplit = true;
            if (a21==labelA&&!isSplit)
            {
                scalar_t disA=get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1, seeds_top_level,
                                            height, width, in_channels,labelA,ori_superP_size,x, y,n);
                scalar_t disB=get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1, seeds_top_level,
                                            height, width, in_channels,labelB,ori_superP_size,x, y,n);
                int curyup=y-1;
                int upsublabel=curyup*width+x;
                int uplabelA=labels[top_level_index+upsublabel];
                int uplabelB=labels[top_level_index+upsublabel+1];
                int curlabelA=labelA;
              //  int curlabelB=labelB;
                int uplabelAl=labels[top_level_index+upsublabel-1];

                bool bedone=true;
                int upNum=0;
                while(uplabelA!=curlabelA&&uplabelA!=uplabelB&&curyup>=0)
                {
                    if(uplabelA!=uplabelAl)
                    {
                        bedone = false;
                        break;
                    }
                    if(curyup>0)
                    {
                        int upuplabelA=labels[top_level_index+upsublabel-width];
                        int upuplabelAl=labels[top_level_index+upsublabel-width-1];
                        if(upuplabelA==uplabelA&&upuplabelAl!=uplabelA)
                        {
                            bedone = false;
                            break;
                        }
                    }
                    disA=disA+get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1,  seeds_top_level,
                                               height, width, in_channels,uplabelA,ori_superP_size,x, curyup,n);
                    disB=disB+get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1,  seeds_top_level,
                                               height, width, in_channels,uplabelB,ori_superP_size,x, curyup,n);
                    curlabelA=uplabelA;
                  //  curlabelB=uplabelB;
                    upNum++;
                    curyup--;
                    if(curyup<0) break;
                    upsublabel=upsublabel-width;
                    uplabelA=labels[top_level_index+upsublabel];
                    uplabelB=labels[top_level_index+upsublabel+1];
                    uplabelAl=labels[top_level_index+upsublabel-1];

                }
                int downNum=0;
                if(bedone==true)
                {
                    int curydown=y+1;
                    int downsublabel=curydown*width+x;
                    int downlabelA=labels[top_level_index+downsublabel];
                    int downlabelB=labels[top_level_index+downsublabel+1];
                    int downlabelAl=labels[top_level_index+downsublabel-1];
                    curlabelA=labelA;
                 //   curlabelB=labelB;
                    while(downlabelA!=curlabelA&&downlabelA!=downlabelB&&curydown<height)
                    {
                        if(downlabelA!=downlabelAl)
                        {
                            bedone = false;
                            break;
                        }
                        if(curydown<height-1)
                        {
                            int downdownlabelA=labels[top_level_index+downsublabel+width];
                            int downdownlabelAl=labels[top_level_index+downsublabel+width-1];
                            if(downdownlabelA==downlabelA&&downdownlabelAl!=downlabelA)
                            {
                                bedone = false;
                                break;
                            }
                        }

                        disA=disA+get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1,  seeds_top_level,
                                                    height, width, in_channels,downlabelA,ori_superP_size,x, curydown,n);
                        disB=disB+get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1,  seeds_top_level,
                                                    height, width, in_channels,downlabelB,ori_superP_size,x, curydown,n);
                        downNum++;
                        curlabelA=downlabelA;
                       // curlabelB=downlabelB;
                        curydown++;
                        if(curydown>=height) break;
                        downsublabel=downsublabel+width;
                        downlabelA=labels[top_level_index+downsublabel];
                        downlabelB=labels[top_level_index+downsublabel+1];
                        downlabelAl=labels[top_level_index+downsublabel-1];
                    }
                }

                if(disA>disB&&bedone==true)
                {
                    //update_pixels(block_features,T, features, labels,
                    //             nr_seeds_h1, nr_seeds_w1,levels,top_level,height,  width,
                    //             in_channels,labelB, x, y, n);
                    pixel_Tag[spatial_dim*n+sublabel]=1;
                    dis[spatial_dim*n+sublabel]=1;
                    for(int i=1;i<=upNum;i++)
                    {
                        //uplabelB=labels[top_level_index+(y-i)*width+x+1];
                       // update_pixels(block_features,T, features, labels,
                       //          nr_seeds_h1, nr_seeds_w1,levels,top_level,height,  width,
                        //         in_channels,uplabelB, x, y-i, n);
                        pixel_Tag[spatial_dim*n+(y-i)*width+x]=1;
                        dis[spatial_dim*n+(y-i)*width+x]=1;
                    }
                    for(int i=1;i<=downNum;i++)
                    {
                       // downlabelB=labels[top_level_index+(y+i)*width+x+1];
                      //  update_pixels(block_features,T, features, labels,
                       //          nr_seeds_h1, nr_seeds_w1,levels,top_level,height,  width,
                       //          in_channels,downlabelB, x, y+i, n);
                       pixel_Tag[spatial_dim*n+(y+i)*width+x]=1;
                       dis[spatial_dim*n+(y+i)*width+x]=1;
                    }
                   // done=true;
                }
            }
         }
    }
  }
}
template <typename scalar_t>
__global__ void update_pixels_X_left(const int nthreads,
  scalar_t* labels, scalar_t* parent, scalar_t* pixel_Tag,const scalar_t* features,
   scalar_t* block_features, scalar_t* T,  const int nr_seeds_h1, const int nr_seeds_w1,
   const int height, const int width, int shift_x,int shift_y,  const int levels,
   const int seeds_top_level,int ori_superP_size, const int in_channels) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index<nthreads) {
        int spatial_dim = height * width;
        int s_width = ceil(float(width)/3.0);
        int s_height = ceil(float(height)/3.0);
        int spatial_dim1 = s_width * s_height;
        int n = index/spatial_dim1;
        int s = index%(spatial_dim1);
        int x = (s%s_width)*3+shift_x;
        int y = (s/s_width)*3+shift_y;
        int sublabel=(y)*width+(x);
        int top_level_index=(n*levels+seeds_top_level)*spatial_dim;
        //pixel_Tag[sublabel]=-1;
		if(x>0&&x<(width-1)&&y>0&&y<(height-1)){
        int labelA = labels[top_level_index+index-1];
		int labelB = labels[top_level_index+index];
         if (labelA != labelB)
		 {
         //   pixel_Tag[sublabel]=-1;
        //    int a11 = labels[top_level_index+sublabel-width-1];
            int a12 = labels[top_level_index+sublabel-width];
            int a13 = labels[top_level_index+sublabel-width+1];

       //     int a21 = labels[top_level_index+sublabel-1];
            int a22 = labels[top_level_index+sublabel];
            int a23 = labels[top_level_index+sublabel+1];

        //    int a31 = labels[top_level_index+sublabel+width-1];
            int a32 = labels[top_level_index+sublabel+width];
            int a33 = labels[top_level_index+sublabel+width+1];
            bool isSplit = false;
            if ((a22 != a13) && (a22 == a12) && (a22 == a23)) isSplit = true;
			if ((a22 != a33) && (a22 == a32) && (a22 == a23)) isSplit = true;

            if (a23==labelB&&!isSplit)//?
            {
             //   scalar_t disA=0.0;
             //   scalar_t disB=0.0;
               // int T_index = (n*levels+seeds_top_level)*nr_seeds_h1*nr_seeds_w1+labelA;
               // scalar_t dis_size=scalar_t(T[T_index]-ori_superP_size)/scalar_t(ori_superP_size)*10.0;
                scalar_t disA=get_pixels_dis_CE(block_features, T,features, levels,nr_seeds_h1, nr_seeds_w1,  seeds_top_level,
                                            height, width, in_channels,labelA,ori_superP_size, x, y,n);
                scalar_t disB=get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1,  seeds_top_level,
                                            height, width, in_channels,labelB, ori_superP_size, x, y,n);

                int curyup=y-1;
                int upsublabel=curyup*width+x;
                int uplabelA=labels[top_level_index+upsublabel-1];
                int uplabelB=labels[top_level_index+upsublabel];
             //   int curlabelA=labelA;
                int curlabelB=labelB;
                int uplabelBR=labels[top_level_index+upsublabel+1];
                bool bedone=true;
                int upNum=0;
                int downNum=0;
                while(uplabelB!=curlabelB&&uplabelA!=uplabelB&&curyup>=0)
                {
                    if(uplabelB!=uplabelBR)
                    {
                        bedone = false;
                        break;
                    }
                    if(curyup>0)
                    {
                        int upuplabelB=labels[top_level_index+upsublabel-width];
                        int upuplabelBR=labels[top_level_index+upsublabel-width+1];
                        if(upuplabelB==uplabelB&&upuplabelBR!=uplabelB)
                        {
                            bedone = false;
                            break;
                        }
                    }
                    disA=disA+get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1, seeds_top_level,
                                               height, width, in_channels,uplabelA,ori_superP_size, x, curyup,n);
                    disB=disB+get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1, seeds_top_level,
                                               height, width, in_channels,uplabelB,ori_superP_size, x, curyup,n);
                //    curlabelA=uplabelA;
                    curlabelB=uplabelB;
                    upNum++;
                    curyup=curyup-1;
                    if(curyup<0) break;
                    upsublabel=upsublabel-width;
                    uplabelA=labels[top_level_index+upsublabel-1];
                    uplabelB=labels[top_level_index+upsublabel];
                    uplabelBR=labels[top_level_index+upsublabel+1];

                }

                if(bedone==true)
                {
                    int curydown=y+1;
                    int downsublabel=curydown*width+x;
                    int downlabelA=labels[top_level_index+downsublabel-1];
                    int downlabelB=labels[top_level_index+downsublabel];
                    int downlabelBR=labels[top_level_index+downsublabel+1];
                 //   curlabelA=labelA;
                    curlabelB=labelB;
                    while(downlabelB!=curlabelB&&downlabelA!=downlabelB&&curydown<height)
                    {
                        if(downlabelB!=downlabelBR)
                        {
                            bedone = false;
                            break;
                        }
                        if(curydown<(height-1))
                        {
                            int downdownlabelB=labels[top_level_index+downsublabel+width];
                            int downdownlabelBR=labels[top_level_index+downsublabel+width+1];
                            if(downdownlabelB==downlabelB&&downdownlabelBR!=downlabelB)
                            {
                                bedone = false;
                                break;
                            }
                        }
                        disA=disA+get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1,  seeds_top_level,
                                                    height, width, in_channels,downlabelA, ori_superP_size, x, curydown,n);
                        disB=disB+get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1,  seeds_top_level,
                                                    height, width, in_channels,downlabelB, ori_superP_size, x, curydown,n);
                        downNum++;
                  //      curlabelA=downlabelA;
                        curlabelB=downlabelB;
                        curydown=curydown+1;
                        if(curydown>=height) break;
                        downsublabel=downsublabel+width;
                        downlabelA=labels[top_level_index+downsublabel-1];
                        downlabelB=labels[top_level_index+downsublabel];
                        downlabelBR=labels[top_level_index+downsublabel+1];
                    }
                }

               //  pixel_Tag[0]=-1;
                if (disB>disA&&bedone)
                {
                 //   pixel_Tag[0]=-1;
                    pixel_Tag[spatial_dim*n+sublabel]=-1;
                    for(int i=1;i<=upNum;i++)
                    {
                      pixel_Tag[spatial_dim*n+(y-i)*width+x]=-1;
                    }
                    for(int i=1;i<=downNum;i++)
                    {
                      pixel_Tag[spatial_dim*n+(y+i)*width+x]=-1;
                    }
                }
            }
		  }
		}
    }
  }
template <typename scalar_t>
__global__ void update_pixels_Y_down(const int nthreads,
  scalar_t* labels, scalar_t* parent, scalar_t* pixel_Tag,const scalar_t* features,  scalar_t* block_features,
  scalar_t* T, const int nr_seeds_h1, const int nr_seeds_w1, const int height, const int width,
   int shift_x,int shift_y,  const int levels,  const int seeds_top_level,int ori_superP_size,
  const int in_channels) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
     if(index < nthreads) {
        int spatial_dim = height * width;
        int s_width = ceil(float(width)/3.0);
        int s_height = ceil(float(height)/3.0);
        int spatial_dim1 = s_width * s_height;
        int n = index/spatial_dim1;
        int s = index%(spatial_dim1);
        //int s_width = ceil(float(width)/3.0);
        int x = (s%s_width)*3+shift_x;
        int y = (s/s_width)*3+shift_y;
        int sublabel=(y)*width+(x);
        int top_level_index=(n*levels+seeds_top_level)*spatial_dim;

		if(x>0&&x<(width-1)&&y>0&&y<(height-1)){
		    int labelA = labels[top_level_index+sublabel];
		    int labelB = labels[top_level_index+sublabel+width];
		    if (labelA != labelB){
		        int a11 = labels[top_level_index+sublabel-width-1];
                int a12 = labels[top_level_index+sublabel-width];
                int a13 = labels[top_level_index+sublabel-width+1];
                int a21 = labels[top_level_index+sublabel-1];
                int a22 = labels[top_level_index+sublabel];
                int a23 = labels[top_level_index+sublabel+1];
            //    int a31 = labels[top_level_index+sublabel+width-1];
             //   int a32 = labels[top_level_index+sublabel+width];
            //    int a33 = labels[top_level_index+sublabel+width+1];
          //  bool done=false;
             bool isSplit = false;
             if ((a22 != a11) && (a22 == a21) && (a22 == a12)) isSplit = true;
			 if ((a22 != a13) && (a22 == a23) && (a22 == a12)) isSplit = true;
			 if (a12==labelA&&!isSplit)
             {
                scalar_t disA=get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1,  seeds_top_level,
                                            height, width, in_channels,labelA,ori_superP_size, x, y,n);
                scalar_t disB=get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1,  seeds_top_level,
                                            height, width, in_channels,labelB,ori_superP_size, x, y,n);
                int curxleft=x-1;
                int leftsublabel=y*width+curxleft;
                int leftlabelA=labels[top_level_index+leftsublabel];
                int leftlabelB=labels[top_level_index+leftsublabel+width];
                int curlabelA=labelA;
             //   int curlabelB=labelB;
                int leftlabelAup=labels[top_level_index+leftsublabel-width];

                bool bedone=true;
                int leftNum=0;
                while(leftlabelA!=curlabelA&&leftlabelA!=leftlabelB&&curxleft>=0)
                {
                    if(leftlabelA!=leftlabelAup)
                    {
                        bedone = false;
                        break;
                    }
                    if(curxleft>0)
                    {
                        int leftleftlabelA=labels[top_level_index+leftsublabel-1];
                        int leftleftlabelAup=labels[top_level_index+leftsublabel-1-width];
                        if(leftleftlabelA==leftlabelA&&leftleftlabelAup!=leftlabelAup)
                        {
                            bedone = false;
                            break;
                        }
                    }
                    disA=disA+get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1,  seeds_top_level,
                                               height, width, in_channels,leftlabelA, ori_superP_size,curxleft, y,n);
                    disB=disB+get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1,  seeds_top_level,
                                               height, width, in_channels,leftlabelB, ori_superP_size, curxleft, y,n);

                    curlabelA=leftlabelA;
                //    curlabelB=leftlabelB;
                    leftNum++;
                    curxleft--;
                    if(curxleft<0) break;
                    leftsublabel=y*width+curxleft;
                    leftlabelA=labels[top_level_index+leftsublabel];
                    leftlabelB=labels[top_level_index+leftsublabel+width];
                    leftlabelAup=labels[top_level_index+leftsublabel-width];

                }
                int rightNum=0;
                if(bedone==true)
                {
                    int curxright=x+1;
                    int rightsublabel=y*width+curxright;
                    int rightlabelA=labels[top_level_index+rightsublabel];
                    int rightlabelB=labels[top_level_index+rightsublabel+width];
                    int rightlabelAup=labels[top_level_index+rightsublabel-width];
                    curlabelA=labelA;
                //    curlabelB=labelB;
                    while(rightlabelA!=curlabelA&&rightlabelA!=rightlabelB&&curxright<width)
                    {
                        if(rightlabelA!=rightlabelAup)
                        {
                            bedone = false;
                            break;
                        }
                        if(curxright<width-1)
                        {
                            int RrightlabelA=labels[top_level_index+rightsublabel+1];
                            int RrightlabelAup=labels[top_level_index+rightsublabel+1-width];
                            if(RrightlabelA==rightlabelA&&RrightlabelAup!=rightlabelA)
                            {
                                bedone = false;
                                break;
                            }
                        }
                        disA=disA+get_pixels_dis_CE(block_features, T,features, nr_seeds_h1, nr_seeds_w1, levels, seeds_top_level,
                                                   height, width, in_channels,rightlabelA,ori_superP_size, curxright, y,n);
                        disB=disB+get_pixels_dis_CE(block_features, T,features, nr_seeds_h1, nr_seeds_w1, levels, seeds_top_level,
                                                   height, width, in_channels,rightlabelB, ori_superP_size,curxright, y,n);
                        rightNum++;
                        curlabelA=rightlabelA;
                      //  curlabelB=rightlabelB;
                        curxright++;
                        if(curxright>=width) break;
                        rightsublabel=y*width+curxright;
                        rightlabelA=labels[top_level_index+rightsublabel];
                        rightlabelB=labels[top_level_index+rightsublabel+width];
                        rightlabelAup=labels[top_level_index+rightsublabel-width];
                    }
                }
                if (disA>disB&&bedone)
                {

                  //  update_pixels(block_features,T, features, labels,
                  //                   nr_seeds_h1, nr_seeds_w1,levels,top_level,height,  width,
                  //                   in_channels,labelB, x, y, n);
                     pixel_Tag[spatial_dim*n+sublabel]=width;
                    for(int i=1;i<=leftNum;i++)
                    {
                    //    leftlabelB=labels[top_level_index+(y+1)*width+x-i];
                    //    update_pixels(block_features,T, features, labels,
                    //                 nr_seeds_h1, nr_seeds_w1,levels,top_level,height,  width,
                    //                 in_channels,leftlabelB, x-i, y, n);
                          pixel_Tag[spatial_dim*n+y*width+x-i]=width;
                    }
                    for(int i=1;i<=rightNum;i++)
                    {
                      //  rightlabelB=labels[top_level_index+(y+1)*width+x+i];
                      //  update_pixels(block_features,T, features, labels,
                      //               nr_seeds_h1, nr_seeds_w1,levels,top_level,height,  width,
                      //               in_channels,rightlabelB, x+i, y, n);
                          pixel_Tag[spatial_dim*n+y*width+x+i]=width;
                    }
                 //   done=true;
                }
             }
        }
     }
  }
}
template <typename scalar_t>
__global__ void update_pixels_Y_up(const int nthreads,
  scalar_t* labels, scalar_t* parent, scalar_t* pixel_Tag,const scalar_t* features,  scalar_t* block_features,
   scalar_t* T, const int nr_seeds_h1, const int nr_seeds_w1, const int height, const int width,
   int shift_x,int shift_y,  const int levels,  const int seeds_top_level,int ori_superP_size,
  const int in_channels) {
     int index = threadIdx.x + blockIdx.x * blockDim.x;
     if(index<nthreads) {
        int spatial_dim = height * width;
        int s_width = ceil(float(width)/3.0);
        int s_height = ceil(float(height)/3.0);
        int spatial_dim1 = s_width * s_height;
        int n = index/spatial_dim1;
        int s = index%(spatial_dim1);
        //int s_width = ceil(float(width)/3.0);
        int x = (s%s_width)*3+shift_x;
        int y = (s/s_width)*3+shift_y;
        int sublabel=(y)*width+(x);
        int top_level_index=(n*levels+seeds_top_level)*spatial_dim;
		if(x>0&&x<(width-1)&&y>0&&y<(height-1)){
		    int labelA = labels[top_level_index+sublabel-width];
		    int labelB = labels[top_level_index+sublabel];
            if (labelA != labelB){
		//    int a11 = labels[top_level_index+sublabel-width-1];
        //    int a12 = labels[top_level_index+sublabel-width];
        //    int a13 = labels[top_level_index+sublabel-width+1];
            int a21 = labels[top_level_index+sublabel-1];
            int a22 = labels[top_level_index+sublabel];
            int a23 = labels[top_level_index+sublabel+1];
            int a31 = labels[top_level_index+sublabel+width-1];
            int a32 = labels[top_level_index+sublabel+width];
            int a33 = labels[top_level_index+sublabel+width+1];
          //  bool done=false;
            bool isSplit = false;
            if ((a22 != a31) && (a22 == a21) && (a22 == a32)) isSplit =  true;
			if ((a22 != a33) && (a22 == a23) && (a22 == a32)) isSplit =  true;

            if (a32==labelB&&!isSplit)
            {
                scalar_t disA=get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1,  seeds_top_level,
                                            height, width, in_channels,labelA, ori_superP_size,x, y, n);
                scalar_t disB=get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1, seeds_top_level,
                                            height, width, in_channels,labelB,ori_superP_size, x, y, n);
                int curxleft=x-1;
                int leftsublabel=y*width+curxleft;
                int leftlabelA=labels[top_level_index+leftsublabel-width];
                int leftlabelB=labels[top_level_index+leftsublabel];
           //     int curlabelA=labelA;
                int curlabelB=labelB;
                int leftlabelBdown=labels[top_level_index+leftsublabel+width];
                bool bedone=true;
                int leftNum=0;
                while(leftlabelB!=curlabelB&&leftlabelA!=leftlabelB&&curxleft>=0)
                {
                    if(leftlabelB!=leftlabelBdown)
                    {
                        bedone = false;
                        break;
                    }
                    if(curxleft>0)
                    {
                        int leftleftlabelB=labels[top_level_index+leftsublabel-1];
                        int leftleftlabelBdown=labels[top_level_index+leftsublabel-1+width];
                        if(leftleftlabelB==leftlabelB&&leftleftlabelBdown!=leftlabelB)
                        {
                            bedone = false;
                            break;
                        }
                    }
                    disA=disA+get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1,  seeds_top_level,
                                               height, width, in_channels,leftlabelA,ori_superP_size, curxleft, y,n);
                    disB=disB+get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1,  seeds_top_level,
                                               height, width, in_channels,leftlabelB,ori_superP_size, curxleft, y,n);
                //    curlabelA=leftlabelA;
                    curlabelB=leftlabelB;
                    leftNum++;
                    curxleft--;
                    if(curxleft<0) break;
                    leftsublabel--;
                    leftlabelA=labels[top_level_index+leftsublabel-width];
                    leftlabelB=labels[top_level_index+leftsublabel];
                    leftlabelBdown=labels[top_level_index+leftsublabel+width];
                }
                int rightNum=0;
                if(bedone==true)
                {
                    int curxright=x+1;
                    int rightsublabel=y*width+curxright;
                    int rightlabelA=labels[top_level_index+rightsublabel-width];
                    int rightlabelB=labels[top_level_index+rightsublabel];
                    int rightlabelBdown=labels[top_level_index+rightsublabel+width];
               //     curlabelA=labelA;
                    curlabelB=labelB;
                    while(rightlabelB!=curlabelB&&rightlabelA!=rightlabelB&&curxright<width)
                    {
                        if(rightlabelB!=rightlabelBdown)
                        {
                            bedone = false;
                            break;
                        }
                        if(curxright<width-1)
                        {
                            int RrightlabelB=labels[top_level_index+rightsublabel+1];
                            int RrightlabelBdown=labels[top_level_index+rightsublabel+1+width];
                            if(RrightlabelB==rightlabelB&&RrightlabelBdown!=rightlabelB)
                            {
                                bedone = false;
                                break;
                            }
                        }
                        disA=disA+get_pixels_dis_CE(block_features, T,features,levels, nr_seeds_h1, nr_seeds_w1,  seeds_top_level,
                                               height, width, in_channels,rightlabelA,ori_superP_size, curxright, y,n);
                        disB=disB+get_pixels_dis_CE(block_features, T,features,levels,  nr_seeds_h1, nr_seeds_w1,  seeds_top_level,
                                               height, width, in_channels,rightlabelB,ori_superP_size,curxright, y,n);
                        rightNum++;
                   //     curlabelA=rightlabelA;
                        curlabelB=rightlabelB;
                        curxright++;
                        if(curxright>=width) break;
                        rightsublabel++;
                        rightlabelA=labels[top_level_index+rightsublabel-width];
                        rightlabelB=labels[top_level_index+rightsublabel];
                        rightlabelBdown=labels[top_level_index+rightsublabel+width];
                    }
                }
                if (disB>disA&&bedone)
                {

                   // update_pixels(block_features,T, features, labels,
                   //                  nr_seeds_h1, nr_seeds_w1,levels,top_level,height,  width,
                   //                  in_channels,labelA, x,y+1, n);
                    pixel_Tag[spatial_dim*n+sublabel]=-width;
                    for(int i=1;i<=leftNum;i++)
                    {
                        //leftlabelA=labels[top_level_index+y*width+x-i];
                        pixel_Tag[spatial_dim*n+y*width+x-i]=-width;
                       // update_pixels(block_features,T, features, labels,
                       //              nr_seeds_h1, nr_seeds_w1,levels,top_level,height,  width,
                       //              in_channels,leftlabelA, x-i,y+1, n);
                    }
                    for(int i=1;i<=rightNum;i++)
                    {

                        //rightlabelA=labels[top_level_index+y*width+x+i];
                       // update_pixels(block_features,T, features, labels,
                       //              nr_seeds_h1, nr_seeds_w1,levels,top_level,height,  width,
                        //             in_channels,rightlabelA, x+i,y+1, n);
                         pixel_Tag[spatial_dim*n+y*width+x+i]=-width;
                    }
            //		y++;
                }
            }
		    }
		}
     }
  }
template <typename scalar_t>
__global__ void clear_partitions(const int nthreads,     //followed by go_down_one_level
  scalar_t* nr_partitions, const int nr_seeds_h1, const int nr_seeds_w1,  int nr_seeds_h, int nr_seeds_w,
   const int levels, int seeds_top_level) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index<nthreads) {
       int spatial_dim = nr_seeds_h * nr_seeds_w;
       int  n = index/spatial_dim;
       int  s = index%spatial_dim;
       int  partitions_index = (n*levels+seeds_top_level)*nr_seeds_h1*nr_seeds_w1+s;
       nr_partitions[partitions_index]=0;
    }
  }

template <typename scalar_t>
__global__ void go_down_one_level(const int nthreads, scalar_t* parent,
  scalar_t* nr_partitions, const int nr_seeds_h1, const int nr_seeds_w1, const int nr_seeds_h, int nr_seeds_w,
   const int levels, int cur_level, int seeds_top_level) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index<nthreads) {
      // int new_level = cur_level;
       int spatial_dim = nr_seeds_h * nr_seeds_w;
       int  n = index/spatial_dim;
       int  s = index%spatial_dim;
       int old_level_index = (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1;
       int new_level_index = (n*levels+cur_level-1)*nr_seeds_h1*nr_seeds_w1;
       int parent_old_id = parent[new_level_index+s];
       int p = parent[old_level_index+parent_old_id];
       parent[new_level_index+s] = p;
       int top_level_index = (n*levels+seeds_top_level)*nr_seeds_h1*nr_seeds_w1;
       atomicAdd(nr_partitions+(top_level_index+p),nr_partitions[new_level_index+s]);
       //caffe_gpu_atomic_add(nr_partitions[new_level_index+s],nr_partitions+(top_level_index+p));
    }
  }
template <typename scalar_t>
__global__ void update_block_Y_down(const int nthreads,
  scalar_t* labels, scalar_t* parent, scalar_t* block_Tag,const scalar_t* features,  scalar_t* block_features,
  scalar_t* nr_partitions,  const int nr_seeds_h1, const int nr_seeds_w1,  int nr_seeds_h,  scalar_t* T,const int height, const int width,
  int step, int shift_y, int shift_x, int cur_level, const int levels, const int seeds_top_level,
  const int in_channels) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index<nthreads) {
        int s_width = ceil(float(step)/3.0);
        int s_height = ceil(float(nr_seeds_h)/3.0);
        const int n = index/(s_width*s_height);
        const int s =index%(s_width*s_height);
        const int y = (s/s_width) * 3+shift_y;
        const int x = (s%s_width) * 3+shift_x;
        //const int n = index/(nr_seeds_h*step/9);
        //const int s =index%(nr_seeds_h*step/9);
        //const int y = (s/(step/3)) * 3+shift_y;
        //const int x = (s%(step/3)) * 3+shift_x;
        int nr_seeds_w=step;
        int level_index = (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1;
        int top_level_index = (n*levels+seeds_top_level)*nr_seeds_h1*nr_seeds_w1;
        if(x>0&&x<(nr_seeds_w-1)&&y>0&&y<(nr_seeds_h-1))
        {
            int sublabel = y*step+x;
            int labelA = parent[level_index+sublabel];
            int labelB = parent[level_index+sublabel+step];
            if (labelA != labelB)
			{
			    int a11 = parent[level_index+(y-1)*step+(x-1)];
				int a12 = parent[level_index+(y-1)*step+(x)];
				int a13 = parent[level_index+(y-1)*step+(x+1)];
				int a21 = parent[level_index+(y)*step+(x-1)];
				int a22 = parent[level_index+(y)*step+(x)];
				int a23 = parent[level_index+(y)*step+(x+1)];
			//	int a31 = parent[level_index+(y+1)*step+(x-1)];
			//	int a32 = parent[level_index+(y+1)*step+(x)];
			//	int a33 = parent[level_index+(y+1)*step+(x+1)];

				//done = false;
				if(a12==labelA)
				{
				    if (nr_partitions[top_level_index+labelA] > 1)
				    {
				        if (nr_partitions[top_level_index+labelA] <= 2)
				        {
				             bool bedone=true;
                            // delete_block_CE(block_features,T,nr_partitions,parent,
                            //                nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                            //                in_channels,labelA,sublabel,int n);
                             scalar_t intA = get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                            levels,seeds_top_level,cur_level,in_channels,labelA,sublabel, n);
                             scalar_t intB = get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                            levels,seeds_top_level,cur_level,in_channels,labelB,sublabel, n);
                           //  int cursublabel=sublabel;
                             int curxleft=x-1;
                             int leftsublabel=y*step+curxleft;
                             int leftlabelA=parent[level_index+leftsublabel];
                             int leftlabelB=parent[level_index+leftsublabel+step];
                             int curlabelA=labelA;
                          //   int curlabelB=labelB;
                             int leftlabelAup=parent[level_index+leftsublabel-step];

                             int leftNum=0;
                             while(leftlabelA!=curlabelA&&leftlabelA!=leftlabelB)
                             {
                                if(leftlabelA!=leftlabelAup)
                                {
                                    bedone = false;
                                    break;
                                }
                                if(curxleft>0)
                                {
                                    int leftleftlabelA=parent[level_index+leftsublabel-1];
                                    int leftleftlabelAup=parent[level_index+leftsublabel-1-step];
                                    if(leftleftlabelA==leftlabelA&&leftleftlabelAup!=leftlabelAup)
                                    {
                                        bedone = false;
                                        break;
                                    }
                                   // delete_block_CE(block_features,T,nr_partitions,parent,
                                   //                 nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                                   //                 in_channels,leftlabelA,leftsublabel,int n);
                                    intA = intA + get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                                levels,seeds_top_level,cur_level,in_channels,leftlabelA,leftsublabel, n);
                                    intB = intB + get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                                levels,seeds_top_level,cur_level,in_channels,leftlabelB,leftsublabel, n);
                                    curlabelA=leftlabelA;
                               //     curlabelB=leftlabelB;
                                    leftNum++;
                                    curxleft--;
                                    if(curxleft<0) break;
                                    leftsublabel=y*step+curxleft;
                                    leftlabelA=parent[level_index+leftsublabel];
                                    leftlabelB=parent[level_index+leftsublabel+step];
                                    leftlabelAup=parent[level_index+leftsublabel-step];
                                }
                             }
                             int rightNum=0;
						     if(bedone==true)
						     {
						        curlabelA=labelA;
							 //   curlabelB=labelB;
							    int curxright=x+1;
                                int rightsublabel=y*step+curxright;
                                int rightlabelA=parent[level_index+rightsublabel];
                                int rightlabelB=parent[level_index+rightsublabel+step];
                                int rightlabelup=parent[level_index+rightsublabel-step];
                                while(rightlabelA!=curlabelA&&rightlabelA!=rightlabelB)
                                {
                                    if(rightlabelA!=rightlabelup)
                                    {
                                        bedone = false;
                                        break;
                                    }
                                    if(curxright<nr_seeds_w-1)
                                    {
                                        int RrightlabelA=parent[level_index+rightsublabel+1];
                                        int RrightlabelAup=parent[level_index+rightsublabel+1-step];
                                        if(RrightlabelA==rightlabelA&&RrightlabelAup!=rightlabelA)
                                        {
                                            bedone = false;
                                            break;
                                        }
                                       // delete_block_CE(block_features,T,nr_partitions,parent,
                                       //                 nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                                       //                 in_channels,rightlabelA,rightsublabel,int n);
                                        intA = intA + get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                                levels,seeds_top_level,cur_level,in_channels,rightlabelA,rightsublabel, n);
                                        intB = intB + get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                                    levels,seeds_top_level,cur_level,in_channels,rightlabelB,rightsublabel, n);
                                        rightNum++;
                                        curlabelA=rightlabelA;
                                    //    curlabelB=rightlabelB;
                                        curxright++;
                                        if(curxright>=nr_seeds_w) break;
                                        rightsublabel=y*step+curxright;
                                        rightlabelA=parent[level_index+rightsublabel];
                                        rightlabelB=parent[level_index+rightsublabel+step];
                                        rightlabelup=parent[level_index+rightsublabel-step];
                                    }
                                }
						     }
						    // float confidence = fabs(intA - intB);
						     int space_dim=nr_seeds_h1*nr_seeds_w1;
						     if ((intB < intA) &&bedone==true)
						     {
						       // add_block_CE(block_features,T,nr_partitions,parent,
                               //            nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                               //            in_channels,labelB,sublabel,int n);
                                block_Tag[space_dim*n+sublabel]=step;
                                for(int i=1;i<=leftNum;i++)
                                {
                                    leftsublabel=y*step+x-i;
                                    block_Tag[space_dim*n+leftsublabel]=step;

                                }
                                for(int i=1;i<=rightNum;i++)
                                {
                                    int rightsublabel=y*step+x+i;
                                    block_Tag[space_dim*n+rightsublabel]=step;

                                }
                             //   done = true;
						     }
				        }
				        else if (nr_partitions[top_level_index+labelA] > 2)
				        {
				            bool isSplit = false;
				            if ((a22 != a11) && (a22 == a21) && (a22 == a12)) isSplit = true;
			                if ((a22 != a13) && (a22 == a23) && (a22 == a12)) isSplit = true;
				            if (!isSplit)
				            {
				             //   int cursublabel=sublabel;
                                int curxleft=x-1;
                                int leftsublabel=y*step+curxleft;
                                int leftlabelA=parent[level_index+leftsublabel];
                                int leftlabelB=parent[level_index+leftsublabel+step];
                                int curlabelA=labelA;
                            //    int curlabelB=labelB;
                                int leftlabelAup=parent[level_index+leftsublabel-step];
                                bool bedone=true;
                                int leftNum=0;
                               // delete_block_CE(block_features,T,nr_partitions,parent,
                               //             nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                               //             in_channels,labelA,sublabel,int n);
                                scalar_t intA = get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                              levels,seeds_top_level,cur_level,in_channels,labelA,sublabel, n);
                                scalar_t intB = get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                              levels,seeds_top_level,cur_level,in_channels,labelB,sublabel, n);
                                while(leftlabelA!=curlabelA&&leftlabelA!=leftlabelB)
                                {
                                    if(leftlabelA!=leftlabelAup)
                                    {
                                        bedone = false;
                                        break;
                                    }
                                    if(curxleft>0)
                                    {
                                        int leftleftlabelA=parent[level_index+leftsublabel-1];
                                        int leftleftlabelAup=parent[level_index+leftsublabel-1-step];
                                        if(leftleftlabelA==leftlabelA&&leftleftlabelAup!=leftlabelAup)
                                        {
                                            bedone = false;
                                            break;
                                        }
                                    }
                                    //delete_block_CE(block_features,T,nr_partitions,parent,
                                    //                nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                                    //                in_channels,leftlabelA,leftsublabel,int n);
                                    intA = intA + get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                                levels,seeds_top_level,cur_level,in_channels,leftlabelA,leftsublabel, n);
                                    intB = intB + get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                                levels,seeds_top_level,cur_level,in_channels,leftlabelB,leftsublabel, n);
                                    curlabelA=leftlabelA;
                               //     curlabelB=leftlabelB;
                                    leftNum++;
                                    curxleft--;
                                    if(curxleft<0) break;
                                    leftsublabel=y*step+curxleft;
                                    leftlabelA=parent[level_index+leftsublabel];
                                    leftlabelB=parent[level_index+leftsublabel+step];
                                    leftlabelAup=parent[level_index+leftsublabel-step];
                                }
                                int rightNum=0;
							    if(bedone==true)
							    {
							        curlabelA=labelA;
								  //  curlabelB=labelB;
								    int curxright=x+1;
                                    int rightsublabel=y*step+curxright;
                                    int rightlabelA=parent[level_index+rightsublabel];
                                    int rightlabelB=parent[level_index+rightsublabel+step];
                                    int rightlabelup=parent[level_index+rightsublabel-step];
                                    while(rightlabelA!=curlabelA&&rightlabelA!=rightlabelB)
                                    {
                                        if(rightlabelA!=rightlabelup)
                                        {
                                            bedone = false;
                                            break;
                                        }
                                        if(curxright<nr_seeds_w-1)
                                        {
                                            int RrightlabelA=parent[level_index+rightsublabel+1];
                                            int RrightlabelAup=parent[level_index+rightsublabel+1-step];
                                            if(RrightlabelA==rightlabelA&&RrightlabelAup!=rightlabelA)
                                            {
                                                bedone = false;
                                                break;
                                            }
                                        }
                                        //delete_block_CE(block_features,T,nr_partitions,parent,
                                        //                nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                                        //               in_channels,rightlabelA,rightsublabel,int n);
                                        intA = intA + get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                                levels,seeds_top_level,cur_level,in_channels,rightlabelA,rightsublabel, n);
                                        intB = intB + get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                                    levels,seeds_top_level,cur_level,in_channels,rightlabelB,rightsublabel, n);
                                        rightNum++;
                                        curlabelA=rightlabelA;
                                  //      curlabelB=rightlabelB;
                                        curxright++;
                                        if(curxright>=nr_seeds_w) break;
                                        rightsublabel=y*step+curxright;
                                        rightlabelA=parent[level_index+rightsublabel];
                                        rightlabelB=parent[level_index+rightsublabel+step];
                                        rightlabelup=parent[level_index+rightsublabel-step];
                                    }
							    }
							//    float confidence = fabs(intA - intB);
							    int space_dim=nr_seeds_h1*nr_seeds_w1;
							    if ((intB < intA) &&bedone==true)
							    {
							        //add_block_CE(block_features,T,nr_partitions,parent,
                                    //               nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                                    //               in_channels,labelB,sublabel,int n);
                                     block_Tag[space_dim*n+sublabel]=step;
                                    for(int i=1;i<=leftNum;i++)
                                    {
                                        leftsublabel=y*step+x-i;
                                        block_Tag[space_dim*n+leftsublabel]=step;

                                    }
                                    for(int i=1;i<=rightNum;i++)
                                    {
                                        int rightsublabel=y*step+x+i;
                                        block_Tag[space_dim*n+rightsublabel]=step;
                                    }
                                  //  done = true;
							    }
				            }
				        }
				    }
				}
			}
		}
	}
}
template <typename scalar_t>
__global__ void update_block_Y_up(const int nthreads,
  scalar_t* labels, scalar_t* parent, scalar_t* block_Tag,const scalar_t* features,
   scalar_t* block_features, scalar_t *nr_partitions, const int nr_seeds_h1, const int nr_seeds_w1,
   int nr_seeds_h,  scalar_t* T,const int height, const int width,
  int step, int shift_y,int shift_x,  int cur_level, const int levels, const int seeds_top_level,
  const int in_channels) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < nthreads) {
        int s_width = ceil(float(step)/3.0);
        int s_height = ceil(float(nr_seeds_h)/3.0);
        const int n = index/(s_width*s_height);
        const int s =index%(s_width*s_height);
        const int y = (s/s_width) * 3+shift_y;
        const int x = (s%s_width) * 3+shift_x;

        //const int n = index/(nr_seeds_h*step/9);
        //const int s =index%(nr_seeds_h*step/9);
        //const int y = (s/(step/3)) * 3+shift_y;
        //const int x = (s%(step/3)) * 3+shift_x;
        int nr_seeds_w=step;
        int level_index = (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1;
        int top_level_index = (n*levels+seeds_top_level)*nr_seeds_h1*nr_seeds_w1;
        if(x>0&&x<(nr_seeds_w-1)&&y>0&&y<(nr_seeds_h-1))
        {
            int sublabel = y*step+x;
            int labelA = parent[level_index+sublabel-step];
            int labelB = parent[level_index+sublabel];
            if (labelA != labelB)
			{
		//	    int a11 = parent[level_index+(y-1)*step+(x-1)];
		//		int a12 = parent[level_index+(y-1)*step+(x)];
		//		int a13 = parent[level_index+(y-1)*step+(x+1)];
				int a21 = parent[level_index+(y)*step+(x-1)];
				int a22 = parent[level_index+(y)*step+(x)];
				int a23 = parent[level_index+(y)*step+(x+1)];
				int a31 = parent[level_index+(y+1)*step+(x-1)];
				int a32 = parent[level_index+(y+1)*step+(x)];
				int a33 = parent[level_index+(y+1)*step+(x+1)];

				//done = false;
				if(a32==labelB)
				{
				    if ( (nr_partitions[top_level_index+labelB] > 1))
				    {
				        //sublabel = (y+1)*step+x;
				        if (nr_partitions[top_level_index+labelB] <= 2)
				        {
				            bool bedone=true;
                            //delete_block_CE(block_features,T,nr_partitions,parent,
                             //               nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                             //               in_channels,labelB,sublabel,int n);
                            scalar_t intA = get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                            levels,seeds_top_level,cur_level,in_channels,labelA,sublabel, n);
                            scalar_t intB = get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                            levels,seeds_top_level,cur_level,in_channels,labelB,sublabel, n);
                          //  int cursublabel=sublabel;
                            int curxleft=x-1;
                            int leftsublabel=(y)*step+curxleft;
                            int leftlabelA=parent[level_index+leftsublabel-step];
                            int leftlabelB=parent[level_index+leftsublabel];
                          //  int curlabelA=labelA;
                            int curlabelB=labelB;
                            int leftlabelBdown=parent[level_index+leftsublabel+step];

                            int leftNum=0;
                            while(leftlabelB!=curlabelB&&leftlabelA!=leftlabelB)
                            {
                                if(leftlabelB!=leftlabelBdown)
                                {
                                    bedone = false;
                                    break;
                                }
                                if(curxleft>0)
                                {
                                    int leftleftlabelB=parent[level_index+leftsublabel-1];
                                    int leftleftlabelBdown=parent[level_index+leftsublabel-1+step];
                                    if(leftleftlabelB==leftlabelB&&leftleftlabelBdown!=leftlabelB)
                                    {
                                        bedone = false;
                                        break;
                                    }
                                }
                               // delete_block_CE(block_features,T,nr_partitions,parent,
                               //                         nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                               //                         in_channels,leftlabelB,leftsublabel,int n);
                                intA = intA + get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                        levels,seeds_top_level,cur_level,in_channels,leftlabelA,leftsublabel, n);
                                intB = intB + get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                            levels,seeds_top_level,cur_level,in_channels,leftlabelB,leftsublabel, n);
                               // curlabelA=leftlabelA;
                                curlabelB=leftlabelB;
                                curxleft--;
                                leftNum++;
                                if(curxleft<0) break;
                                leftsublabel=(y)*step+curxleft;
                                leftlabelA=parent[level_index+leftsublabel-step];
                                leftlabelB=parent[level_index+leftsublabel];
                                leftlabelBdown=parent[level_index+leftsublabel+step];
                            }
                            int rightNum=0;
                            if(bedone==true)
                            {
                               // curlabelA=labelA;
							    curlabelB=labelB;
							    int curxright=x+1;
                                int rightsublabel=(y)*step+curxright;
                                int rightlabelA=parent[level_index+rightsublabel-step];
                                int rightlabelB=parent[level_index+rightsublabel];
                                int rightlabelBdown=parent[level_index+rightsublabel+step];
                                while(rightlabelB!=curlabelB&&rightlabelA!=rightlabelB)
                                {
                                    if(rightlabelB!=rightlabelBdown)
                                    {
                                        bedone = false;
                                        break;
                                    }
                                    if(curxright<nr_seeds_w-1)
                                    {
                                        int RrightlabelB=parent[level_index+rightsublabel+1];
                                        int RrightlabelBdown=parent[level_index+rightsublabel+1+step];
                                        if(RrightlabelB==rightlabelB&&RrightlabelBdown!=rightlabelB)
                                        {
                                            bedone = false;
                                            break;
                                        }
                                    }
                                   // delete_block_CE(block_features,T,nr_partitions,parent,
                                   //                     nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                                   //                     in_channels,rightlabelB,rightsublabel,int n);
                                    intA = intA + get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                            levels,seeds_top_level,cur_level,in_channels,rightlabelA,rightsublabel, n);
                                    intB = intB + get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                                levels,seeds_top_level,cur_level,in_channels,rightlabelB,rightsublabel, n);
                                    rightNum++;
                                  //  curlabelA=rightlabelA;
                                    curlabelB=rightlabelB;
                                    curxright++;
                                    if(curxright>=nr_seeds_w) break;
                                    rightsublabel=(y)*step+curxright;
                                    rightlabelA=parent[level_index+rightsublabel-step];
                                    rightlabelB=parent[level_index+rightsublabel];
                                    rightlabelBdown=parent[level_index+rightsublabel+step];
                                }
                            }
                         //   float confidence = fabs(intA - intB);
                           // leftNum=0;
                           // rightNum=0;
                            int space_dim = nr_seeds_h1*nr_seeds_w1;
                            if ((intA < intB) &&bedone==true)
                            {
                                block_Tag[space_dim*n+sublabel]=-step;
                                for(int i=1;i<=leftNum;i++)
                                {
                                    leftsublabel=(y)*step+x-i;
                                    block_Tag[space_dim*n+leftsublabel]=-step;

                                }
                                for(int i=1;i<=rightNum;i++)
                                {
                                    int rightsublabel=(y)*step+x+i;
                                    block_Tag[space_dim*n+rightsublabel]=-step;

                                }
                            }


				        }
				        else if (nr_partitions[top_level_index+labelB] > 2)
				        {
				            bool isSplit = false;
				            if ((a22 != a31) && (a22 == a21) && (a22 == a32)) isSplit = true;
			                if ((a22 != a33) && (a22 == a23) && (a22 == a32)) isSplit = true;
				            if (!isSplit)
				            {
				               // delete_block_CE(block_features,T,nr_partitions,parent,
                               //                 nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                               //                 in_channels,labelB,sublabel,int n);
                                scalar_t intA = get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                                levels,seeds_top_level,cur_level,in_channels,labelA,sublabel, n);
                                scalar_t intB = get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                                levels,seeds_top_level,cur_level,in_channels,labelB,sublabel, n);
                               // int cursublabel=sublabel;
                                int curxleft=x-1;
                                int leftsublabel=y*step+curxleft;
                                int leftlabelA=parent[level_index+leftsublabel-step];
                                int leftlabelB=parent[level_index+leftsublabel];
                               // int curlabelA=labelA;
                                int curlabelB=labelB;
                                int leftlabelBdown=parent[level_index+leftsublabel+step];

                                bool bedone=true;
                                int leftNum=0;
                                while(leftlabelB!=curlabelB&&leftlabelA!=leftlabelB)
                                {
                                    if(leftlabelB!=leftlabelBdown)
                                    {
                                        bedone = false;
                                        break;
                                    }
                                    if(curxleft>0)
                                    {
                                        int leftleftlabelB=parent[level_index+leftsublabel-1];
                                        int leftleftlabelBdown=parent[level_index+leftsublabel-1+step];
                                        if(leftleftlabelB==leftlabelB&&leftleftlabelBdown!=leftlabelB)
                                        {
                                            bedone = false;
                                            break;
                                        }
                                    }
                                   // delete_block_CE(block_features,T,nr_partitions,parent,
                                    //                    nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                                    //                    in_channels,leftlabelB,leftsublabel,int n);
                                    intA = intA + get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                            levels,seeds_top_level,cur_level,in_channels,leftlabelA,leftsublabel,n);
                                    intB = intB + get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                                levels,seeds_top_level,cur_level,in_channels,leftlabelB,leftsublabel,n);
                                  //  curlabelA=leftlabelA;
                                    curlabelB=leftlabelB;
                                    curxleft--;
                                    leftNum++;
                                    if(curxleft<0) break;
                                    leftsublabel=y*step+curxleft;
                                    leftlabelA=parent[level_index+leftsublabel-step];
                                    leftlabelB=parent[level_index+leftsublabel];
                                    leftlabelBdown=parent[level_index+leftsublabel+step];
                                }
                                int rightNum=0;
							    if(bedone==true)
							    {
							    //    curlabelA=labelA;
								    curlabelB=labelB;
								    int curxright=x+1;
                                    int rightsublabel=y*step+curxright;
                                    int rightlabelA=parent[level_index+rightsublabel-step];
                                    int rightlabelB=parent[level_index+rightsublabel];
                                    int rightlabelBdown=parent[level_index+rightsublabel+step];
                                    while(rightlabelB!=curlabelB&&rightlabelA!=rightlabelB)
                                    {
                                        if(rightlabelB!=rightlabelBdown)
                                        {
                                            bedone = false;
                                            break;
                                        }
                                        if(curxright<nr_seeds_w-1)
                                        {
                                            int RrightlabelB=parent[level_index+rightsublabel+1];
                                            int RrightlabelBdown=parent[level_index+rightsublabel+1+step];
                                            if(RrightlabelB==rightlabelB&&RrightlabelBdown!=rightlabelB)
                                            {
                                                bedone = false;
                                                break;
                                            }
                                        }
                                       // delete_block_CE(block_features,T,nr_partitions,parent,
                                        //                nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                                        //                in_channels,rightlabelB,leftsublabel,int n);
                                        intA = intA + get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                                levels,seeds_top_level,cur_level,in_channels,rightlabelA,leftsublabel,n);
                                        intB = intB + get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                                    levels,seeds_top_level,cur_level,in_channels,rightlabelB,leftsublabel,n);
                                        rightNum++;
                                   //     curlabelA=rightlabelA;
                                        curlabelB=rightlabelB;
                                        curxright++;
                                        if(curxright>=nr_seeds_w) break;
                                        rightsublabel=y*step+curxright;
                                        rightlabelA=parent[level_index+rightsublabel-step];
                                        rightlabelB=parent[level_index+rightsublabel];
                                        rightlabelBdown=parent[level_index+rightsublabel+step];
                                    }
							    }
							 //   float confidence = fabs(intA - intB);
							    int space_dim = nr_seeds_h1*nr_seeds_w1;
							  //  leftNum=0;
							  //  rightNum=0;
                                if ((intA < intB) &&bedone==true)
                                {
                                   // add_block_CE(block_features,T,nr_partitions,parent,
                                   //                nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                                   //                in_channels,labelA,sublabel,int n);
                                    block_Tag[space_dim*n+sublabel]=-step;
                                    for(int i=1;i<=leftNum;i++)
                                    {
                                        leftsublabel=y*step+x-i;
                                        block_Tag[space_dim*n+leftsublabel]=-step;

                                    }
                                    for(int i=1;i<=rightNum;i++)
                                    {
                                        int rightsublabel=y*step+x+i;
                                        block_Tag[space_dim*n+rightsublabel]=-step;
                                    }

                                }

				            }
				        }
				    }
				}
			}
        }
      //  update_labels(level);
    }
   }
template <typename scalar_t>
__global__ void update_block_X_right(const int nthreads,
  scalar_t* labels, scalar_t* parent, scalar_t* block_Tag,const scalar_t* features,  scalar_t* block_features,
  scalar_t* nr_partitions, const int nr_seeds_h1, const int nr_seeds_w1,  int nr_seeds_h,  scalar_t* T,
  const int height, const int width, int step, int shift_y,int shift_x,  int cur_level, const int levels,  int seeds_top_level,
  const int in_channels) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < nthreads) {
        int s_width = ceil(float(step)/3.0);
        int s_height = ceil(float(nr_seeds_h)/3.0);
        const int n = index/(s_width*s_height);
        const int s =index%(s_width*s_height);
        const int y = (s/s_width) * 3+shift_y;
        const int x = (s%s_width) * 3+shift_x;
        int nr_seeds_w = step;
        if(x>0&&x<(nr_seeds_w-1)&&y>0&&y<(nr_seeds_h1-1)){
           int sublabel = y*step+x;
           int level_index = (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1;
           int top_level_index = (n*levels+seeds_top_level)*nr_seeds_h1*nr_seeds_w1;
        //   int parent_index= (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1+sublabel;
           int labelA = parent[level_index+sublabel];
           int labelB = parent[level_index+sublabel+1];

           if(labelA!=labelB)
           {

             int a11 = parent[level_index+(y-1)*step+(x-1)];
             int a12 = parent[level_index+(y-1)*step+(x)];
           //  int a13 = parent[level_index+(y-1)*step+(x+1)];

             int a21 = parent[level_index+(y)*step+(x-1)];
             int a22 = parent[level_index+(y)*step+(x)];
          //   int a23 = parent[level_index+(y)*step+(x+1)];

             int a31 = parent[level_index+(y+1)*step+(x-1)];
             int a32 = parent[level_index+(y+1)*step+(x)];
       //      int a33 = parent[level_index+(y+1)*step+(x+1)];
       //      bool done = false;
             if(a21==labelA)
             {

                if(nr_partitions[top_level_index+labelA]>1&&nr_partitions[top_level_index+labelA]<=2)
                {
                 // int cursublabel = sublabel;
                  int curyup=y-1;
                  int upsublabel=curyup*step+x;
                  int uplabelA = parent[level_index+upsublabel];
                  int uplabelB = parent[level_index+upsublabel+1];
                  int curlabelA=labelA;
             //     int curlabelB=labelB;
                  int uplabelAl=parent[level_index+upsublabel-1];
                  //down

                  bool bedone=true;
                //  delete_block_CE(block_features,T,nr_partitions,parent,
                               // nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                               // in_channels,labelA,sublabel,int n);
                  scalar_t intA = get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                levels,seeds_top_level,cur_level,in_channels,labelA,sublabel, n);
                  scalar_t intB = get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                levels,seeds_top_level,cur_level,in_channels,labelB,sublabel, n);
                  int upNum=0;


                 while(uplabelA!=curlabelA&&uplabelA!=uplabelB)
                  {
                    if(uplabelA!=uplabelAl)
                    {
                        bedone = false;
                        break;
                    }
                    if(curyup>0)
                    {
                        int upuplabelA=parent[level_index+upsublabel-step];
                        int upuplabelAl=parent[level_index+upsublabel-step-1];
                        if(upuplabelA==uplabelA&&upuplabelAl!=uplabelA)//对角分裂检查
                        {
                            bedone = false;
                            break;
                        }
                    }
                  //  delete_block_CE(block_features,T,nr_partitions,parent,
                             //      nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                               //    in_channels,uplabelA,upsublabel,int n);
                    intA = intA + get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                levels,seeds_top_level,cur_level,in_channels,uplabelA,upsublabel, n);
                    intB = intB + get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                levels,seeds_top_level,cur_level,in_channels,uplabelB,upsublabel, n);
                    curlabelA=uplabelA;
               //     curlabelB=uplabelB;
                    curyup=curyup-1;
                    upNum++;
                    if(curyup<0) break;
                    upsublabel=curyup*step+x;
                    uplabelA=parent[level_index+upsublabel];
                    uplabelB=parent[level_index+upsublabel+1];
                    uplabelAl=parent[level_index+upsublabel-1];
                  }
                  int downNum=0;
                  if(bedone==true)
				  {
					int curydown=y+1;
                    int downsublabel=curydown*step+x;
                    int downlabelA=parent[level_index+downsublabel];
                    int downlabelB=parent[level_index+downsublabel+1];
                    int downlabelAl=parent[level_index+downsublabel-1];
                    curlabelA=labelA;
					// curlabelB=labelB;
					while(downlabelA!=curlabelA&&downlabelA!=downlabelB)
				    {
				        if(downlabelA!=downlabelAl)
                        {
                            bedone = false;
                            break;
                        }
                        if(curydown<nr_seeds_h-1)
                        {
                            int downdownlabelA=parent[level_index+downsublabel+step];
                            int downdownlabelAl=parent[level_index+downsublabel+step-1];
                            if(downdownlabelA==downlabelA&&downdownlabelAl!=downlabelA)
                            {
                                bedone = false;
                                break;
                            }
                        }
                     //   delete_block_CE(block_features,T,nr_partitions,parent,
                           //        nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                              //     in_channels,downlabelA,downsublabel,int n);
                        intA = intA + get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                levels,seeds_top_level,cur_level,in_channels,downlabelA,downsublabel, n);
                        intB = intB + get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                levels,seeds_top_level,cur_level,in_channels,downlabelB,downsublabel, n);
                        downNum++;
                        curlabelA=downlabelA;
                       // curlabelB=downlabelB;
                        curydown=curydown+1;
                        if(curydown>=nr_seeds_h) break;
                        downsublabel=curydown*step+x;
                        downlabelA=parent[level_index+downsublabel];
                        downlabelB=parent[level_index+downsublabel+1];
                        downlabelAl=parent[level_index+downsublabel-1];
				    }
				  }
				 // float confidence = fabs(intA - intB);
				  int space_dim = nr_seeds_h1*nr_seeds_w1;
				  if ((intB < intA) &&bedone==true)
                    {
                        block_Tag[space_dim*n+sublabel]=1;
                        for(int i=1;i<=upNum;i++)
                        {
                             curyup=y-i;
                            upsublabel=curyup*step+x;
                            block_Tag[space_dim*n+upsublabel]=1;
                        }
                        for(int i=1;i<=downNum;i++)
                        {
                            int  curydown=y+i;
                            int downsublabel=curydown*step+x;
                            block_Tag[space_dim*n+downsublabel]=1;
                        }

                    }

                }
                else if (nr_partitions[top_level_index+labelA] > 2) // 3 or more partitions
			    {
					bool isSplit = false;
					if ((a22 != a11) && (a22 == a12) && (a22 == a21)) isSplit = true;
			        if ((a22 != a31) && (a22 == a32) && (a22 == a21)) isSplit = true;
					if (!isSplit)
                    {
                        //int cursublabel=sublabel;
                        int curyup=y-1;
                        int upsublabel=curyup*step+x;
                        int uplabelA=parent[level_index+upsublabel];
                        int uplabelB=parent[level_index+upsublabel+1];
                        int curlabelA=labelA;
                     //   int curlabelB=labelB;
                        int uplabelAl=parent[level_index+upsublabel-1];

                        bool bedone=true;
                      //  delete_block_CE(block_features,T,nr_partitions,parent,
                      //                  nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                      //                  in_channels,labelA,sublabel,int n);
                         scalar_t intA = get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                        levels,seeds_top_level,cur_level,in_channels,labelA,sublabel, n);
                         scalar_t intB = get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                       levels,seeds_top_level,cur_level,in_channels,labelB,sublabel, n);
                        int upNum=0;

                        while(uplabelA!=curlabelA&&uplabelA!=uplabelB)
                        {
                            if(uplabelA!=uplabelAl)
                            {
                                bedone = false;
                                break;
                            }
                            if(curyup>0)
                            {
                                int upuplabelA=parent[level_index+upsublabel-step];
                                int upuplabelAl=parent[level_index+upsublabel-step-1];
                                if(upuplabelA==uplabelA&&upuplabelAl!=uplabelA)//对角分裂检查
                                {
                                    bedone = false;
                                    break;
                                }
                            }
                          //  delete_block_CE(block_features,T,nr_partitions,parent,
                           //                 nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                            //                in_channels,uplabelA,upsublabel,int n);
                            intA = intA + get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                           levels,seeds_top_level,cur_level,in_channels,uplabelA,upsublabel, n);
                            intB = intB + get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                           levels,seeds_top_level,cur_level,in_channels,uplabelB,upsublabel, n);
                            curlabelA=uplabelA;
                          //  curlabelB=uplabelB;
                            curyup=curyup-1;
                            upNum++;
                            if(curyup<0) break;
                            upsublabel=curyup*step+x;
                            uplabelA=parent[level_index+upsublabel];
                            uplabelB=parent[level_index+upsublabel+1];
                            uplabelAl=parent[level_index+upsublabel-1];
                        }
                        int downNum=0;
						if(bedone==true)
						{
							int curydown=y+1;
                            int downsublabel=curydown*step+x;
                            int downlabelA=parent[level_index+downsublabel];
                            int downlabelB=parent[level_index+downsublabel+1];
                            int downlabelAl=parent[level_index+downsublabel-1];
							curlabelA=labelA;
						//	curlabelB=labelB;
							while(downlabelA!=curlabelA&&downlabelA!=downlabelB)
							{
								if(downlabelA!=downlabelAl)
								{
									bedone = false;
									break;
								}
								if(curydown<nr_seeds_h-1)
								{
									int downdownlabelA=parent[level_index+downsublabel+step];
									int downdownlabelAl=parent[level_index+downsublabel+step-1];
									if(downdownlabelA==downlabelA&&downdownlabelAl!=downlabelA)
									{
										bedone = false;
										break;
									}
								}
								//delete_block_CE(block_features,T,nr_partitions,parent,
                                  //              nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                                  //              in_channels,downlabelA,downsublabel,int n);
								intA = intA + get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                               levels,seeds_top_level,cur_level,in_channels,downlabelA,downsublabel, n);
                                intB = intB + get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                               levels,seeds_top_level,cur_level,in_channels,downlabelB,downsublabel, n);
								downNum++;
								curlabelA=downlabelA;
								//curlabelB=downlabelB;
								curydown=curydown+1;
								if(curydown>=nr_seeds_h) break;
								downsublabel=curydown*step+x;
								downlabelA=parent[level_index+downsublabel];
								downlabelB=parent[level_index+downsublabel+1];
								downlabelAl=parent[level_index+downsublabel-1];
							}
						}

						//float confidence = fabs(intA - intB);
						int space_dim = nr_seeds_h1*nr_seeds_w1;
						// downNum=1;
						// upNum=0;
						if ((intB < intA) && bedone==true)
						{

                            block_Tag[space_dim*n+sublabel]=1;
							for(int i=1;i<=upNum;i++)
							{
								upsublabel=(y-i)*step+x;
								block_Tag[space_dim*n+upsublabel]=1;
							}
							for(int i=1;i<=downNum;i++)
							{

								int downsublabel=(y+i)*step+x;
								block_Tag[space_dim*n+downsublabel]=1;
							}
						}

                    }
				}
             }
           }
        }
    }
  }
  template <typename scalar_t>
__global__ void update_block_X_left(const int nthreads,
  scalar_t* labels, scalar_t* parent, scalar_t* block_Tag,const scalar_t* features,  scalar_t* block_features,
  scalar_t *nr_partitions, const int nr_seeds_h1, const int nr_seeds_w1,  int nr_seeds_h,  scalar_t* T,
  const int height, const int width,
  int step, int shift_y,int shift_x,  int cur_level, const int levels,  const int seeds_top_level,
  const int in_channels) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < nthreads) {
        int s_width = ceil(float(step)/3.0);
        int s_height = ceil(float(nr_seeds_h)/3.0);
        const int n = index/(s_width*s_height);
        const int s =index%(s_width*s_height);
        const int y = (s/s_width) * 3+shift_y;
        const int x = (s%s_width) * 3+shift_x;

      //  const int n = index/(nr_seeds_h*step/9);
      //  const int s =index%(nr_seeds_h*step/9);
     //   const int y = (s/(step/3)) * 3+shift_y;
     //   const int x = (s%(step/3)) * 3+shift_x;
        int nr_seeds_w = step;
         if(x>0&&x<(nr_seeds_w-1)&&y>0&&y<(nr_seeds_h1-1)){
           int sublabel = y*step+x;
          // int parent_index= (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1+sublabel;

           int level_index = (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1;
           int top_level_index = (n*levels+seeds_top_level)*nr_seeds_h1*nr_seeds_w1;
           int labelA = parent[level_index+sublabel-1];
           int labelB = parent[level_index+sublabel];
           if(labelA!=labelB)
           {
          //   int a11 = parent[level_index+(y-1)*step+(x-1)];
             int a12 = parent[level_index+(y-1)*step+(x)];
             int a13 = parent[level_index+(y-1)*step+(x+1)];
          //   int a14 = parent[level_index+(y-1)*step+(x+2)];
         //    int a21 = parent[level_index+(y)*step+(x-1)];
             int a22 = parent[level_index+(y)*step+(x)];
             int a23 = parent[level_index+(y)*step+(x+1)];
          //   int a24 = parent[level_index+(y)*step+(x+2)];
          //   int a31 = parent[level_index+(y+1)*step+(x-1)];
             int a32 = parent[level_index+(y+1)*step+(x)];
             int a33 = parent[level_index+(y+1)*step+(x+1)];
             //bool done = false;
             if(a23==labelB)
             {

               if(nr_partitions[top_level_index+labelB]>1&&nr_partitions[top_level_index+labelB]<=2)
               {
             //   int cursublabel=sublabel;
                int curyup=y-1;
                int upsublabel=curyup*step+x;
                int uplabelA=parent[level_index+upsublabel-1];
                int uplabelB=parent[level_index+upsublabel];
            //    int curlabelA=labelA;
                int curlabelB=labelB;
                int uplabelBR=parent[level_index+upsublabel+1];
                bool bedone=true;
              //  delete_block_CE(block_features,T,nr_partitions,parent,
              //                  nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
              //                  in_channels,labelB,sublabel,int n);
                scalar_t intA = get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                levels,seeds_top_level,cur_level,in_channels,labelA,sublabel, n);
                scalar_t intB = get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                levels,seeds_top_level,cur_level,in_channels,labelB,sublabel, n);
                int upNum=0;
                while(uplabelB!=curlabelB&&uplabelA!=uplabelB)
                {
                    if(uplabelB!=uplabelBR)
                    {
                        bedone = false;
                        break;
                    }
                    if(curyup>0)
                    {
                        int upuplabelB=parent[level_index+upsublabel-step];
                        int upuplabelBR=parent[level_index+upsublabel-step+1];
                        if(upuplabelB==uplabelB&&upuplabelBR!=uplabelB)
                        {
                            bedone = false;
                            break;
                        }
                    }
                  //  delete_block_CE(block_features,T,nr_partitions,parent,
                  //                  nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                   //                 in_channels,uplabelB,upsublabel,int n);
                    intA = intA + get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                levels,seeds_top_level,cur_level,in_channels,uplabelA,upsublabel, n);
                    intB = intB + get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                levels,seeds_top_level,cur_level,in_channels,uplabelB,upsublabel, n);
                //    curlabelA=uplabelA;
                    curlabelB=uplabelB;
                    curyup=curyup-1;
                    upNum++;
                    if(curyup<0) break;
                    upsublabel=curyup*step+x;
                    uplabelA=parent[level_index+upsublabel-1];
                    uplabelB=parent[level_index+upsublabel];
                    uplabelBR=parent[level_index+upsublabel+1];
                }
                int downNum=0;
                if(bedone==true)
                {
                    int curydown=y+1;
                    int downsublabel=curydown*step+x;
                    int downlabelA=parent[level_index+downsublabel-1];
                    int downlabelB=parent[level_index+downsublabel];
                    int downlabelBR=parent[level_index+downsublabel+1];
                 //   curlabelA=labelA;
                    curlabelB=labelB;
                    while(downlabelB!=curlabelB&&downlabelA!=downlabelB)
                    {
                        if(downlabelB!=downlabelBR)
                        {
                            bedone = false;
                            break;
                        }
                        if(curydown<nr_seeds_h-1)
                        {
                            int downdownlabelB=parent[level_index+downsublabel+step];
                            int downdownlabelBR=parent[level_index+downsublabel+step+1];
                            if(downdownlabelB==downlabelB&&downdownlabelBR!=downlabelB)
                            {
                                bedone = false;
                                break;
                            }
                        }
                      //  delete_block_CE(block_features,T,nr_partitions,parent,
                      //                 nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                       //                in_channels,downlabelB,downsublabel,int n);
                        intA = intA + get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                levels,seeds_top_level,cur_level,in_channels,downlabelA,downsublabel, n);
                        intB = intB + get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                levels,seeds_top_level,cur_level,in_channels,downlabelB,downsublabel, n);
                        downNum++;
                     //   curlabelA=downlabelA;
                        curlabelB=downlabelB;
                        curydown=curydown+1;
                        if(curydown>=nr_seeds_h) break;
                        downsublabel=curydown*step+x;
                        downlabelA=parent[level_index+downsublabel-1];
                        downlabelB=parent[level_index+downsublabel];
                        downlabelBR=parent[level_index+downsublabel+1];
                    }
                }
               // float confidence = fabs(intA - intB);
                int space_dim = nr_seeds_h1*nr_seeds_w1;
              //  upNum =0;
              //  downNum=0;
                if ((intA < intB) && bedone==true)
                {

                    block_Tag[space_dim*n+sublabel]=-1;
                    for(int i=1;i<=upNum;i++)
                        {
                            upsublabel=(y-i)*step+x;
                            block_Tag[space_dim*n+upsublabel]=-1;
                        }
                    for(int i=1;i<=downNum;i++)
                        {
                            //curydown=y+i;
                            int  downsublabel=(y+i)*step+x;
                            block_Tag[space_dim*n+downsublabel]=-1;
                        }
                }
                }
             else if(nr_partitions[top_level_index+labelB] > 2)
             {
                bool isSplit = false;
                if ((a22 != a13) && (a22 == a12) && (a22 == a23)) isSplit = true;
			    if ((a22 != a33) && (a22 == a32) && (a22 == a23)) isSplit = true;
                if (!isSplit)
				{
				//    int cursublabel=sublabel;
                    int curyup=y-1;
                    int upsublabel=curyup*step+x;
                    int uplabelA=parent[level_index+upsublabel-1];
                    int uplabelB=parent[level_index+upsublabel];
                  //  int curlabelA=labelA;
                    int curlabelB=labelB;
                    int uplabelBR=parent[level_index+upsublabel+1];


                 //   delete_block_CE(block_features,T,nr_partitions,parent,
                  //                 nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                 //                  in_channels,labelB,sublabel,int n);
                    scalar_t intA = get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                levels,seeds_top_level,cur_level,in_channels,labelA,sublabel, n);
                    scalar_t intB = get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                levels,seeds_top_level,cur_level,in_channels,labelB,sublabel, n);
                    int upNum=0;
                    bool bedone = true;
                    while(uplabelB!=curlabelB&&uplabelA!=uplabelB)
                    {
                        if(uplabelB!=uplabelBR)
                        {
                            bedone = false;
                            break;
                        }
                        if(curyup>0)
                        {
                            int upuplabelB=parent[level_index+upsublabel-step];
                            int upuplabelBR=parent[level_index+upsublabel-step+1];
                            if(upuplabelB==uplabelB&&upuplabelBR!=uplabelB)
                            {
                                bedone = false;
                                break;
                            }
                        }
                     //   delete_block_CE(block_features,T,nr_partitions,parent,
                     //                  nr_seeds_h1, nr_seeds_w1,levels,cur_level,seeds_top_level,
                      //                 in_channels,uplabelB,upsublabel,int n);
                        intA = intA + get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                levels,seeds_top_level,cur_level,in_channels,uplabelA,upsublabel, n);
                        intB = intB + get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                    levels,seeds_top_level,cur_level,in_channels,uplabelB,upsublabel, n);
                     //   curlabelA=uplabelA;
                        curlabelB=uplabelB;
                        curyup=curyup-1;
                        upNum++;
                        if(curyup<0) break;
                        upsublabel=curyup*step+x;
                        uplabelA=parent[level_index+upsublabel-1];
                        uplabelB=parent[level_index+upsublabel];
                        uplabelBR=parent[level_index+upsublabel+1];
                    }
                    int downNum=0;
                    if(bedone==true)
                    {
                     //   curlabelA=labelA;
                        curlabelB=labelB;
                        int curydown=y+1;
                        int downsublabel=curydown*step+x;
                        int downlabelA=parent[level_index+downsublabel-1];
                        int downlabelB=parent[level_index+downsublabel];
                        int downlabelBR=parent[level_index+downsublabel+1];
                       // bool bedone=true;
                        while(downlabelB!=curlabelB&&downlabelA!=downlabelB)
                        {
                            if(downlabelB!=downlabelBR)
                            {
                                bedone = false;
                                break;
                            }
                            if(curydown<nr_seeds_h-1)
                            {
                                int downdownlabelB=parent[level_index+downsublabel+step];
                                int downdownlabelBR=parent[level_index+downsublabel+step+1];
                                if(downdownlabelB==downlabelB&&downdownlabelBR!=downlabelB)
                                {
                                    bedone = false;
                                    break;
                                }
                            }
                         //   delete_block_CE(seeds_top_level, downlabelB, level, downsublabel);
                            intA = intA + get_block_dis_CE(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                levels,seeds_top_level,cur_level,in_channels,downlabelA,downsublabel, n);
                            intB = intB + get_block_dis_CE1(block_features,T, nr_seeds_h1, nr_seeds_w1,
                                                    levels,seeds_top_level,cur_level,in_channels,downlabelB,downsublabel, n);
                         //   intA = intA + get_block_dis_CE(seeds_top_level, downlabelA, level, downsublabel);
                          //  intB = intB + get_block_dis_CE1(seeds_top_level, downlabelB, level, downsublabel);
                            downNum++;
                            //curlabelA=downlabelA;
                            curlabelB=downlabelB;
                            curydown=curydown+1;
                            if(curydown>=nr_seeds_h) break;
                            downsublabel=curydown*step+x;
                            downlabelA=parent[level_index+downsublabel-1];
                            downlabelB=parent[level_index+downsublabel];
                            downlabelBR=parent[level_index+downsublabel+1];
                        }
                    }
                   // float confidence = fabs(intA - intB);
                    int space_dim = nr_seeds_h1*nr_seeds_w1;
                 //   upNum=0;
                 //   downNum=0;
                    if ((intA < intB) &&bedone==true)
                    {

                        block_Tag[space_dim*n+sublabel]=-1;
                        for(int i=1;i<=upNum;i++)
                        {
                            //curyup=y-i;
                            int upsublabel=(y-i)*step+x;
                            block_Tag[space_dim*n+upsublabel]=-1;

                        }
                        for(int i=1;i<=downNum;i++)
                        {
                            //curydown=y+i;
                            int downsublabel=(y+i)*step+x;
                             block_Tag[space_dim*n+downsublabel]=-1;
                        }
                    }
				}
             }
           }
        }

       }
    }
}
/*template <typename scalar_t>
__global__ void update_block_X(const int nthreads,
  scalar_t* labels, scalar_t* parent, scalar_t* block_Tag,const scalar_t* features,  scalar_t* block_features,
  scalar_t *nr_partitions,const int nr_seeds_h1, const int nr_seeds_w1,  int nr_seeds_h,  scalar_t* T,
  const int height, const int width,
  int step,  int cur_level, const int levels,  int seeds_top_level,
  const int in_channels) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < nthreads) {

        int spatial_dim=step*nr_seeds_h;
        int n = index/(spatial_dim);
        int s = index%(spatial_dim);
        int x = s % step;
        int y = s / step;
        if(block_Tag[s]!=0){
           int sublabel = s;
           int child_index= (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1+sublabel;
           int labelA = parent[child_index+block_Tag[index]];
           int labelB = parent[child_index];
           //int level_index = (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1;
           for (int k=0;k<in_channels;k++)
           {
                int block_child_index=((n*levels+cur_level)*in_channels+k)*nr_seeds_h1*nr_seeds_w1+sublabel;
                int block_parent_index=((n*levels+seeds_top_level)*in_channels+k)*nr_seeds_h1*nr_seeds_w1+labelB;
                caffe_gpu_atomic_add(-block_features[block_child_index],block_features+block_parent_index);
            }
         //   int level_index = (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1+sublabel;
            int top_level_index = (n*levels+seeds_top_level)*nr_seeds_h1*nr_seeds_w1+labelB;
            caffe_gpu_atomic_add(-T[child_index],T+top_level_index);
            caffe_gpu_atomic_add(-1,nr_partitions+top_level_index);
           // parent[level_index]=-1;

            for (int k=0;k<in_channels;k++)
            {
                int block_child_index=((n*levels+cur_level)*in_channels+k)*nr_seeds_h1*nr_seeds_w1+sublabel;
                int block_parent_index=((n*levels+seeds_top_level)*in_channels+k)*nr_seeds_h1*nr_seeds_w1+labelA;
             //  block_features[block_parent_index]=block_features[block_parent_index]+block_features[block_child_index];
               caffe_gpu_atomic_add(block_features[block_child_index],block_features+block_parent_index);
            }
         //   int level_index = (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1+sublabel;
            top_level_index = (n*levels+seeds_top_level)*nr_seeds_h1*nr_seeds_w1+labelA;
            caffe_gpu_atomic_add(T[child_index],T+top_level_index);
            caffe_gpu_atomic_add(1,nr_partitions+top_level_index);
            parent[child_index]=labelA;
        }
    }
  }*/
template <typename scalar_t>
__global__ void update_blocks(const int nthreads,
  scalar_t* labels, scalar_t* parent, scalar_t* block_Tag,const scalar_t* features,  scalar_t* block_features,scalar_t *nr_partitions,
  const int nr_seeds_h1, const int nr_seeds_w1, const int nr_seeds_h,  scalar_t* T,const int height, const int width,
  int step,   int cur_level, const int levels,  int seeds_top_level,
  const int in_channels) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < nthreads) {
        int spatial_dim=step*nr_seeds_h;
        int n = index/(spatial_dim);
        int s = index%(spatial_dim);
        int x = s % step;
        int y = s / step;
        if(block_Tag[s]!=0){
           int sublabel = y*step+x;
           int child_index= (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1+sublabel;
           int new_index = child_index+block_Tag[index];
           int labelA = parent[new_index];
           int labelB = parent[child_index];
           //int level_index = (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1;
           for (int k=0;k<in_channels;k++)
           {
                int block_child_index=((n*levels+cur_level)*in_channels+k)*nr_seeds_h1*nr_seeds_w1+sublabel;
                int block_parent_index=((n*levels+seeds_top_level)*in_channels+k)*nr_seeds_h1*nr_seeds_w1+labelB;
               // caffe_gpu_atomic_add(-block_features[block_child_index],block_features+block_parent_index);
                atomicAdd(block_features+block_parent_index,static_cast<scalar_t>(-block_features[block_child_index]));

            }
         //   int level_index = (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1+sublabel;
            int top_level_index = (n*levels+seeds_top_level)*nr_seeds_h1*nr_seeds_w1+labelB;
            atomicAdd(T+top_level_index,-T[child_index]);
            atomicAdd(nr_partitions+top_level_index,-1);
           // parent[level_index]=-1;

            for (int k=0;k<in_channels;k++)
            {
                int block_child_index=((n*levels+cur_level)*in_channels+k)*nr_seeds_h1*nr_seeds_w1+sublabel;
                int block_parent_index=((n*levels+seeds_top_level)*in_channels+k)*nr_seeds_h1*nr_seeds_w1+labelA;
             //  block_features[block_parent_index]=block_features[block_parent_index]+block_features[block_child_index];
                atomicAdd(block_features+block_parent_index,block_features[block_child_index]);
              //  caffe_gpu_atomic_add(block_features[block_child_index],block_features+block_parent_index);
            }
         //   int level_index = (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1+sublabel;
             top_level_index = (n*levels+seeds_top_level)*nr_seeds_h1*nr_seeds_w1+labelA;
            //caffe_gpu_atomic_add(T[child_index],T+top_level_index);
            //caffe_gpu_atomic_add(1,nr_partitions+top_level_index);
            atomicAdd(T+top_level_index,static_cast<scalar_t>(T[child_index]));
            atomicAdd(nr_partitions+top_level_index,static_cast<scalar_t>(1));
            parent[child_index]=labelA;
        }
    }
  }
/*template <typename scalar_t>
__global__ void update_block_Y(const int nthreads,
  scalar_t* labels, scalar_t* parent, scalar_t* block_Tag,const scalar_t* features,  scalar_t* block_features,scalar_t* nr_partitions,
  const int nr_seeds_h1, const int nr_seeds_w1, const int nr_seeds_h,  scalar_t* T,const int height, const int width,
  int step,   int cur_level, const int levels,  int seeds_top_level,
  const int in_channels) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < nthreads) {
        int spatial_dim=step*nr_seeds_h;
        int n = index/(spatial_dim);
        int s = index%(spatial_dim);
        int x = s % step;
        int y = s / step;
        if(block_Tag[s]!=0){
           int sublabel = y*step+x;
           int child_index= (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1+sublabel;
           int labelA = parent[child_index+block_Tag[index]*step];
           int labelB = parent[child_index];
           //int level_index = (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1;
           for (int k=0;k<in_channels;k++)
           {
                int block_child_index=((n*levels+cur_level)*in_channels+k)*nr_seeds_h1*nr_seeds_w1+sublabel;
                int block_parent_index=((n*levels+seeds_top_level)*in_channels+k)*nr_seeds_h1*nr_seeds_w1+labelB;
                atomicAdd(block_features+block_parent_index,static_cast<scalar_t>(-block_features[block_child_index]));
                //caffe_gpu_atomic_add(-block_features[block_child_index],block_features+block_parent_index);
            }
         //   int level_index = (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1+sublabel;
            int top_level_index = (n*levels+seeds_top_level)*nr_seeds_h1*nr_seeds_w1+labelB;
            //caffe_gpu_atomic_add(-T[child_index],T+top_level_index);
            atomicAdd(T+top_level_index,-T[child_index]);
            atomicAdd(nr_partitions+top_level_index,-1);
            //caffe_gpu_atomic_add(-1,nr_partitions+top_level_index);
           // parent[level_index]=-1;

            for (int k=0;k<in_channels;k++)
            {
                int block_child_index=((n*levels+cur_level)*in_channels+k)*nr_seeds_h1*nr_seeds_w1+sublabel;
                int block_parent_index=((n*levels+seeds_top_level)*in_channels+k)*nr_seeds_h1*nr_seeds_w1+labelA;
             //  block_features[block_parent_index]=block_features[block_parent_index]+block_features[block_child_index];
               atomicAdd(block_features+block_parent_index,static_cast<scalar_t>(block_features[block_child_index]));
               //caffe_gpu_atomic_add(block_features[block_child_index],block_features+block_parent_index);
            }
         //   int level_index = (n*levels+cur_level)*nr_seeds_h1*nr_seeds_w1+sublabel;
             top_level_index = (n*levels+seeds_top_level)*nr_seeds_h1*nr_seeds_w1+labelA;
            atomicAdd(T+top_level_index,T[child_index]);
            //caffe_gpu_atomic_add(T[child_index],T+top_level_index);
            //caffe_gpu_atomic_add(1,nr_partitions+top_level_index);
            atomicAdd(nr_partitions+top_level_index,1);
            parent[child_index]=labelA;
        }
    }
  }*/
template <typename scalar_t>
__global__ void Copy_to_Top(const int nthreads,  scalar_t* labels, scalar_t *top_data, const int seeds_top_level, int spatial_dim,
  const int levels) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < nthreads){
        int n = index/spatial_dim;
        int s = index%(spatial_dim);
        int top_level_index=(n*levels+seeds_top_level)*spatial_dim;
        top_data[index]=labels[top_level_index+s];
    }
  }
template <typename scalar_t>
__global__ void get_Total_Color(const int nthreads, const scalar_t* image_data,  int* labels, scalar_t *Av_color,
int spatial_dim,int label_dim,const int levels, const int seeds_top_level) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < nthreads){
        int n = index/spatial_dim;
        int s = index%(spatial_dim);
        int top_level_index=(n*levels+seeds_top_level)*spatial_dim;
        int pixel_label=labels[top_level_index+s];
        for(int k=0;k<3;k++)
        {
         //  int k1 = abs(k-2);
            int out_index=(n*3+k)*label_dim+pixel_label;
            int image_index = (n*3+k)*spatial_dim+s;
          //  int image_index1 = (n*3+k1)*spatial_dim+s;
          //  Av_color[out_index]+=image_data[image_index];
          //  caffe_gpu_atomic_add(image_data[image_index],Av_color+out_index);
            atomicAdd(Av_color+out_index,static_cast<scalar_t>(image_data[image_index]));
        }
    }
  }
template <typename scalar_t>
__global__ void get_AV_Color(const int nthreads, const int* T,  scalar_t *Av_color,const int nr_seeds_h1, const int nr_seeds_w1,
int label_dim,const int levels, const int seeds_top_level) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < nthreads) {
        int n = index/label_dim;
        int s = index%(label_dim);
        int T_index=(n*levels+seeds_top_level)*nr_seeds_h1*nr_seeds_w1+s;
        for(int k=0;k<3;k++){
            int out_index=(n*3+k)*label_dim+s;
            Av_color[out_index]=Av_color[out_index]/(float)T[T_index];
        }
    }
}

template <typename scalar_t>
__global__ void latticesuperpixel_cuda_backward_kernel(
      const int n,
      const int nbatch,
      const int channels,
      const int oheight,
      const int owidth,
      const scalar_t* __restrict__ odata,
      scalar_t* __restrict__ idata) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int spatial_dim = oheight * owidth;
    if (index < n){
          int n1= index/spatial_dim;
          int n2= index%spatial_dim;
          for (int c = 0; c < channels; ++c) {
   		        int64_t pindex = (n1*channels+c)*spatial_dim+n2;
                idata[pindex] = odata[pindex];
          }
     }
  }
}
at::Tensor latticesuperpixel_cuda_forward(
    const at::Tensor input,
    const int seed_h,
    const int seed_w,
    const int nr_levels_) {
  auto output = at::empty_like(input);
  const int height = input.size(2);
  const int width = input.size(3);
  const int num_ = input.size(0);
  const int channels = input.size(1);
  int seeds_h = seed_h;
  int seeds_w = seed_w;
  if(height>width)
  {
   int temp = seeds_h;
   seeds_h = seeds_w;
   seeds_w = temp;
  }
  int nr_seeds_w1 = floor(float(float(width)/float(seeds_w))+0.5);
  int nr_seeds_h1 = floor(float(float(height)/float(seeds_h))+0.5);
  output.resize_({num_, 1, height, width});
  output.zero_();
  //auto labels_ = at::empty_like(input, at::kInt);
  auto labels_ = at::empty_like(input);
  labels_.resize_({num_, nr_levels_, height, width});

  auto parents_ = at::empty_like(input);
  parents_.resize_({num_, nr_levels_, nr_seeds_h1, nr_seeds_w1});

  auto nr_partitions_ = at::empty_like(input);
  nr_partitions_.resize_({num_, nr_levels_, nr_seeds_h1, nr_seeds_w1});

  auto Block_feature_ = at::empty_like(input);
  Block_feature_.resize_({num_, nr_levels_*channels, nr_seeds_h1, nr_seeds_w1});

  auto T_ = at::empty_like(input);
  T_.resize_({num_, nr_levels_, nr_seeds_h1, nr_seeds_w1});

  auto pixel_Tag_ = at::empty_like(input);
  pixel_Tag_.resize_({num_, 1, height, width});

  auto block_Tag_ = at::empty_like(input);
  block_Tag_.resize_({num_, 1, nr_seeds_h1, nr_seeds_w1});

  auto dis = at::empty_like(input);
  dis.resize_({num_, 1, height, width});

   labels_.contiguous();
   labels_.zero_();
   parents_.contiguous();
  parents_.zero_();
   nr_partitions_.contiguous();
   nr_partitions_.zero_();
   input.contiguous();
   Block_feature_.contiguous();
   Block_feature_.zero_();
   T_.contiguous();
   T_.zero_();

   const int totathread = num_ * height * width;
   const int num_threads = 512;
   const int num_kernels = totathread/num_threads+1;
   int nr_seeds_h = nr_seeds_h1;
   int nr_seeds_w = nr_seeds_w1;
   int step_h=seeds_h;
   int step_w=seeds_w;
   int nr_label_W[7];
   int nr_label_H[7];
   nr_label_W[0] = nr_seeds_w1;
   nr_label_H[0] = nr_seeds_h1;
   const int seeds_top_level = nr_levels_-1;
   int level =0;
   AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
    Assign_labels<scalar_t><<<num_kernels, num_threads>>>(
      totathread,
      labels_.data<scalar_t>(),
      parents_.data<scalar_t>(),
      nr_partitions_.data<scalar_t>(),
      input.data<scalar_t>(),
      Block_feature_.data<scalar_t>(),
      T_.data<scalar_t>(),
      nr_seeds_h, nr_seeds_w, height, width,  step_h,  step_w,
      nr_seeds_h1, nr_seeds_w1,  level, nr_levels_, channels);
   }));
  for( level=1;level<nr_levels_;level++){
   nr_seeds_w = floor(float(nr_seeds_w/2.0));
   nr_seeds_h = floor(float(nr_seeds_h/2.0));
   nr_label_W[level] = nr_seeds_w;
   nr_label_H[level] = nr_seeds_h;
   step_w *= 2;
   step_h *= 2;
   AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
    Assign_labels<scalar_t><<<num_kernels, num_threads>>>(
      totathread,
      labels_.data<scalar_t>(),
      parents_.data<scalar_t>(),
      nr_partitions_.data<scalar_t>(),
      input.data<scalar_t>(),
      Block_feature_.data<scalar_t>(),
      T_.data<scalar_t>(),
      nr_seeds_h, nr_seeds_w, height, width,  step_h,  step_w,
      nr_seeds_h1, nr_seeds_w1,  level, nr_levels_, channels);
   }));
  }
  nr_seeds_h = nr_seeds_h1;
  nr_seeds_w = nr_seeds_w1;
  int  totathread1 = num_ * nr_seeds_w * nr_seeds_h;
  int num_kernels1 = totathread1/num_threads+1;
  for(level = 0; level<nr_levels_-1;level++)
   {
      AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
        Compute_block_Feature<scalar_t><<<num_kernels1, num_threads>>>(
      totathread1,
      labels_.data<scalar_t>(),
      parents_.data<scalar_t>(),
      nr_partitions_.data<scalar_t>(),
      Block_feature_.data<scalar_t>(),
      T_.data<scalar_t>(),
      nr_seeds_h1, nr_seeds_w1,nr_seeds_h, nr_seeds_w,
      level, nr_levels_, channels);
      }));
      nr_seeds_w = nr_label_W[level+1];
      nr_seeds_h = nr_label_H[level+1];
      totathread1 = num_ * nr_seeds_w * nr_seeds_h;
      num_kernels1 = totathread1/num_threads+1;
   }

   int totathread4 = num_ * nr_label_W[seeds_top_level]*nr_label_H[seeds_top_level];
   int num_kernels4 = totathread4/num_threads+1;
   nr_seeds_w = nr_label_W[seeds_top_level-1];
   nr_seeds_h = nr_label_H[seeds_top_level-1];
   totathread1 =  num_ * nr_seeds_w * nr_seeds_h;
   num_kernels1 = totathread1/num_threads+1;
   //int num_Tag =  num_ *nr_seeds_h1*nr_seeds_w1;
   int cur_level= nr_levels_-2;
   if(cur_level>0){
      AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
      clear_partitions<scalar_t><<<num_kernels4, num_threads>>>(
      totathread4,
      nr_partitions_.data<scalar_t>(),
      nr_seeds_h1, nr_seeds_w1, nr_label_H[seeds_top_level],
      nr_label_W[seeds_top_level], nr_levels_, seeds_top_level);
      }));
      
      nr_seeds_w = nr_label_W[cur_level-1];
      nr_seeds_h = nr_label_H[cur_level-1];
      totathread1 =  num_ * nr_seeds_w * nr_seeds_h;
      num_kernels1 = totathread1/num_threads+1;
      
      AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
      go_down_one_level<scalar_t><<<num_kernels1, num_threads>>>(
      totathread1,
      parents_.data<scalar_t>(),
      nr_partitions_.data<scalar_t>(),
      nr_seeds_h1, nr_seeds_w1, nr_seeds_h, nr_seeds_w, 
      nr_levels_, cur_level, seeds_top_level);
      }));
   }
   block_Tag_.contiguous();
   for( cur_level= cur_level-1;cur_level>=0;cur_level--){
      //  int nthreads2 = num_ * nr_seeds_w * nr_seeds_h/9;
        int s_width = ceil(float(nr_seeds_w)/3.0);
        int s_height = ceil(float(nr_seeds_h)/3.0);
        int totathread2 = num_ * s_width * s_height;
        int num_kernels2 = totathread2/num_threads+1;  
        for(int shift_y=0;shift_y<3;shift_y++){
         for(int shift_x=0;shift_x<3;shift_x++){
            block_Tag_.zero_();
            AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
             update_block_X_right<scalar_t><<<num_kernels2, num_threads>>>(
             totathread2,
             labels_.data<scalar_t>(),
             parents_.data<scalar_t>(),
             block_Tag_.data<scalar_t>(),
             input.data<scalar_t>(),    
             Block_feature_.data<scalar_t>(),
             nr_partitions_.data<scalar_t>(),
             nr_seeds_h1, nr_seeds_w1, nr_seeds_h,  
             T_.data<scalar_t>(),
             height, width,
             nr_seeds_w, shift_y, shift_x, cur_level, nr_levels_,  
             seeds_top_level, channels);
             }));
            AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
             update_blocks<scalar_t><<<num_kernels1, num_threads>>>(
             totathread1,
             labels_.data<scalar_t>(),
             parents_.data<scalar_t>(),
             block_Tag_.data<scalar_t>(),
             input.data<scalar_t>(),
             Block_feature_.data<scalar_t>(),
             nr_partitions_.data<scalar_t>(),
             nr_seeds_h1, nr_seeds_w1, nr_seeds_h,
             T_.data<scalar_t>(),
             height, width,
             nr_seeds_w,  cur_level, nr_levels_,
             seeds_top_level, channels);
             }));

             block_Tag_.zero_();

             AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
             update_block_X_left<scalar_t><<<num_kernels2, num_threads>>>(
             totathread2,
             labels_.data<scalar_t>(),
             parents_.data<scalar_t>(),
             block_Tag_.data<scalar_t>(),
             input.data<scalar_t>(),
             Block_feature_.data<scalar_t>(),
             nr_partitions_.data<scalar_t>(),
             nr_seeds_h1, nr_seeds_w1, nr_seeds_h,
             T_.data<scalar_t>(),
             height, width,
             nr_seeds_w, shift_y, shift_x, cur_level, nr_levels_,
             seeds_top_level, channels);
             }));

             AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
             update_blocks<scalar_t><<<num_kernels1, num_threads>>>(
             totathread1,
             labels_.data<scalar_t>(),
             parents_.data<scalar_t>(),
             block_Tag_.data<scalar_t>(),
             input.data<scalar_t>(),
             Block_feature_.data<scalar_t>(),
             nr_partitions_.data<scalar_t>(),
             nr_seeds_h1, nr_seeds_w1, nr_seeds_h,
             T_.data<scalar_t>(),
             height, width,
             nr_seeds_w,  cur_level, nr_levels_,
             seeds_top_level, channels);
             }));

             block_Tag_.zero_();

             AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
             update_block_Y_down<scalar_t><<<num_kernels2, num_threads>>>(
             totathread2,
             labels_.data<scalar_t>(),
             parents_.data<scalar_t>(),
             block_Tag_.data<scalar_t>(),
             input.data<scalar_t>(),
             Block_feature_.data<scalar_t>(),
             nr_partitions_.data<scalar_t>(),
             nr_seeds_h1, nr_seeds_w1, nr_seeds_h,
             T_.data<scalar_t>(),
             height, width,
             nr_seeds_w, shift_y, shift_x, cur_level, nr_levels_,
             seeds_top_level, channels);
             }));

             AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
             update_blocks<scalar_t><<<num_kernels1, num_threads>>>(
             totathread1,
             labels_.data<scalar_t>(),
             parents_.data<scalar_t>(),
             block_Tag_.data<scalar_t>(),
             input.data<scalar_t>(),
             Block_feature_.data<scalar_t>(),
             nr_partitions_.data<scalar_t>(),
             nr_seeds_h1, nr_seeds_w1, nr_seeds_h,
             T_.data<scalar_t>(),
             height, width,
             nr_seeds_w,  cur_level, nr_levels_,
             seeds_top_level, channels);
             }));

             block_Tag_.zero_();

             AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
             update_block_Y_up<scalar_t><<<num_kernels2, num_threads>>>(
             totathread2,
             labels_.data<scalar_t>(),
             parents_.data<scalar_t>(),
             block_Tag_.data<scalar_t>(),
             input.data<scalar_t>(),
             Block_feature_.data<scalar_t>(),
             nr_partitions_.data<scalar_t>(),
             nr_seeds_h1, nr_seeds_w1, nr_seeds_h,
             T_.data<scalar_t>(),
             height, width,
             nr_seeds_w, shift_y, shift_x, cur_level, nr_levels_,
             seeds_top_level, channels);
             }));

             AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
             update_blocks<scalar_t><<<num_kernels1, num_threads>>>(
             totathread1,
             labels_.data<scalar_t>(),
             parents_.data<scalar_t>(),
             block_Tag_.data<scalar_t>(),
             input.data<scalar_t>(),
             Block_feature_.data<scalar_t>(),
             nr_partitions_.data<scalar_t>(),
             nr_seeds_h1, nr_seeds_w1, nr_seeds_h,
             T_.data<scalar_t>(),
             height, width,
             nr_seeds_w,  cur_level, nr_levels_,
             seeds_top_level, channels);
             }));
           }
        }
         AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
             update_labels<scalar_t><<<num_kernels, num_threads>>>(
             totathread,
             labels_.data<scalar_t>(),
             parents_.data<scalar_t>(),
             height, width, nr_levels_, cur_level,
             seeds_top_level, nr_seeds_h1,  nr_seeds_w1);
             }));
        if(cur_level>0){
        AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
         clear_partitions<scalar_t><<<num_kernels4, num_threads>>>(
         totathread4,
         nr_partitions_.data<scalar_t>(),
         nr_seeds_h1, nr_seeds_w1, nr_label_H[seeds_top_level],
         nr_label_W[seeds_top_level], nr_levels_, seeds_top_level);
         }));
        nr_seeds_w = nr_label_W[cur_level-1];
        nr_seeds_h = nr_label_H[cur_level-1];
        totathread1 =  num_ * nr_seeds_w * nr_seeds_h;
        num_kernels1 = totathread1/num_threads+1;
        AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
        go_down_one_level<scalar_t><<<num_kernels1, num_threads>>>(
        totathread1,
        parents_.data<scalar_t>(),
        nr_partitions_.data<scalar_t>(),
        nr_seeds_h1, nr_seeds_w1, nr_seeds_h, nr_seeds_w,
        nr_levels_, cur_level, seeds_top_level);
        }));
      }
   }
   int ori_superP = height*width/(nr_label_W[seeds_top_level]*nr_label_H[seeds_top_level]);
   int s_width = ceil(float(width)/3.0);
   int s_height = ceil(float(height)/3.0);
   int totathread5 = num_ * s_width * s_height;
   int num_kernels5 = totathread5/num_threads+1;
   //int num_pixel = num_ * height * width;
   pixel_Tag_.contiguous();
   for(int iter= 0;iter < 10;iter++){
        for(int shift_y=0;shift_y<3;shift_y++){
         for(int shift_x=0;shift_x<3;shift_x++){
             pixel_Tag_.zero_();
             AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
             update_pixels_X_right<scalar_t><<<num_kernels5, num_threads>>>(
             totathread5,
             labels_.data<scalar_t>(),
             parents_.data<scalar_t>(),
             pixel_Tag_.data<scalar_t>(),
             input.data<scalar_t>(),
             Block_feature_.data<scalar_t>(),
             T_.data<scalar_t>(),
             dis.data<scalar_t>(),
             nr_seeds_h1, nr_seeds_w1,
             height, width,
             shift_x, shift_y,nr_levels_,
             seeds_top_level,ori_superP,channels);
             }));

             AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
             update_pixels<scalar_t><<<num_kernels, num_threads>>>(
             totathread,
             Block_feature_.data<scalar_t>(),
             T_.data<scalar_t>(),
             input.data<scalar_t>(),
             labels_.data<scalar_t>(),
             pixel_Tag_.data<scalar_t>(),
             nr_seeds_h1, nr_seeds_w1, nr_levels_,
             seeds_top_level,height, width, channels);
             }));

             pixel_Tag_.zero_();
             AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
             update_pixels_X_left<scalar_t><<<num_kernels5, num_threads>>>(
             totathread5,
             labels_.data<scalar_t>(),
             parents_.data<scalar_t>(),
             pixel_Tag_.data<scalar_t>(),
             input.data<scalar_t>(),
             Block_feature_.data<scalar_t>(),
             T_.data<scalar_t>(),
             nr_seeds_h1, nr_seeds_w1,
             height, width,
             shift_x, shift_y,nr_levels_,
             seeds_top_level,ori_superP,channels);
             }));

             AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
             update_pixels<scalar_t><<<num_kernels, num_threads>>>(
             totathread,
             Block_feature_.data<scalar_t>(),
             T_.data<scalar_t>(),
             input.data<scalar_t>(),
             labels_.data<scalar_t>(),
             pixel_Tag_.data<scalar_t>(),
             nr_seeds_h1, nr_seeds_w1, nr_levels_,
             seeds_top_level,height, width, channels);
             }));

             pixel_Tag_.zero_();
             AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
             update_pixels_Y_down<scalar_t><<<num_kernels5, num_threads>>>(
             totathread5,
             labels_.data<scalar_t>(),
             parents_.data<scalar_t>(),
             pixel_Tag_.data<scalar_t>(),
             input.data<scalar_t>(),
             Block_feature_.data<scalar_t>(),
             T_.data<scalar_t>(),
             nr_seeds_h1, nr_seeds_w1,
             height, width,
             shift_x, shift_y,nr_levels_,
             seeds_top_level,ori_superP,channels);
             }));

             AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
             update_pixels<scalar_t><<<num_kernels, num_threads>>>(
             totathread,
             Block_feature_.data<scalar_t>(),
             T_.data<scalar_t>(),
             input.data<scalar_t>(),
             labels_.data<scalar_t>(),
             pixel_Tag_.data<scalar_t>(),
             nr_seeds_h1, nr_seeds_w1, nr_levels_,
             seeds_top_level,height, width, channels);
             }));

             pixel_Tag_.zero_();
             AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
             update_pixels_Y_up<scalar_t><<<num_kernels5, num_threads>>>(
             totathread5,
             labels_.data<scalar_t>(),
             parents_.data<scalar_t>(),
             pixel_Tag_.data<scalar_t>(),
             input.data<scalar_t>(),
             Block_feature_.data<scalar_t>(),
             T_.data<scalar_t>(),
             nr_seeds_h1, nr_seeds_w1,
             height, width,
             shift_x, shift_y,nr_levels_,
             seeds_top_level,ori_superP,channels);
             }));

             AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
             update_pixels<scalar_t><<<num_kernels, num_threads>>>(
             totathread,
             Block_feature_.data<scalar_t>(),
             T_.data<scalar_t>(),
             input.data<scalar_t>(),
             labels_.data<scalar_t>(),
             pixel_Tag_.data<scalar_t>(),
             nr_seeds_h1, nr_seeds_w1, nr_levels_,
             seeds_top_level,height, width, channels);
             }));
           }
        }
      }
  output.contiguous();
  int spatial_dim = height * width;
  AT_DISPATCH_FLOATING_TYPES(input.type(), "latticesuperpixel_forward_cuda", ([&] {
    Copy_to_Top<scalar_t><<<num_kernels, num_threads>>>(
    totathread,
    labels_.data<scalar_t>(),
    output.data<scalar_t>(),
    seeds_top_level, spatial_dim,
    nr_levels_);
  }));
  return output;
}

at::Tensor latticesuperpixel_cuda_backward(const at::Tensor& grad_output){
 const int nbatch = grad_output.size(0);
  const int channels = grad_output.size(1);
  const int output_height = grad_output.size(2);
  const int output_width = grad_output.size(3);
   auto new_grad_output = at::empty_like(grad_output);
   new_grad_output.resize_({nbatch, channels, output_height, output_width});
  new_grad_output.contiguous();
  new_grad_output.zero_();
 const int totlalsize = nbatch * output_height * output_width;
  const size_t num_kernels = totlalsize/512+1;
  const int num_threads = 512;
 AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "latticesuperpixel_backward_cuda", ([&] {
    latticesuperpixel_cuda_backward_kernel<scalar_t><<<num_kernels, num_threads>>>(
      totlalsize,
      nbatch, channels,
      output_height,
      output_width,
      grad_output.data<scalar_t>(),
      new_grad_output.data<scalar_t>());
  }));

  return new_grad_output;
}
