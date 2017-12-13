
//
//  CLsolver2dclass.cpp
//  tokuen1
//
//  Created by Tatsuya Minagawa on 2017/09/07.
//  Copyright © 2017年 Tatsuya Minagawa. All rights reserved.
//

#define IX(i,j) ((i)+(N+2)*(j))
#define IX3(i,j,k) ((i)+(N+2)*(j)+(N+2)*(N+2)*(k))
#define SWAP(x0,x) {float * tmp=x0;x0=x;x=tmp;}

#define SWAP_mem(x0,x) {cl_mem tmp=x0;x0=x;x=tmp;}
//#define SWAP_int(x0,x) {int * tmp=x0;x0=x;x=tmp;}

#define CL_DEVICE_ID 2

/*
 
 もしIX3が1つなってしまったなら
 i の増加は そのまま
 j の増減は (N+2)
 k の増減は (N+2) * (N+2)
 
 */

#include <GLUT/glut.h>
#include <iostream>

#include <OpenCL/opencl.h>

#include <limits>
#include <math.h>

#define MAX_SOURCE_SIZE (0x100000)

#define MAX_PLATFORM_IDS 8
#define MAX_DEVICE_IDS 8

#define BUFFER_SIZE 4096

class CLFluid3d{
    
public:
    
    int N;
    float dt, diff, visc;
    float force, source;
    
    float * u, * v, * w ,* u_prev, * v_prev, * w_prev , *temp;
    float * dens, * dens_prev;
    
    int layer;
    
    cl_context context = NULL; //コンテキスト
    cl_command_queue command_queue = NULL; //コマンドキュー
    cl_program program = NULL; //カーネルプログラム
    cl_kernel kernel = NULL;
    
    cl_kernel kernel_lin_solve = NULL;
    cl_kernel kernel_add_source3d = NULL;
    cl_kernel kernel_set_bnd_x2 = NULL;
    cl_kernel kernel_set_bnd_x1 = NULL;
    cl_kernel kernel_set_bnd_x0 = NULL;
    cl_kernel kernel_mem_copy = NULL;
    cl_kernel kernel_project_1 = NULL;
    cl_kernel kernel_project_2 = NULL;
    cl_kernel kernel_advect = NULL;
    cl_kernel kernel_clear = NULL;
    cl_kernel kernel_mem_set = NULL;
    
    cl_platform_id platform_id[MAX_PLATFORM_IDS]; //利用できるプラットフォーム
    cl_uint ret_num_platforms; //実際に利用できるプラットフォーム数
    cl_device_id device_id[MAX_DEVICE_IDS]; //利用できるデバイス
    cl_uint ret_num_devices; //実際に利用できるデバイス
    
    cl_int ret;
    
    char value[BUFFER_SIZE];
    int core;
    size_t size;
    
    cl_mem memobjA = NULL;
    cl_mem memobjB = NULL;
    cl_mem memobjC = NULL;
    
    cl_mem memobjD = NULL;
    
    cl_mem mem_u = NULL;
    cl_mem mem_v = NULL;
    cl_mem mem_w = NULL;
    cl_mem mem_u_prev = NULL;
    cl_mem mem_v_prev = NULL;
    cl_mem mem_w_prev = NULL;
    cl_mem mem_dens = NULL;
    cl_mem mem_dens_prev = NULL;
    
    cl_mem mem_tmp = NULL;
    
    //コンストラクタ
    CLFluid3d(){
        N = 64;
        dt = 0.1f;
        diff = 0.0f;
        visc = 0.0f;
        force = 5.0f;
        source = 100.0f;
        if ( !allocate_data () ) exit ( 1 );
        clear_data ();
        layer = 30;
        
        CLinit();
    }
    
    //デストラクタ
    ~CLFluid3d(){
        free_data();
        CLfree();
    }
    
    void CLinit(){
        FILE *fp;
        char fileName[] = "/Users/cha84rakanal/Documents/2017Projects/tokuen/tokuen1/fluid3d_cl/solver.cl";
        char *source_str;
        size_t source_size;
        
        /* カーネルを含むソースコードをロード */
        fp = fopen(fileName, "r");
        if (!fp) {
            fprintf(stderr, "Failed to load kernel.\n");
            exit(1);
        }
        source_str = (char*)malloc(MAX_SOURCE_SIZE);
        source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose( fp );
        
        ret = clGetPlatformIDs(1, platform_id, &ret_num_platforms);
        if (ret != CL_SUCCESS) {
            printf("Error (clGetPlatformIDs): %d\n", ret);
            exit(1);
        }
        
        ret = clGetPlatformInfo(platform_id[0], CL_PLATFORM_NAME, BUFFER_SIZE, value, &size);
        printf("Platform: %s\n", value);
        //ret = clGetPlatformInfo(platform_id[0], CL_PLATFORM_VERSION, BUFFER_SIZE, value, &size);
        //printf("CL_PLATFORM_VERSION: %s\n", value);
        
        ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_ALL, MAX_DEVICE_IDS, device_id, &ret_num_devices);
        
        if (ret != CL_SUCCESS) {
            printf("Error (clGetDeviceIDs): %d\n", ret);
            exit(1);
        }
        
        
        for (int didx = 0; didx < ret_num_devices; ++didx) {
            ret = clGetDeviceInfo(device_id[didx], CL_DEVICE_NAME, BUFFER_SIZE, value, &size);
            printf("\tDevice No.%d: %s\n",didx, value);
            ret = clGetDeviceInfo(device_id[didx], CL_DEVICE_MAX_COMPUTE_UNITS, BUFFER_SIZE, &core, &size);
            printf("  Compute Unit: %d\n", core);
            //ret = clGetDeviceInfo(device_id[didx], CL_DEVICE_VERSION, BUFFER_SIZE, value, &size);
            //printf("  CL_DEVICE_VERSION: %s\n", value);
        }
        
        if (ret_num_devices == 0) {
            printf("Error Device Not Exist:\n");
            exit(1);
        }
        
        context = clCreateContext( NULL, 3, device_id, NULL, NULL, &ret);
        
        command_queue = clCreateCommandQueue(context, device_id[CL_DEVICE_ID], 0, &ret);
        //command_queue_cpu = clCreateCommandQueue(context, device_id[0], 0, &ret);
        
        memobjA = clCreateBuffer(context, CL_MEM_READ_WRITE,(N+2)*(N+2)*(N+2) * sizeof(float), NULL, &ret);
        memobjB = clCreateBuffer(context, CL_MEM_READ_WRITE,(N+2)*(N+2)*(N+2) * sizeof(float), NULL, &ret);
        memobjC = clCreateBuffer(context, CL_MEM_READ_WRITE,(N+2)*(N+2)*(N+2) * sizeof(float), NULL, &ret);
        
        memobjD = clCreateBuffer(context, CL_MEM_READ_WRITE,(N+2)*(N+2)*(N+2) * sizeof(float), NULL, &ret);
        
        mem_u = clCreateBuffer(context, CL_MEM_READ_WRITE,(N+2)*(N+2)*(N+2) * sizeof(float), NULL, &ret);
        mem_v = clCreateBuffer(context, CL_MEM_READ_WRITE,(N+2)*(N+2)*(N+2) * sizeof(float), NULL, &ret);
        mem_w = clCreateBuffer(context, CL_MEM_READ_WRITE,(N+2)*(N+2)*(N+2) * sizeof(float), NULL, &ret);
        mem_u_prev = clCreateBuffer(context, CL_MEM_READ_WRITE,(N+2)*(N+2)*(N+2) * sizeof(float), NULL, &ret);
        mem_v_prev = clCreateBuffer(context, CL_MEM_READ_WRITE,(N+2)*(N+2)*(N+2) * sizeof(float), NULL, &ret);
        mem_w_prev = clCreateBuffer(context, CL_MEM_READ_WRITE,(N+2)*(N+2)*(N+2) * sizeof(float), NULL, &ret);
        mem_dens = clCreateBuffer(context, CL_MEM_READ_WRITE,(N+2)*(N+2)*(N+2) * sizeof(float), NULL, &ret);
        mem_dens_prev = clCreateBuffer(context, CL_MEM_READ_WRITE,(N+2)*(N+2)*(N+2) * sizeof(float), NULL, &ret);
        mem_tmp = clCreateBuffer(context, CL_MEM_READ_WRITE,(N+2)*(N+2)*(N+2) * sizeof(float), NULL, &ret);
        
        /*読み込んだソースからカーネルプログラムを作成*/
        program = clCreateProgramWithSource(context, 1, (const char **)&source_str,(const size_t *)&source_size, &ret);
        /*カーネルプログラムをビルド*/
        ret = clBuildProgram(program, 3, device_id, NULL, NULL, NULL);
        
        for (int didx = 0; didx < ret_num_devices; ++didx) {
            ret = clGetProgramBuildInfo(program, device_id[didx], CL_PROGRAM_BUILD_LOG, BUFFER_SIZE, value, &size);
            printf("[ERROR:%d]\t%s\n",didx,value);
        }
        
        kernel_lin_solve = clCreateKernel(program, "lin_solve", &ret);
        kernel_add_source3d = clCreateKernel(program, "add_source", &ret);
        kernel_set_bnd_x2 = clCreateKernel(program, "set_bnd_x2", &ret);
        kernel_set_bnd_x1 = clCreateKernel(program, "set_bnd_x1", &ret);
        kernel_set_bnd_x0 = clCreateKernel(program, "set_bnd_x0", &ret);
        kernel_mem_copy = clCreateKernel(program, "mem_copy", &ret);
        kernel_clear = clCreateKernel(program, "mem_clear", &ret);
        kernel_project_1 = clCreateKernel(program, "project_1", &ret);
        kernel_project_2 = clCreateKernel(program, "project_2", &ret);
        kernel_advect = clCreateKernel(program, "advect", &ret);
        kernel_mem_set = clCreateKernel(program, "mem_set", &ret);
        
        ret = clEnqueueWriteBuffer(command_queue,mem_u , CL_TRUE, 0, (N+2)*(N+2)*(N+2)*sizeof(float), u, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue,mem_v , CL_TRUE, 0, (N+2)*(N+2)*(N+2)*sizeof(float), v, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue,mem_w , CL_TRUE, 0, (N+2)*(N+2)*(N+2)*sizeof(float), w, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue,mem_dens , CL_TRUE, 0, (N+2)*(N+2)*(N+2)*sizeof(float), dens, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue,mem_u_prev , CL_TRUE, 0, (N+2)*(N+2)*(N+2)*sizeof(float), u_prev, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue,mem_v_prev , CL_TRUE, 0, (N+2)*(N+2)*(N+2)*sizeof(float), v_prev, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue,mem_w_prev , CL_TRUE, 0, (N+2)*(N+2)*(N+2)*sizeof(float), w_prev, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue,mem_dens_prev , CL_TRUE, 0, (N+2)*(N+2)*(N+2)*sizeof(float), dens_prev, 0, NULL, NULL);
        
        if (ret != CL_SUCCESS) {
            printf("Build Not Success\n");
            exit(1);
        }
        
    }
    
    void CLfree(){
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        
        ret = clReleaseMemObject(memobjA);
        ret = clReleaseMemObject(memobjB);
        ret = clReleaseMemObject(memobjC);
        
        ret = clReleaseMemObject(memobjD);
        
        
        ret = clReleaseMemObject(mem_u);
        ret = clReleaseMemObject(mem_v);
        ret = clReleaseMemObject(mem_w);
        ret = clReleaseMemObject(mem_u_prev);
        ret = clReleaseMemObject(mem_v_prev);
        ret = clReleaseMemObject(mem_w_prev);
        ret = clReleaseMemObject(mem_dens);
        ret = clReleaseMemObject(mem_dens_prev);
        ret = clReleaseMemObject(mem_tmp);
        
        ret = clReleaseKernel(kernel_lin_solve);
        ret = clReleaseKernel(kernel_add_source3d);
        ret = clReleaseKernel(kernel_set_bnd_x2);
        ret = clReleaseKernel(kernel_set_bnd_x1);
        ret = clReleaseKernel(kernel_set_bnd_x0);
        ret = clReleaseKernel(kernel_mem_copy);
        ret = clReleaseKernel(kernel_project_1);
        ret = clReleaseKernel(kernel_project_2);
        ret = clReleaseKernel(kernel_advect);
        ret = clReleaseKernel(kernel_clear);
        ret = clReleaseKernel(kernel_mem_set);
        
        ret = clReleaseProgram(program);
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);
    }
    
    void free_data ( void ){
        if ( u ) free ( u );
        if ( v ) free ( v );
        if ( u_prev ) free ( u_prev );
        if ( w ) free ( w );
        if ( w_prev ) free ( w_prev );
        if ( v_prev ) free ( v_prev );
        if ( dens ) free ( dens );
        if ( dens_prev ) free ( dens_prev );
        if ( temp ) free (temp);
    }
    
    void clear_data ( void ){
        int i, size=(N+2)*(N+2)*(N+2);
        
        for ( i=0 ; i<size ; i++ ) {
            u[i] = v[i] = u_prev[i] = w[i] = w_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = temp[i] = 0;
        }
    }
    
    int allocate_data ( void ){
        int size = (N+2)*(N+2)*(N+2);
        
        u            = (float *) malloc ( size*sizeof(float) );
        v            = (float *) malloc ( size*sizeof(float) );
        w            = (float *) malloc ( size*sizeof(float) );
        u_prev        = (float *) malloc ( size*sizeof(float) );
        v_prev        = (float *) malloc ( size*sizeof(float) );
        w_prev        = (float *) malloc ( size*sizeof(float) );
        dens        = (float *) malloc ( size*sizeof(float) );
        dens_prev    = (float *) malloc ( size*sizeof(float) );
        temp        = (float *) malloc ( size*sizeof(float) );
        
        if ( !u || !v || !u_prev || !v_prev || !w || !w_prev || !dens || !dens_prev || !temp) {
            fprintf ( stderr, "cannot allocate data\n" );
            return ( 0 );
        }
        
        return ( 1 );
    }
    
    void draw_velocity_2d ( void ){
        int i, j;
        float x, y, h;
        
        h = 1.0f/N;
        
        glColor3f ( 1.0f, 1.0f, 1.0f );
        glLineWidth ( 1.0f );
        
        glBegin ( GL_LINES );
        
        int k = layer;
        
        for ( i=1 ; i<=N ; i++ ) {
            x = (i-0.5f)*h;
            for ( j=1 ; j<=N ; j++ ) {
                y = (j-0.5f)*h;
                
                glVertex2f ( x, y );
                glVertex2f ( x+u[IX3(i,j,k)], y+v[IX3(i,j,k)] );
            }
        }
        glEnd ();
    }
    
    void draw_density_2d ( void ){
        int i, j;
        float x, y, h, d00, d01, d10, d11;
        
        h = 1.0f/N;
        
        glBegin ( GL_QUADS );
        
        int k = layer;
        
        for ( i=0 ; i<=N ; i++ ) {
            x = (i-0.5f)*h;
            for ( j=0 ; j<=N ; j++ ) {
                y = (j-0.5f)*h;
                
                d00 = dens[IX3(i,j,k)];
                d01 = dens[IX3(i,j+1,k)];
                d10 = dens[IX3(i+1,j,k)];
                d11 = dens[IX3(i+1,j+1,k)];
                
                glColor3f ( d00, d00, d00 ); glVertex2f ( x, y );
                glColor3f ( d10, d10, d10 ); glVertex2f ( x+h, y );
                glColor3f ( d11, d11, d11 ); glVertex2f ( x+h, y+h );
                glColor3f ( d01, d01, d01 ); glVertex2f ( x, y+h );
            }
        }
        glEnd ();
    }
    
    void cl_advect3d ( int N, int b, cl_mem d, cl_mem d0, cl_mem u,cl_mem v, cl_mem w,float dt ){
        
        ret = clSetKernelArg(kernel_advect, 0, sizeof(int), &N);
        ret = clSetKernelArg(kernel_advect, 1, sizeof(int), &b);
        ret = clSetKernelArg(kernel_advect, 2, sizeof(cl_mem), (void *)&d);
        ret = clSetKernelArg(kernel_advect, 3, sizeof(cl_mem), (void *)&d0);
        ret = clSetKernelArg(kernel_advect, 4, sizeof(cl_mem), (void *)&u);
        ret = clSetKernelArg(kernel_advect, 5, sizeof(cl_mem), (void *)&v);
        ret = clSetKernelArg(kernel_advect, 6, sizeof(cl_mem), (void *)&w);
        ret = clSetKernelArg(kernel_advect, 7, sizeof(float), (void *)&dt);
        
        size_t global_item_size = (N+2)*(N+2)*(N+2);
        size_t local_item_size = 8;
        
        ret = clEnqueueNDRangeKernel(command_queue, kernel_advect, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
        
        cl_set_bnd3d ( N, b,d );
    }
    
    
    void cl_project3d ( int N, cl_mem u, cl_mem v,cl_mem w, cl_mem p, cl_mem div){
        
        ret = clSetKernelArg(kernel_project_1, 0, sizeof(int), &N);
        ret = clSetKernelArg(kernel_project_1, 1, sizeof(cl_mem), (void *)&u);
        ret = clSetKernelArg(kernel_project_1, 2, sizeof(cl_mem), (void *)&v);
        ret = clSetKernelArg(kernel_project_1, 3, sizeof(cl_mem), (void *)&w);
        ret = clSetKernelArg(kernel_project_1, 4, sizeof(cl_mem), (void *)&p);
        ret = clSetKernelArg(kernel_project_1, 5, sizeof(cl_mem), (void *)&div);
        
        size_t global_item_size = (N+2)*(N+2)*(N+2);
        size_t local_item_size = 8;
        
        ret = clEnqueueNDRangeKernel(command_queue, kernel_project_1, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
        
        
        cl_set_bnd3d( N, 0, div );
        cl_set_bnd3d ( N, 0, p );
        cl_lin_solve3d ( N, 0, p, div, 1, 6 );
        
        ret = clSetKernelArg(kernel_project_2, 0, sizeof(int), &N);
        ret = clSetKernelArg(kernel_project_2, 1, sizeof(cl_mem), (void *)&u);
        ret = clSetKernelArg(kernel_project_2, 2, sizeof(cl_mem), (void *)&v);
        ret = clSetKernelArg(kernel_project_2, 3, sizeof(cl_mem), (void *)&w);
        ret = clSetKernelArg(kernel_project_2, 4, sizeof(cl_mem), (void *)&p);
        
        global_item_size = (N+2)*(N+2)*(N+2);
        local_item_size = 8;
        
        ret = clEnqueueNDRangeKernel(command_queue, kernel_project_2, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
        
        cl_set_bnd3d( N,1, u );
        cl_set_bnd3d( N,2, v );
        cl_set_bnd3d( N,3, w );
    }

    void cl_add_source3d ( int N, cl_mem x, cl_mem s, float dt ){
        
        ret = clSetKernelArg(kernel_add_source3d, 0, sizeof(cl_mem), (void *)&x);
        ret = clSetKernelArg(kernel_add_source3d, 1, sizeof(cl_mem), (void *)&s);
        ret = clSetKernelArg(kernel_add_source3d, 2, sizeof(float), &dt);
        
        if(ret != CL_SUCCESS){printf("Set Kernel Arg Not Success\n");exit(1);}
        
        size_t global_item_size = (N+2)*(N+2)*(N+2);
        size_t local_item_size = 8;
        
        ret = clEnqueueNDRangeKernel(command_queue, kernel_add_source3d, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    }
    
    void cl_set_bnd3d ( int N, int b, cl_mem x ){
        
        size_t global_item_size = (N+2) * (N+2);
        size_t local_item_size = 8;
        ret = clSetKernelArg(kernel_set_bnd_x2, 0, sizeof(int), &N);
        ret = clSetKernelArg(kernel_set_bnd_x2, 1, sizeof(cl_mem), (void *)&x);
        ret = clSetKernelArg(kernel_set_bnd_x2, 2, sizeof(int), (void *)&b);
        ret = clEnqueueNDRangeKernel(command_queue, kernel_set_bnd_x2, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

        global_item_size = (N+2);
        local_item_size = 8;
        ret = clSetKernelArg(kernel_set_bnd_x1, 0, sizeof(int), &N);
        ret = clSetKernelArg(kernel_set_bnd_x1, 1, sizeof(cl_mem), (void *)&x);
        ret = clEnqueueNDRangeKernel(command_queue, kernel_set_bnd_x1, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
        
        ret = clSetKernelArg(kernel_set_bnd_x0, 0, sizeof(int), &N);
        ret = clSetKernelArg(kernel_set_bnd_x0, 1, sizeof(cl_mem), (void *)&x);
        ret = clEnqueueTask(command_queue,kernel_set_bnd_x0, 0, NULL,NULL);
        
    }
    
    void cl_lin_solve3d( int N, int b, cl_mem x, cl_mem x0, float a, float c ){
        
        
        size_t global_item_size = (N+2) * (N+2) * (N+2);
        size_t local_item_size = 8;
        ret = clSetKernelArg(kernel_mem_copy, 0, sizeof(cl_mem), (void *)&mem_tmp);
        ret = clSetKernelArg(kernel_mem_copy, 1, sizeof(cl_mem), (void *)&x0);
        ret = clEnqueueNDRangeKernel(command_queue, kernel_mem_copy, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

        ret = clSetKernelArg(kernel_lin_solve, 0, sizeof(int), &N);
        ret = clSetKernelArg(kernel_lin_solve, 1, sizeof(int), &b);
        ret = clSetKernelArg(kernel_lin_solve, 2, sizeof(cl_mem), (void *)&x);
        ret = clSetKernelArg(kernel_lin_solve, 3, sizeof(cl_mem), (void *)&x0);
        ret = clSetKernelArg(kernel_lin_solve, 4, sizeof(cl_mem), (void *)&mem_tmp);
        ret = clSetKernelArg(kernel_lin_solve, 5, sizeof(float), &a);
        ret = clSetKernelArg(kernel_lin_solve, 6, sizeof(float), &c);


        global_item_size = (N+2)*(N+2)*(N+2);
        local_item_size = 8;
        
        for(int l = 0;l < 25;l++){
            
            ret = clEnqueueNDRangeKernel(command_queue, kernel_lin_solve, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
            
            cl_set_bnd3d (N, b, x);
            
            ret = clSetKernelArg(kernel_mem_copy, 0, sizeof(cl_mem), (void *)&x0);
            ret = clSetKernelArg(kernel_mem_copy, 1, sizeof(cl_mem), (void *)&x);
            ret = clEnqueueNDRangeKernel(command_queue, kernel_mem_copy, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
            
        }
        
        ret = clSetKernelArg(kernel_mem_copy, 0, sizeof(cl_mem), (void *)&x0);
        ret = clSetKernelArg(kernel_mem_copy, 1, sizeof(cl_mem), (void *)&x);
        ret = clEnqueueNDRangeKernel(command_queue, kernel_mem_copy, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    }
    
    void cl_diffuse3d ( int N, int b, cl_mem x, cl_mem x0, float diff, float dt ){
        float a = dt*diff*N*N*N;
        cl_lin_solve3d ( N, b, x, x0, a, 1+6*a );
    }
    
    void cl_step(){
        
        /*****
         ここに力を加える
         *****/
        
        size_t global_item_size = (N+2) * (N+2) * (N+2);
        size_t local_item_size = 8;
        ret = clSetKernelArg(kernel_clear, 0, sizeof(cl_mem), (void *)&mem_dens_prev);
        ret = clEnqueueNDRangeKernel(command_queue, kernel_clear, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
        ret = clSetKernelArg(kernel_clear, 0, sizeof(cl_mem), (void *)&mem_u_prev);
        ret = clEnqueueNDRangeKernel(command_queue, kernel_clear, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
        ret = clSetKernelArg(kernel_clear, 0, sizeof(cl_mem), (void *)&mem_v_prev);
        ret = clEnqueueNDRangeKernel(command_queue, kernel_clear, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
        ret = clSetKernelArg(kernel_clear, 0, sizeof(cl_mem), (void *)&mem_w_prev);
        ret = clEnqueueNDRangeKernel(command_queue, kernel_clear, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
        
        int id = IX3(26,39,30);
        
        ret = clSetKernelArg(kernel_mem_set, 0, sizeof(cl_mem), (void *)&mem_v_prev);
        ret = clSetKernelArg(kernel_mem_set, 1, sizeof(int), &id);
        ret = clSetKernelArg(kernel_mem_set, 2, sizeof(float), &force);
        ret = clEnqueueTask(command_queue,kernel_mem_set, 0, NULL,NULL);
        
        ret = clSetKernelArg(kernel_mem_set, 0, sizeof(cl_mem), (void *)&mem_dens_prev);
        ret = clSetKernelArg(kernel_mem_set, 1, sizeof(int), &id);
        ret = clSetKernelArg(kernel_mem_set, 2, sizeof(float), &source);
        ret = clEnqueueTask(command_queue,kernel_mem_set, 0, NULL,NULL);
        
        /** 速度場 **/
        cl_add_source3d ( N, mem_u, mem_u_prev, dt );
        cl_add_source3d ( N, mem_v, mem_v_prev, dt );
        cl_add_source3d ( N, mem_w, mem_w_prev, dt );
        
        SWAP_mem ( mem_u_prev, mem_u );
        cl_diffuse3d ( N, 1, mem_u, mem_u_prev, visc, dt );
        SWAP_mem ( mem_v_prev, mem_v );
        cl_diffuse3d ( N, 2, mem_v, mem_v_prev, visc, dt );
        SWAP_mem ( mem_w_prev, mem_w );
        cl_diffuse3d ( N, 3, mem_w, mem_w_prev, visc, dt );
        
        cl_project3d ( N, mem_u, mem_v, mem_w, mem_u_prev, mem_v_prev );
        
        SWAP_mem ( mem_u_prev, mem_u );
        SWAP_mem ( mem_v_prev, mem_v );
        SWAP_mem ( mem_w_prev, mem_w );
        
        cl_advect3d ( N, 1, mem_u, mem_u_prev, mem_u_prev, mem_v_prev,mem_w_prev, dt );
        cl_advect3d ( N, 2, mem_v, mem_v_prev, mem_u_prev, mem_v_prev,mem_w_prev, dt );
        cl_advect3d ( N, 3, mem_w, mem_w_prev, mem_u_prev, mem_v_prev,mem_w_prev, dt );

        cl_project3d ( N, mem_u, mem_v, mem_w, mem_u_prev, mem_v_prev );
        
        /** 密度場 **/
        cl_add_source3d ( N, mem_dens, mem_dens_prev, dt );
        SWAP_mem ( mem_dens_prev, mem_dens );
        cl_diffuse3d ( N, 0, mem_dens, mem_dens_prev, diff, dt );
        SWAP_mem ( mem_dens_prev, mem_dens );
        
        cl_advect3d ( N, 0, mem_dens, mem_dens_prev, mem_u, mem_v, mem_w, dt );
        
        ret = clEnqueueReadBuffer(command_queue, mem_dens, CL_TRUE, 0, (N+2)*(N+2)*(N+2)* sizeof(float),dens, 0, NULL, NULL);
        ret = clEnqueueReadBuffer(command_queue, mem_u, CL_TRUE, 0, (N+2)*(N+2)*(N+2)* sizeof(float),u, 0, NULL, NULL);
        ret = clEnqueueReadBuffer(command_queue, mem_v, CL_TRUE, 0, (N+2)*(N+2)*(N+2)* sizeof(float),v, 0, NULL, NULL);
        ret = clEnqueueReadBuffer(command_queue, mem_w, CL_TRUE, 0, (N+2)*(N+2)*(N+2)* sizeof(float),w, 0, NULL, NULL);
        
    }
    
};

