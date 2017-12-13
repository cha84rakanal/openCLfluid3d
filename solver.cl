#define IX3(i,j,k) ((i)+(N+2)*(j)+(N+2)*(N+2)*(k))

__kernel void lin_solve(int N,int b,__global float * x,__global float * x0,__global float *temp,float a,float c){
    
    int id = get_global_id(0);
    
    int k = (int)id/((N+2)*(N+2));
    int j = (int)(id%((N+2)*(N+2)))/(N+2);
    int i = (int)(id%((N+2)*(N+2)))%(N+2);
    
    if(i == 0 || i == N+1 || j == 0 || j == N+1 || k == 0 || k == N+1){
        x[id] = 0.0;
        return;
    }else
        x[id] = (temp[id] + a * ( x0[id-1] + x0[id+1] + x0[id-(N+2)] + x0[id+(N+2)] + x0[id-(N+2)*(N+2)] + x0[id+(N+2)*(N+2)]))/c;
    
}

__kernel void mem_copy(__global float * dist,__global float * src){
    int id = get_global_id(0);
    dist[id] = src[id];
}

__kernel void mem_clear(__global float * dist){
    int id = get_global_id(0);
    dist[id] = 0.0;
}

__kernel void mem_set(__global float * dist,int id,float value){
    dist[id] = value;
}

__kernel void add_source(__global float * x,__global float * s, float dt){
    int id = get_global_id(0);
    x[id] += dt*s[id];
}

__kernel void advect(int N, int b,  __global float * d,  __global float * d0,  __global float *u, __global float * v,  __global float * w,float dt){
    int id = get_global_id(0);
    
    int k = (int)id/((N+2)*(N+2));
    int j = (int)(id%((N+2)*(N+2)))/(N+2);
    int i = (int)(id%((N+2)*(N+2)))%(N+2);
    
    int k0, k1;
    int i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;
    float z,u0,u1;
    
    dt0 = dt*N;
    
    if(i == 0 || i == N+1 || j == 0 || j == N+1 || k == 0 || k == N+1){
        return;
    }else{
        x = i-dt0*u[IX3(i,j,k)];
        y = j-dt0*v[IX3(i,j,k)];
        z = k-dt0*w[IX3(i,j,k)];
        
        if (x<0.5f) x=0.5f;
        if (x>N+0.5f) x=N+0.5f;
        i0=(int)x; i1=i0+1;
        if (y<0.5f) y=0.5f;
        if (y>N+0.5f) y=N+0.5f;
        j0=(int)y; j1=j0+1;
        
        if (z<0.5f) z=0.5f;
        if (z>N+0.5f) z=N+0.5f;
        k0=(int)z; k1=k0+1;
        
        s1 = x-i0;
        s0 = 1-s1;
        t1 = y-j0;
        t0 = 1-t1;
        
        u1 = z-k0;
        u0 = 1-u1;
        
        d[IX3(i,j,k)] =
        s0*t0*u0 *d0[IX3(i0,j0,k0)] +
        s0*t0*u1 *d0[IX3(i0,j0,k1)] +
        s0*t1*u0 *d0[IX3(i0,j1,k0)] +
        s0*t1*u1 *d0[IX3(i0,j1,k1)] +
        s1*t0*u0 *d0[IX3(i1,j0,k0)] +
        s1*t0*u1 *d0[IX3(i1,j0,k1)] +
        s1*t1*u0 *d0[IX3(i1,j1,k0)] +
        s1*t1*u1 *d0[IX3(i1,j1,k1)];
    }
    
    
}

__kernel void project_1(int N, __global float * u, __global float *  v,__global float * w, __global float *  p,__global float * div){
    int id = get_global_id(0);
    
    int k = (int)id/((N+2)*(N+2));
    int j = (int)(id%((N+2)*(N+2)))/(N+2);
    int i = (int)(id%((N+2)*(N+2)))%(N+2);
    
    if(i == 0 || i == N+1 || j == 0 || j == N+1 || k == 0 || k == N+1){
        return;
    }else{
        div[IX3(i,j,k)] = -0.33f*(u[IX3(i+1,j,k)]-u[IX3(i-1,j,k)]+v[IX3(i,j+1,k)]-v[IX3(i,j-1,k)]+w[IX3(i,j,k+1)]-w[IX3(i,j,k-1)])/N;
        p[IX3(i,j,k)] = 0;
    }
}

__kernel void project_2(int N, __global float * u, __global float *  v,__global float * w, __global float *  p){
    int id = get_global_id(0);
    
    int k = (int)id/((N+2)*(N+2));
    int j = (int)(id%((N+2)*(N+2)))/(N+2);
    int i = (int)(id%((N+2)*(N+2)))%(N+2);
    
    if(i == 0 || i == N+1 || j == 0 || j == N+1 || k == 0 || k == N+1){
        return;
    }else{
        u[IX3(i,j,k)] -= 0.33f*N*(p[IX3(i+1,j,k)]-p[IX3(i-1,j,k)]);
        v[IX3(i,j,k)] -= 0.33f*N*(p[IX3(i,j+1,k)]-p[IX3(i,j-1,k)]);
        w[IX3(i,j,k)] -= 0.33f*N*(p[IX3(i,j,k+1)]-p[IX3(i,j,k-1)]);
    }
}

__kernel void set_bnd_x2(int N,__global float * x,int b){
    int id = get_global_id(0);
    
    int j = (int)id/(N+2);
    int i = (int)id%(N+2);
    
    if(j == 0 || j == N+1 || i == 0 || i == N+1)return;
    
    x[IX3(0  ,i,j)] = b==1 ? -x[IX3(1,i,j)] : x[IX3(1,i,j)];
    x[IX3(N+1,i,j)] = b==1 ? -x[IX3(N,i,j)] : x[IX3(N,i,j)];
    
    x[IX3(i,0  ,j)] = b==2 ? -x[IX3(i,1,j)] : x[IX3(i,1,j)];
    x[IX3(i,N+1,j)] = b==2 ? -x[IX3(i,N,j)] : x[IX3(i,N,j)];
    
    x[IX3(j,i  ,0)] = b==3 ? -x[IX3(j,i  ,1)] : x[IX3(j,i  ,1)];
    x[IX3(j,i,N+1)] = b==3 ? -x[IX3(j,i,N)] : x[IX3(j,i,N)];
}

__kernel void set_bnd_x1(int N,__global float * x ){
    
    int j = get_global_id(0);
    
    if(j == 0 || j == N+1)return;
    
    x[IX3(0  ,0  ,j)] = 0.5f*(x[IX3(1,0  ,j)]+x[IX3(0  ,1,j)]);
    x[IX3(0  ,N+1,j)] = 0.5f*(x[IX3(1,N+1,j)]+x[IX3(0  ,N,j)]);
    x[IX3(N+1,0  ,j)] = 0.5f*(x[IX3(N,0  ,j)]+x[IX3(N+1,1,j)]);
    x[IX3(N+1,N+1,j)] = 0.5f*(x[IX3(N,N+1,j)]+x[IX3(N+1,N,j)]);
    
    x[IX3(j,0  ,0  )] = 0.5f*(x[IX3(j,1  ,0  )]+x[IX3(j,0  ,1  )]);
    x[IX3(j,N+1,0  )] = 0.5f*(x[IX3(j,N,0  )]+x[IX3(j,N+1,1)]);
    x[IX3(j,0  ,N+1)] = 0.5f*(x[IX3(j,1 ,N+1)]+x[IX3(j,0  ,N)]);
    x[IX3(j,N+1,N+1)] = 0.5f*(x[IX3(j,N,N+1)]+x[IX3(j,N+1,N)]);
    
    x[IX3(0  ,j  ,0  )] = 0.5f*(x[IX3(0  ,j  ,1  )]+x[IX3(1  ,j  ,0  )]);
    x[IX3(0  ,j  ,N+1)] = 0.5f*(x[IX3(1  ,j  ,N+1)]+x[IX3(0  ,j  ,N  )]);
    x[IX3(N+1,j  ,0  )] = 0.5f*(x[IX3(N  ,j  ,0  )]+x[IX3(N+1,j  ,1  )]);
    x[IX3(N+1,j  ,N+1)] = 0.5f*(x[IX3(N  ,j  ,N+1)]+x[IX3(N+1,j  ,N  )]);
    
}

__kernel void set_bnd_x0(int N,__global float * x ){
    x[IX3(0,0,0)] = 0.33f*(x[IX3(0,0,1)]+x[IX3(0,1,0)]+x[IX3(1,0,0)]);
    x[IX3(0,0,N+1)] = 0.33f*(x[IX3(1,0,N+1)]+x[IX3(0,1,N+1)]+x[IX3(0,0,N)]);
    x[IX3(0,N+1,0)] = 0.33f*(x[IX3(1,N+1,0)]+x[IX3(0,N+1,1)]+x[IX3(0,N,0)]);
    x[IX3(0,N+1,N+1)] = 0.33f*(x[IX3(1,N+1,N+1)]+x[IX3(0,N,N+1)]+x[IX3(0,N+1,N)]);
    x[IX3(N+1,0,0)] = 0.33f*(x[IX3(N+1,0,0)]+x[IX3(N+1,0,0)]+x[IX3(N+1,0,0)]);
    x[IX3(N+1,0,N+1)] = 0.33f*(x[IX3(N+1,1,N+1)]+x[IX3(N,0,N+1)]+x[IX3(N+1,0,N)]);
    x[IX3(N+1,N+1,0)] = 0.33f*(x[IX3(N+1,N+1,1)]+x[IX3(N,N+1,0)]+x[IX3(N+1,N,0)]);
    x[IX3(N+1,N+1,N+1)] = 0.33f*(x[IX3(N,N+1,N+1)]+x[IX3(N+1,N,N+1)]+x[IX3(N+1,N+1,N)]);
}
