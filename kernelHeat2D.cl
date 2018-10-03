// OpenCL kernel. Each work item takes care of one element of c
// const char *kernelSource =                                    
#pragma OPENCL EXTENSION cl_khr_fp64 : enable                  
__kernel void heat2d(  __global double *T,                     
                       __global double *Tnew,                  
                       const unsigned int nx,                  
                       const unsigned int ny,                  
                       const double kappa)                     
{                                                              
    //Get our global thread ID                                 
    int i = get_global_id(0);                                  
    int j = get_global_id(1);                                  
                                                               
    //Make sure we do not go out of bounds                     
    if (i > 0 && i<nx-1 ){                               
    if(j>0 && j<ny-1){                                  
    int id = i*ny + j;   //row-major                            
    //int id = i + j*nx; //column-major                         
        Tnew[id] = T[id] +kappa*((T[id-1]-2.0*T[id]+T[id+1])    
                              +(T[id-nx]-2.0*T[id]+T[id+nx]));  
        } 
        } 
};                              