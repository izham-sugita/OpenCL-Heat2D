#define __CL_ENABLE_EXCEPTIONS
 
#include "cl.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>
 
//#include "heat2Dkernel.cl"
//#include "testkernel.cl"

#define pi 4.0*atan(1.0)

using namespace std;
using namespace std::chrono;

int main(int argc, char *argv[])
{
 
    // Length of vectors
    unsigned int nx = 128;
    unsigned int ny = 128;
    unsigned int n  = nx*ny;

    double dx = 1.0/(double)(nx-1);
    double dy = 1.0/(double)(ny-1);
    double dt = 0.01*(dx*dx);

    double kappa =  1.0*(dt/(dx*dx));
 
    // Host input vector
    double *h_T;
    // Host output vector
    double *h_Tnew;
    
    // Device input buffer
    cl::Buffer d_T;

    // Device output buffer
    cl::Buffer d_Tnew;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
 
    // Allocate memory for each vector on host
    h_T = new double[n];
    h_Tnew = new double[n];
 
    // Initialize vectors on host
    for(int i = 0; i < nx; i++ ){
      int id = i*ny;
      for(int j = 0; j <ny; j++){
	int index = id + j;
	h_T[index] = sin((double)i*dx*pi) * sin((double)j*dy*pi);
        h_Tnew[index] = 0.0;
      }
    }
 
    cl_int err = CL_SUCCESS;
    try {
 
        // Query platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            std::cout << "Platform size 0\n";
            return -1;
         }
 
        // Get list of devices on default platform and create context
        cl_context_properties properties[] =
           { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
 
        // Create command queue for first device
        cl::CommandQueue queue(context, devices[0], 0, &err);

	std::cout<<devices[0].getInfo<CL_DEVICE_NAME>()<<std::endl;
	
        // Create device memory buffers
        d_T = cl::Buffer(context, CL_MEM_READ_ONLY, bytes);
        d_Tnew = cl::Buffer(context, CL_MEM_READ_ONLY, bytes);
 
        // Bind memory buffers
        queue.enqueueWriteBuffer(d_T, CL_TRUE, 0, bytes, h_T);
        queue.enqueueWriteBuffer(d_Tnew, CL_TRUE, 0, bytes, h_Tnew);

	//Read program source
	std::ifstream sourceFileName("kernelHeat2D.cl");
	std::string sourceFile(
			       std::istreambuf_iterator<char>(sourceFileName),
			       (std::istreambuf_iterator<char>()) );
	cl::Program::Sources source(
				    1,
				    std::make_pair(sourceFile.c_str(),
						   sourceFile.length()+1));
	//create program
	cl::Program program_ =cl::Program(context,source);
	program_.build(devices);

	//create kernel object
	cl::Kernel kernel(program_,"heat2d", &err);
	
	/*	
        //Build kernel from source string
        cl::Program::Sources source(1,
            std::make_pair(kernelSource,strlen(kernelSource)));
        cl::Program program_ = cl::Program(context, source);
        program_.build(devices);
 
        // Create kernel object
        cl::Kernel kernel(program_, "heat2d", &err);
 
	*/

        // Bind kernel arguments to kernel
        kernel.setArg(0, d_T);
        kernel.setArg(1, d_Tnew);
	kernel.setArg(2, nx);
        kernel.setArg(3, ny);
	kernel.setArg(4, kappa);

	//std::cout<<"Argument clear!\n";
 
        // Number of work items in each local work group
	//cl::NDRange localSize(8);
        // Number of total work items - localSize must be devisor
        // cl::NDRange globalSize((int)(ceil(n/(float)8)*8));

	cl::NDRange localSize(32,32); //localSize is critical
	cl::NDRange globalSize(nx,ny);

	 int iter = 0;
	int itermax = 20000;
	
        // Enqueue kernel
        cl::Event event;

	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	
	do{
	queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            globalSize,
            localSize,
            NULL,
            &event);
 
        // Block until kernel completion
        event.wait();

	//swap arguments
	kernel.setArg(0, d_Tnew); // change from d_a to d_c; reflecting update
        	
	//swap
	//std::swap(d_T,d_Tnew);
	//kernel.setArg(0, d_T);
	//kernel.setArg(1, d_Tnew);
	
	iter +=1;
	}while(iter<itermax+1);

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> elapsed_time =
	  duration_cast<duration<double> >(t2-t1);
	cout<<"Time for "<<itermax<<" steps: "
	    <<elapsed_time.count()<<" secs."
	    <<endl;
	
        // Read back d_a
        queue.enqueueReadBuffer(d_Tnew, CL_TRUE, 0, bytes, h_Tnew);
        }
    catch (cl::Error err) {
         std::cerr
            << "ERROR: "<<err.what()<<"("<<err.err()<<")"<<std::endl;
    }
 
    // result output
    ofstream dataFile;
    dataFile.open ("opencl-heat2D.csv");
    dataFile << "x, y, z, T\n";
    for(int i=0; i<nx; ++i){
      for(int j=0; j<ny; ++j){
	int id = i*ny + j;
	double xg = (double)i*dx;
	double yg = (double)j*dy;
	dataFile <<xg<<","
		 <<yg<<","
		 <<h_Tnew[id]<<","
		 <<h_Tnew[id]<<endl;
      }
    }

    dataFile.close();
  

 
    // Release host memory
    delete(h_T);
    delete(h_Tnew);


 
    return 0;
}
