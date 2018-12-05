////
//// cuda_image_filtering_main.cpp: An example program for image filtering using CUDA
////

///
/// The standard include files
///
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#include <cstdlib>
#include <cstring>

///
/// The include files for CUDA
///
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

///
/// The include files for image filtering
///
#include "cuda_image_filtering_options.h"
#include "path_handler.h"
#include "image_rw_cuda.h"
#include "padding.h"
#include "color_converter.h"
#include "image_filter.h"
#include "image_mse.h"
#include "postprocessing.h"
#include "get_micro_second.h"

///
/// The main function
///
int main( int argc, char *argv[] )
{

    //----------------------------------------------------------------------

    //
    // Set the device number as a program option
    //
    CUDAImageFilteringOptions<float> opt;
    opt.setProgramOptions( &argc, argv );

    //----------------------------------------------------------------------

    //
    // Variables for measuring execution times
    //
    enum execTimes {
	IMAGE_TRANSFER_TIME,
	Y_COMP_TIME,
	IMAGE_FILTER_TIME,
	POSTPROCESSING_TIME,
	TOTAL_TIME,
	NO_EXEC_TIME
    };

    std::string execTimeName[NO_EXEC_TIME] = {
	"Image transfer",
	"Y component",
	"Image filter",
	"Postprocessing",
	"Total time"
    };

    double sTime, eTime;
    static double execTimeGPU[NO_EXEC_TIME], execTimeCPU[NO_EXEC_TIME];

    //----------------------------------------------------------------------

    //
    // Check GPU(s)
    //
    int deviceCount;
    checkCudaErrors( cudaGetDeviceCount(&deviceCount) );
    if( deviceCount == 0 ) {
	std::cerr << "No CUDA enable device is found." << std::endl;
	exit(1);
    }

    //
    // Set a device number
    //
    int deviceId;
    if( opt.getDeviceId() == -1 )
	deviceId = gpuGetMaxGflopsDeviceId(); // The fastest device supporting CUDA is used.
    else
	deviceId = opt.getDeviceId();

    cudaDeviceProp deviceProp;
    checkCudaErrors( cudaGetDeviceProperties( &deviceProp, deviceId ) );
    std::cout << deviceProp.name << " is used." << std::endl;
    checkCudaErrors( cudaSetDevice( deviceId ) );

    //----------------------------------------------------------------------

    //
    // Input file paths
    //
    std::string inputImageFilePath;
    std::string filterDataFilePath;
    std::string outputImageFilePath;
    if( argc <= 1 ) {
	std::cerr << "Input image file path: ";
	std::cin >> inputImageFilePath;
	std::cerr << "Filter data file path: ";
	std::cin >> filterDataFilePath;
	std::cerr << "Output image file path: ";
	std::cin >> outputImageFilePath;
    } else if( argc <= 2 ) {
	inputImageFilePath = argv[1];
	std::cerr << "Filter data file path: ";
	std::cin >> filterDataFilePath;
	std::cerr << "Output image file path: ";
	std::cin >> outputImageFilePath;
    } else if( argc <= 3 ) {
	inputImageFilePath = argv[1];
	filterDataFilePath = argv[2];
	std::cerr << "Output image file path: ";
	std::cin >> outputImageFilePath;
    } else {
	inputImageFilePath = argv[1];
	filterDataFilePath = argv[2];
	outputImageFilePath = argv[3];
    }

    //----------------------------------------------------------------------

    //
    // Set the prefix and extension of the input image file
    //
    std::string imageFileDir;
    std::string imageFileName;
    getDirFileName( inputImageFilePath, &imageFileDir, &imageFileName );

    std::string imageFilePrefix;
    std::string imageFileExt;
    getPrefixExtension( imageFileName, &imageFilePrefix, &imageFileExt );

    //----------------------------------------------------------------------

    //
    // Read the intput image in pageable memory on a host
    // Page-locked memory (write-combining memory) is not used, because padding is performed on a host 
    //
    hsaImage<float> h_inputImage;
    if( imageFileExt == "tif" ) { // TIFF
      h_inputImage.tiffGetImageSize( inputImageFilePath );
      h_inputImage.allocImage( PAGEABLE_MEMORY );
      h_inputImage.tiffReadImage( inputImageFilePath );
    } else if( imageFileExt == "jpg" ) { // JPEG
      h_inputImage.jpegGetImageSize( inputImageFilePath );
      h_inputImage.allocImage( PAGEABLE_MEMORY );
      h_inputImage.jpegReadImage( inputImageFilePath );
    } else if( imageFileExt == "png" ) { // PNG
      h_inputImage.pngGetImageSize( inputImageFilePath );
      h_inputImage.allocImage( PAGEABLE_MEMORY );
      h_inputImage.pngReadImage( inputImageFilePath );
    }

    //
    // Show the size of the input image
    //
    std::cout << "The size of the input image: ("
	      << h_inputImage.getImageWidth()
	      << ", "
	      << h_inputImage.getImageHeight() 
	      << ")"
	      << std::endl;

    //----------------------------------------------------------------------

    //
    // Read the filter data file
    //
    std::ifstream fin;
    fin.open( filterDataFilePath.c_str() );
    if( !fin ) {
	std::cerr << "Could not open the filter data file: "
		  << filterDataFilePath
		  << std::endl;
	exit(1);
    }

    // Read the size of the filter
    unsigned int filterSize;
    fin >> filterSize;

    // Read the filter kernel
    float *filterKernel;
    try {
	filterKernel = new float[ filterSize * filterSize ];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for filterKernel: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    for( unsigned int i = 0; i < filterSize; i++ )
	for( unsigned int j = 0; j < filterSize; j++ )
	    fin >> filterKernel[ i * filterSize + j ];
    
    // Show the filter kernel
    if( opt.getVerboseFlag() ) {
	std::cout << "*** Filter kernel ***" << std::endl;
	std::cout << "Filter size: " << filterSize << std::endl;
	for( unsigned int i = 0; i < filterSize; i++ ) {
	    for( unsigned int j = 0; j < filterSize; j++ )
		std::cout << filterKernel[ i * filterSize + j ] << " ";
	    std::cout << std::endl;
	}
    }
    
    fin.close();

    //----------------------------------------------------------------------

    //
    // Perform image filtering on a GPU
    //

    //
    // Pad the input image for filtering
    //
    const unsigned int hFilterSize = filterSize / 2;
    const unsigned int paddedImageWidth = h_inputImage.getImageWidth() + 2 * hFilterSize;
    const unsigned int paddedImageHeight = h_inputImage.getImageHeight() + 2 * hFilterSize;

    // Note that using page-locked memory (write-combining memory) improves transfer
    // performance between a CPU and a GPU.
    hsaImage<float> h_padded_inputImageGPU;
    h_padded_inputImageGPU.allocImage( paddedImageWidth, paddedImageHeight, PAGE_LOCKED_MEMORY );
    imagePadding( h_inputImage, filterSize, REPLICATION_PADDING, &h_padded_inputImageGPU );

    if( opt.getVerboseFlag() ) {
	std::cout << "Saving the padded input image." << std::endl; 
	h_padded_inputImageGPU.pngSaveImage( "h_padded_inputImageGPU.png", RGB_DATA );
    }
    
    //
    // Allocate memory space for the input image on a GPU
    //
    dsaImage<float> d_iImage;
    d_iImage.allocImage( h_padded_inputImageGPU.getImageWidth(), h_padded_inputImageGPU.getImageHeight() );

    //
    // Transfer the input image from a CPU to a GPU
    //
    sTime = getMicroSecond();
    d_iImage.transferImage( h_padded_inputImageGPU );
    eTime = getMicroSecond();
    execTimeGPU[IMAGE_TRANSFER_TIME] = ( eTime - sTime );

    //
    // Compute Y Component by a GPU
    //
    colorConverter<float> cConverter;

    sTime = getMicroSecond();
    cConverter.calYComponentGPU( &d_iImage );
    eTime = getMicroSecond();
    execTimeGPU[Y_COMP_TIME] = ( eTime - sTime );

    //
    // Apply an image filter on a GPU
    //
    imageFilter<float> imageFilter( filterSize, filterKernel );
    dsaImage<float> d_oImage;
    d_oImage.allocImage( h_inputImage.getImageWidth(), h_inputImage.getImageHeight() );

    sTime = getMicroSecond();
    imageFilter.applyFilterGPU( d_iImage, opt.getNoChannels(), &d_oImage );
    eTime = getMicroSecond();
    execTimeGPU[IMAGE_FILTER_TIME] = ( eTime - sTime );

    //
    // Back-transfer the output image from a GPU to a CPU
    // Note that these images are allocated using page-locked memory for transfer.
    //
    hsaImage<float> h_backTransOImage;
    h_backTransOImage.allocImage( h_inputImage.getImageWidth(), h_inputImage.getImageHeight(), PAGE_LOCKED_MEMORY );
    if( opt.getNoChannels() == 3 ) {
	sTime = getMicroSecond();
	d_oImage.backTransferImage( &h_backTransOImage, RGB_DATA );
	eTime = getMicroSecond();
	execTimeGPU[IMAGE_TRANSFER_TIME] += ( eTime - sTime );
    } else {
	sTime = getMicroSecond();
	d_oImage.backTransferImage( &h_backTransOImage, Y_COMPONENT );
	eTime = getMicroSecond();
	execTimeGPU[IMAGE_TRANSFER_TIME] += ( eTime - sTime );
    }

    //----------------------------------------------------------------------

    //
    // Perform image filtering on a CPU
    //

    //
    // Pad the input image for filtering
    // Note that pageable memory is used for a CPU.
    //
    hsaImage<float> h_padded_inputImageCPU;
    h_padded_inputImageCPU.allocImage( paddedImageWidth, paddedImageHeight, PAGEABLE_MEMORY );
    imagePadding( h_inputImage, filterSize, REPLICATION_PADDING, &h_padded_inputImageCPU );

    //
    // Compute Y Component by a CPU
    //
    sTime = getMicroSecond();
    cConverter.calYComponentCPU( &h_padded_inputImageCPU );
    eTime = getMicroSecond();
    execTimeCPU[Y_COMP_TIME] = ( eTime - sTime );

    //
    // Apply the filter on a CPU
    //
    hsaImage<float> h_oImage;
    h_oImage.allocImage( h_inputImage.getImageWidth(), h_inputImage.getImageHeight(), PAGEABLE_MEMORY );

    sTime = getMicroSecond();
    imageFilter.applyFilterCPU( h_padded_inputImageCPU, opt.getNoChannels(), &h_oImage );
    eTime = getMicroSecond();
    execTimeCPU[IMAGE_FILTER_TIME] = ( eTime - sTime );

    if( opt.getVerboseFlag() ) {
	std::cout << "Saveing the filtering result by a CPU." << std::endl;
	if( opt.getNoChannels() == 3 )
	    h_oImage.pngSaveImage( "h_oImage.png", RGB_DATA );
	else
	    h_oImage.pngSaveImage( "h_oImage.png", Y_COMPONENT );
    }
    
    //----------------------------------------------------------------------

    //
    // Check the difference between the filtering results obtained by a GPU and a CPU
    // The result of a GPU may not be identical to that of a CPU due to truncation.
    // In particular, the --fmad option of nvcc affects the accuracy.
    // See the CUDA C Programming Guide.
    //
    std::cout << "The MSE between the image filtering results obtained by the GPU and CPU: " 
	      << calMSE( h_backTransOImage, h_oImage, opt.getNoChannels() ) 
	      << "."
	      << std::endl;
    
    //----------------------------------------------------------------------

    //
    // Postprocessing for the filtering result
    // Take the absolute value of each pixel value, normalize its result to
    // [0,255] and adjust the level of the image for the sake of clarity.
    //
    std::cout << "Postprocessing is performed." << std::endl; 
    if( opt.getPostprocessingFlag() ) {

	// CPU
	// Do not use page-locked memory, because using it makes processing much slower
	sTime = getMicroSecond();
	takeImageAbsoluteValueCPU( &h_oImage, opt.getNoChannels() );
	normalizeImageCPU( &h_oImage, opt.getNoChannels() );
	adjustImageLevelCPU( &h_oImage, opt.getNoChannels(), opt.getMaxLevel() );
	eTime = getMicroSecond();
	execTimeCPU[POSTPROCESSING_TIME] = ( eTime - sTime );

	// GPU
	sTime = getMicroSecond();
	takeImageAbsoluteValueGPU( &d_oImage, opt.getNoChannels() );
	normalizeImageGPU( &d_oImage, opt.getNoChannels() );
	adjustImageLevelGPU( &d_oImage, opt.getNoChannels(), opt.getMaxLevel() );
	eTime = getMicroSecond();
	execTimeGPU[POSTPROCESSING_TIME] = ( eTime - sTime );

	// Check the difference between the postprocessing results obtained by a GPU and a CPU
	// Note that the time for back-transfer does not take into account here, because
	// the time is measured for the image filtering result.
	if( opt.getNoChannels() == 3 )
	    d_oImage.backTransferImage( &h_backTransOImage, RGB_DATA );
	else
	    d_oImage.backTransferImage( &h_backTransOImage, Y_COMPONENT );
	
	std::cout << "The MSE between the postprocessing results obtained by the GPU and CPU: " 
		  << calMSE( h_backTransOImage, h_oImage, opt.getNoChannels() ) 
		  << "."
		  << std::endl;

    }

    //----------------------------------------------------------------------

    //
    // Calculate total times
    //
    execTimeGPU[TOTAL_TIME] = 0.0;
    for( int i = 0; i < NO_EXEC_TIME-1; i++ )
	execTimeGPU[TOTAL_TIME] += execTimeGPU[i];

    execTimeCPU[TOTAL_TIME] = 0.0;
    for( int i = 0; i < NO_EXEC_TIME-1; i++ )
	execTimeCPU[TOTAL_TIME] += execTimeCPU[i];

    //
    // Show computational times
    // The unit is micro second. 
    //
    std::cout << "*** Computational times of the GPU ***" << std::endl;
    for( int i = 0; i < NO_EXEC_TIME; i++ )
	std::cout << execTimeName[i] << ": " << execTimeGPU[i] * 1e3 << "[ms]" << std::endl;

    std::cout << "*** Computational times of the CPU ***" << std::endl;
    for( int i = 0; i < NO_EXEC_TIME; i++ )
	std::cout << execTimeName[i] << ": " << execTimeCPU[i] * 1e3 << "[ms]" << std::endl;

    //----------------------------------------------------------------------

    //
    // Show the speed-up by GPU for computing Y component
    //
    std::cout << "Computing Y component: the GPU is " 
	      << execTimeCPU[Y_COMP_TIME] / execTimeGPU[Y_COMP_TIME] 
	      << "x faster than the CPU." 
	      << std::endl;

    //
    // Show the speed-up by GPU for the image filter
    //
    std::cout << "Performing image fitlering: the GPU is " 
	      << execTimeCPU[IMAGE_FILTER_TIME] / execTimeGPU[IMAGE_FILTER_TIME] 
	      << "x faster than the CPU." 
	      << std::endl;

    //
    // Show the speed-up by GPU for the postprocessing
    //
    std::cout << "Performing postprocessing: the GPU is " 
	      << execTimeCPU[POSTPROCESSING_TIME] / execTimeGPU[POSTPROCESSING_TIME] 
	      << "x faster than the CPU." 
	      << std::endl;

    //
    // Show the overall speed-up by GPU
    //
    std::cout << "The overall speed-up by the GPU is " 
	      << execTimeCPU[TOTAL_TIME] / execTimeGPU[TOTAL_TIME] 
	      << "x." 
	      << std::endl;

    //----------------------------------------------------------------------

    //
    // Save the processing result by a GPU
    //
    std::cout << "Saving the filtering result by the GPU to " << outputImageFilePath << std::endl;
    getDirFileName( outputImageFilePath, &imageFileDir, &imageFileName );
    getPrefixExtension( imageFileName, &imageFilePrefix, &imageFileExt );
    if( imageFileExt == "tif" ) { // TIFF
	if( opt.getNoChannels() == 3 )
	    h_backTransOImage.tiffSaveImage( outputImageFilePath, RGB_DATA );
	else
	    h_backTransOImage.tiffSaveImage( outputImageFilePath, Y_COMPONENT );
    } else if( imageFileExt == "jpg" ) { // JPEG
	if( opt.getNoChannels() == 3 )
	    h_backTransOImage.jpegSaveImage( outputImageFilePath, RGB_DATA );
	else
	    h_backTransOImage.jpegSaveImage( outputImageFilePath, Y_COMPONENT );
    } else if( imageFileExt == "png" ) { // PNG
	if( opt.getNoChannels() == 3 )
	    h_backTransOImage.pngSaveImage( outputImageFilePath, RGB_DATA );
	else
	    h_backTransOImage.pngSaveImage( outputImageFilePath, Y_COMPONENT );
    } else { // without an extension, PNG
	if( opt.getNoChannels() == 3 )
	    h_backTransOImage.pngSaveImage( outputImageFilePath, RGB_DATA );
	else
	    h_backTransOImage.pngSaveImage( outputImageFilePath, Y_COMPONENT );
    }

    //----------------------------------------------------------------------

    //
    // Delete the memory spaces
    //
    h_oImage.freeImage();
    h_padded_inputImageCPU.freeImage();

    h_backTransOImage.freeImage();
    d_oImage.freeImage();
    d_iImage.freeImage();

    h_padded_inputImageGPU.freeImage();

    delete [] filterKernel;
    filterKernel = 0;

    h_inputImage.freeImage();

    //----------------------------------------------------------------------
    
    return 0;

}
