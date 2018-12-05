////
//// program_options.cpp: The member functions for the CUDAImageFilteringOptions class
////

///
/// The standard include files
///
#include <iostream>
#include <sstream>

///
/// The include file for the class CUDAImageFilteringOptions
///
#include "cuda_image_filtering_options.h"

///
/// The default constructor and destructor
///
template <typename T>
CUDAImageFilteringOptions<T>::CUDAImageFilteringOptions()
{

    //
    // Set the default values of the options
    //

    // The flag for verbosely listing and saving intermediate results
    mVerboseFlag = false;

    // The device Id of a GPU
    mDeviceId = 0;

    // The number of channels of an image to be processed; 3:RGB, 1:Y
    mNoChannels = 3;

    // The maximum level for level adjustment (postprocessing)
    mMaxLevel = 180;

    // The flag for postprocessing
    mPostprocessingFlag = true;

}

template <typename T>
CUDAImageFilteringOptions<T>::~CUDAImageFilteringOptions()
{

}

///
/// Set the program options
///
template <typename T>
int CUDAImageFilteringOptions<T>::setProgramOptions( int *argc, char *argv[] )
{

    std::istringstream ist;
    std::string emptyString;

    unsigned int j = 1;
    for( unsigned int i = 1; i < *argc; i++ ) {
	if( argv[i][0] == '-' ) {
	    switch( argv[i][1] ) {
    
	      case 'v':
		mVerboseFlag = true;
		break;

	      case 'd':
		ist.str(&argv[i][2]);
		ist >> mDeviceId;
		std::cout << "The GPU with the device number "
			  << mDeviceId
			  << " is used."
			  << std::endl;
		ist.str(emptyString);
		ist.clear();
		break;

	      case 'c':
		ist.str(&argv[i][2]);
		ist >> mNoChannels;
		std::cout << "The number of channels to be processed: "
			  << mNoChannels
			  << "."
			  << std::endl;
		ist.str(emptyString);
		ist.clear();
		break;

	      case 'l':
		ist.str(&argv[i][2]);
		ist >> mMaxLevel;
		std::cout << "The maximum level for level adjustment: "
			  << mNoChannels
			  << "."
			  << std::endl;
		ist.str(emptyString);
		ist.clear();
		break;

	      case 'p':
		mPostprocessingFlag = false;
		break;

	      case 'h':
	      default:
		showUsage();
		exit(0);
		
	    }
	} else {
	    argv[j++] = argv[i];
	}
    }
    *argc = j;

    return 0;

}

///
/// Show usage
///
template <typename T>
void CUDAImageFilteringOptions<T>::showUsage()
{

    std::cerr << "-v: Verbosely listing and saving intermediate results." << std::endl;
    std::cerr << "-d: the device ID of a GPU: " << mDeviceId << "." << std::endl;
    std::cerr << "-c: the number of channels to be proecssed; 3:RGB, 1:Y : " << mNoChannels << "." <<std::endl;
    std::cerr << "-l: the maximum level for level adjustment (postprocessing): " << mMaxLevel << "." <<std::endl;
    std::cerr << "-p: Disabling postprocessing for filtering results." << std::endl;
    std::cerr << "-h: help." << std::endl;

}

////
//// Explicit instantiation for the template class
////
template class CUDAImageFilteringOptions<float>;
template class CUDAImageFilteringOptions<double>;
