#ifndef PROGRAM_OPTIONS_H
#define PROGRAM_OPTIONS_H

template <typename T>
class CUDAImageFilteringOptions {

  private:
    bool mVerboseFlag;		// The flag for verbosely listing and saving intermediate results
    unsigned int mDeviceId; 	// The device ID of a GPU
    unsigned int mNoChannels;	// The number of channels of an image to be processed; 3:RGB, 1:Y
    T mMaxLevel;		// The maximum level of level adjustment (postprocessing)
    bool mPostprocessingFlag;	// The flag for disabling postprocessing

  public:
    //
    // The default constructor and destructor
    //
    CUDAImageFilteringOptions();
    ~CUDAImageFilteringOptions();

    //
    // The inline functions for getters
    //
    bool getVerboseFlag() { return mVerboseFlag; }
    unsigned int getDeviceId() { return mDeviceId; }
    unsigned int getNoChannels() { return mNoChannels; }
    T getMaxLevel() { return mMaxLevel; }
    bool getPostprocessingFlag() { return mPostprocessingFlag; }

    //
    // Set the program options
    //
    int setProgramOptions( int *argc, char *argv[] );

    //
    // Show usage
    //
    void showUsage();
    
};

#endif // PROGRAM_OPTIONS_H
