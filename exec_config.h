#ifndef EXEC_CONFIG_H
#define EXEC_CONFIG_H

//
// Execution configuration
// Check the specification of your GPU
//
const int CONSTANT_MEMORY_SIZE = 64 * 1024;	// 64[KB]
const int SHARED_MEMORY_SIZE = 48 * 1024;	// 48[KB]
const int MAX_NO_THREAD = 1024; 		// the maximum number of threads per block

//
// Inline function for the execution configuration
//
inline unsigned int iDivUp( const unsigned int &a , const unsigned int &b ) { return ( a%b != 0 ) ? (a/b+1):(a/b); }

#endif // EXEC_CONFIG_H
