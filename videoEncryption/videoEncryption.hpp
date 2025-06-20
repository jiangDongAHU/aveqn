#ifndef __VIDEO_ENCRYPTION_HPP__
#define __VIDEO_ENCRYPTION_HPP__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <semaphore.h>
#include <openssl/sha.h>
#include "opencv2/opencv.hpp"

using namespace cv;

#define ORIGINAL_VIDEO_FILE         "../originalData/foreman_cif.yuv.mp4"
#define QUANTUM_BIT_FILE_POINTER    "../quantumBits/quantumBits.txt"

#define FRAME_WIDTH                 512
#define FRAME_HEIGHT                512
#define VIDEO_FPS                   24

//macros for audio encryption and decryption
#define CONFUSION_ROUNDS            3
#define DIFFUSION_CONFUSION_ROUNDS  3
#define NUMBER_OF_THREADS           32
#define FRAMES_TO_PROCESS           10

#define BYTES_RESERVED              6
#define PRE_ITERATIONS              200
#define CONFUSION_SEED_LOWWER_BOUND 3000
#define CONFUSION_SEED_UPPER_BOUND  30000
#define PI                          acos(-1)

//This structure is utilized to convey relevant information during the creation of assistant threads
struct assistantThreadParameter{
    int threadIdx;
    int iterations;
    double * initParameterArray;
};

//assistant thread execute this function to encrypt and decrypt the video frames
static void * assistantThreadFunc(void * arg);

double GetCPUSecond(void);

//calculate SHA256 hash of the original frame
void calculateSHA256Hash(unsigned char * frameData, size_t dataLength, unsigned char * SHA256HashResultArray);

//generate initial condition or control parameter using the SHA256 hash of the original frame and the quantum bits
double constructParameters(unsigned char * SHA256HashResultArray, FILE * fp);

//iterate the PLCM and return the iteration result
double PLCM(double initialCondition, double controlParameter);

//generate parameters for initializing PLCMs of assistant threads
void GenerateParameters(double controlParameter1, double * initialCondition1, 
                        double controlParameter2, double * initialCondition2,
                        double * initParameterArray);

//fetch 16 quantum bits to construct a confusion seed
int GenerateConfusionSeedFromQuantumBits(FILE * fp);

//iterate the PLCM and store results
double IteratePLCM(double controlParameter, double initialCondition, int iterations, double * iterationResultArray);

//convert iteration results to byte sequence
void ConvertResultToByte(double * resultArray, unsigned char * byteSequence, int elems);

//Act XOR operation and store the bytes for encryption
void GenerateBytes(int iterations, unsigned char * uCharResultArray1, unsigned char * uCharResultArray2, unsigned char * byteSequence);

//use the chaotic map to generate diffusion seeds for encryption a frame
void GenerateDiffusionSeeds(double controlParameter1, double * initialCondition1, 
                            double controlParameter2, double * initialCondition2,
                            unsigned char * diffusionSeedArray);

//confusion function
void Confusion(int startingRow, int endingRow);

//diffusion function
int Diffusion(int startingRow, int endingRow, unsigned char * diffusionSeed, unsigned char * byteSequence, int idx);

//inverse confusion function
void InverseConfusion(int startingRow, int endingRow);

//inverse diffusion function
int InverseDiffusion(int startingRow, int endingRow, unsigned char * diffusionSeed, unsigned char * byteSequence, int idx);
#endif