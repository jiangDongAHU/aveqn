#ifndef __AUDIO_ENCRYPTION_H__
#define __AUDIO_ENCRYPTION_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <alsa/asoundlib.h>
#include <openssl/sha.h>

#define QUANTUM_BIT_FILE_PATH       "../quantumBits/quantumBits.txt"
#define ORIGINAL_AUDIO_FILE         "../originalData/yvws.wav"

//macros for the initialization of capture and playbcak devices
#define PCM_DEVICE                  "default"
#define CAPTURE_SAMPLE_RATE         44100
#define PLAYBACK_SAMPLE_RATE        CAPTURE_SAMPLE_RATE
#define CHANNELS                    1
#define SAMPLE_WIDTH                2
#define AUDIO_MATRICES              10

//memory allocated for stroing the captured audio samples
#define AUDIO_MATRIX_ROWS_COLUMNS   64
#define AUDIO_BUFFER_SIZE           AUDIO_MATRIX_ROWS_COLUMNS * AUDIO_MATRIX_ROWS_COLUMNS

//macros for audio encryption and decryption
#define CONFUSION_ROUNDS            3
#define CONFUSION_DIFFUSION_ROUNDS  3
#define INT16_RESERVED              3
#define PRE_ITERATIONS              200
#define PI                          acos(-1)

//initialzie the playback device
snd_pcm_t * InitializePlaybackDevice();

double GetCPUSecond(void);

//calculate SHA256 hash of the original audio samples
void calculateSHA256Hash(int16_t * audioMatrixData, size_t dataLength, unsigned char * SHA256HashResultArray);

//generate initial condition or control parameter using the SHA256 hash of the original frame ann the quantum bits
double constructParameters(unsigned char * SHA256HashResultArray, FILE * fp);

//iterate the PLCM and return the iteration result
double PLCM(double initialCondition, double controlParameter);

//iterate the PLCM and store results
double IteratePLCM(double controlParameter, double initialCondition, int iterations, double * iterationResultArray);

//convert an iteration result into int16_t data
void ConvertIterationResultToInt16t(double * iterationResultArray, int16_t * int16tArray, int iterations);

//convert the linear buffer into a two-dimensional matrix
void convertLinearBufferToMatrix(int16_t * buffer, int16_t matrix[AUDIO_MATRIX_ROWS_COLUMNS][AUDIO_MATRIX_ROWS_COLUMNS]);

//fetch a set of quantum bits to generate a uint16_t data for confusion or diffusion operations
uint16_t GenerateUint16tDataFromQuantumBits(FILE * fp);

//perform confusion operation on the matrix
void Confusion(int16_t matrix[AUDIO_MATRIX_ROWS_COLUMNS][AUDIO_MATRIX_ROWS_COLUMNS], uint16_t confusionSeed);

//fetch a set of quantum bits to generate a uint16_t data for diffusion operations
int16_t GenerateInt16tDataFromQuantumBits(FILE * fp);

//perform diffusion operations on the matrix
int Diffusion(int16_t matrix[AUDIO_MATRIX_ROWS_COLUMNS][AUDIO_MATRIX_ROWS_COLUMNS], int16_t * finalKeyArray, int16_t diffusionSeed, int idx);

//perform inverse confusion operation on the matrix
void InverseConfusion(int16_t matrix[AUDIO_MATRIX_ROWS_COLUMNS][AUDIO_MATRIX_ROWS_COLUMNS], uint16_t confusionSeed);

//perform inverse diffusion operation on the matrix
int inverseDiffusion(int16_t matrix[AUDIO_MATRIX_ROWS_COLUMNS][AUDIO_MATRIX_ROWS_COLUMNS], int16_t * finalKeyArray, int16_t diffusionSeed, int idx);

//convert the two-dimensional matrix into a linear buffer
void convertMatrixToLinearBuffer(int16_t * buffer, int16_t matrix[AUDIO_MATRIX_ROWS_COLUMNS][AUDIO_MATRIX_ROWS_COLUMNS]);

#endif