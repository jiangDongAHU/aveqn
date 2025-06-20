#include "audioEncryption.h"

double quantumBitsConsumed;

int main(int argc, const char ** argv){
    //open the file that stores the generated quantum bits
    FILE * quantumBitFilePointer    = fopen(QUANTUM_BIT_FILE_PATH, "r");
    if(quantumBitFilePointer == NULL){
        printf("failed to open the quantum bit file\n");
        exit(1);
    }

    //open the original audio file
    FILE * originalAudioFilePointer = fopen(ORIGINAL_AUDIO_FILE, "rb");
    if(originalAudioFilePointer == NULL){
        printf("failed to open the original audio file\n");
        exit(1);
    }

    //SHA256 hashes of the original audio samples are stored in this array
    unsigned char SHA256HashResultArray[SHA256_DIGEST_LENGTH];

    //allocate memory for storing captured audio samples, PRBG iteration results, and int16_t data used in encryption and decryption
    int iterations                  = int(AUDIO_BUFFER_SIZE * CONFUSION_DIFFUSION_ROUNDS / INT16_RESERVED) + 1;
    int16_t * audioBuffer           = (int16_t *)malloc(AUDIO_BUFFER_SIZE * sizeof(int16_t));
    double  * iterationResultArray1 = (double *)malloc(iterations * sizeof(double));
    double  * iterationResultArray2 = (double *)malloc(iterations * sizeof(double));
    int16_t * int16tArray1          = (int16_t *)malloc(iterations * INT16_RESERVED * sizeof(int16_t));
    int16_t * int16tArray2          = (int16_t *)malloc(iterations * INT16_RESERVED * sizeof(int16_t));
    int16_t * finalKeyArray         = (int16_t *)malloc(iterations * INT16_RESERVED * sizeof(int16_t));

    double   initialCondition1, controlParameter1;
    double   initialCondition2, controlParameter2;
    uint16_t confusionSeed;
    int16_t  diffusionSeed[CONFUSION_DIFFUSION_ROUNDS];
    double   totalTime           = 0;
    int      audioMatrixIdx      = 0;
             quantumBitsConsumed = 0;

    //read the head of the original wav file
    fread(audioBuffer, SAMPLE_WIDTH, 44, originalAudioFilePointer);

    //initialize the playback device
    snd_pcm_t * playbackHandler = InitializePlaybackDevice();

    while(1){
        double startTime = GetCPUSecond();

        //read audio samples from the original audio file
        size_t bytesRead = fread(audioBuffer, SAMPLE_WIDTH, AUDIO_BUFFER_SIZE, originalAudioFilePointer);
        if(bytesRead < AUDIO_BUFFER_SIZE)
            break;
        
        //calculate the SHA-256 hash of the original samples
        calculateSHA256Hash(audioBuffer, AUDIO_BUFFER_SIZE * sizeof(int16_t), SHA256HashResultArray);

        //construct initial conditions and control parameters using the SHA256 hash of the original frame and the quantum bits
        calculateSHA256Hash(audioBuffer, AUDIO_BUFFER_SIZE * sizeof(int16_t), SHA256HashResultArray);

        initialCondition1 = constructParameters(&SHA256HashResultArray[0], quantumBitFilePointer);
        controlParameter1 = constructParameters(&SHA256HashResultArray[8], quantumBitFilePointer);
        if(controlParameter1 > 0.5) 
            controlParameter1 = controlParameter1 - 0.5;

        initialCondition2 = constructParameters(&SHA256HashResultArray[16], quantumBitFilePointer);
        controlParameter2 = constructParameters(&SHA256HashResultArray[24], quantumBitFilePointer);
        if(controlParameter2 > 0.5)
            controlParameter2 = controlParameter2 - 0.5;
            
        //pre-iterate PRBG
        for(int i = 0; i < PRE_ITERATIONS; i++){
            initialCondition1 = PLCM(initialCondition1, controlParameter1);
            initialCondition2 = PLCM(initialCondition2, controlParameter2);
        }

        if(audioMatrixIdx % AUDIO_MATRICES == 0){      
            system("clear");
            printf("\033[1maudio encrytpion and decryption using quantum bits\033[0m\n");
            printf("rounds of confusion operations: %d\n", CONFUSION_ROUNDS + CONFUSION_DIFFUSION_ROUNDS);
            printf("rounds of diffusion operations: %d\n", CONFUSION_DIFFUSION_ROUNDS);
            printf("initial condition 1: %f control parameter 1: %f\n", initialCondition1, controlParameter1);
            printf("initial condition 2: %f control parameter 2: %f\n", initialCondition2, controlParameter2);
            printf("quantum bits consumed: %f(kb)\n", quantumBitsConsumed / 1024);
            printf("audio matrices (%d x %d samples) that have been processed: %d\n", AUDIO_MATRIX_ROWS_COLUMNS, AUDIO_MATRIX_ROWS_COLUMNS, audioMatrixIdx);
            printf("playing back the decrypted audio samples ...\n");
        }

        //iterate PLCMs and generate int16_t data for encryption and decryption
        initialCondition1 = IteratePLCM(controlParameter1, initialCondition1, iterations, iterationResultArray1);
        initialCondition2 = IteratePLCM(controlParameter2, initialCondition2, iterations, iterationResultArray2);
        ConvertIterationResultToInt16t(iterationResultArray1, int16tArray1, iterations);
        ConvertIterationResultToInt16t(iterationResultArray2, int16tArray2, iterations);
        for(int i = 0; i < iterations * INT16_RESERVED; i++)
            finalKeyArray[i] = int16tArray1[i] ^ int16tArray2[i];
        
        //convert the linear buffer into a two-dimensional matrix
        int16_t audioMatrix[AUDIO_MATRIX_ROWS_COLUMNS][AUDIO_MATRIX_ROWS_COLUMNS];
        convertLinearBufferToMatrix(audioBuffer, audioMatrix);

        //encrypt the audio matrix
        //generate a confusion seed for confusion operations
        confusionSeed = GenerateUint16tDataFromQuantumBits(quantumBitFilePointer);
        //perform confusion operations on the audio matrix
        for(int i = 0; i < CONFUSION_ROUNDS; i++)
            Confusion(audioMatrix, confusionSeed);

        int finalKeyArrayIdx = 0;
        //perform diffusion and confusion operations on the audio matrix
        for(int i = 0; i < CONFUSION_DIFFUSION_ROUNDS; i++){
            //generate a diffusion seed for diffusion operations
            diffusionSeed[i] = GenerateInt16tDataFromQuantumBits(quantumBitFilePointer);

            finalKeyArrayIdx = Diffusion(audioMatrix, finalKeyArray, diffusionSeed[i], finalKeyArrayIdx);
            Confusion(audioMatrix, confusionSeed);
        }       

        double endTime = GetCPUSecond();
        totalTime = totalTime + ((endTime - startTime) * 1000);

        //decrypt the encrypted audio matrix
        //perform inverse diffusion and inverse confusion operations on the encrypted audio matrix
        for(int i = 0; i < CONFUSION_DIFFUSION_ROUNDS; i++){
            InverseConfusion(audioMatrix, confusionSeed);
            finalKeyArrayIdx = inverseDiffusion(audioMatrix, finalKeyArray, diffusionSeed[CONFUSION_DIFFUSION_ROUNDS - i - 1], finalKeyArrayIdx);
        }

        //perform inverse confusion operations on the encrypted audio matrix
        for(int i = 0; i < CONFUSION_ROUNDS; i++)
            InverseConfusion(audioMatrix, confusionSeed);

        //convert the decrypted audio matrix into a linear buffer
        convertMatrixToLinearBuffer(audioBuffer, audioMatrix);

        //play back the audio samples stored in audioBuffer
        int err = snd_pcm_writei(playbackHandler, audioBuffer, AUDIO_BUFFER_SIZE);
        if (err < 0) {  
            fprintf(stderr, "Error playing audio: %s\n", snd_strerror(err)); 
            snd_pcm_prepare(playbackHandler);  
        } 

        usleep(30000);

        audioMatrixIdx ++;
    }

    printf("waiting for the buffered decrypted audio samples to complete playback ...\n");
    snd_pcm_drain(playbackHandler);
    
    printf("average encryption time of audio matrices: %f(ms)\n", totalTime / audioMatrixIdx);

    free(audioBuffer);
    free(iterationResultArray1);
    free(iterationResultArray2);
    free(int16tArray1);
    free(int16tArray2);
    free(finalKeyArray);

    snd_pcm_close(playbackHandler);
    fclose(quantumBitFilePointer);
    fclose(originalAudioFilePointer);

    return 0;
}

//initialzie the playback device
snd_pcm_t * InitializePlaybackDevice(){
    snd_pcm_t           * playbackHandler;
    snd_pcm_hw_params_t * playbackParams;
    int                 err;

    //open the playback device
    if((err = snd_pcm_open(&playbackHandler, PCM_DEVICE, SND_PCM_STREAM_PLAYBACK, 0)) < 0){
        fprintf(stderr, "unable to open playback device: %s\n", snd_strerror(err));  
        exit(1); 
    }

    //set up the playback device
    unsigned int sampleRate = PLAYBACK_SAMPLE_RATE;
    snd_pcm_hw_params_alloca(&playbackParams);
    snd_pcm_hw_params_any(playbackHandler, playbackParams);
    snd_pcm_hw_params_set_access(playbackHandler, playbackParams, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(playbackHandler, playbackParams, SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_rate_near(playbackHandler, playbackParams, &sampleRate, 0);
    snd_pcm_hw_params_set_channels(playbackHandler, playbackParams, CHANNELS);
    snd_pcm_hw_params(playbackHandler, playbackParams);

    return playbackHandler;
}

double GetCPUSecond(void){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

//calculate SHA256 hash of the original audio samples
void calculateSHA256Hash(int16_t * audioMatrixData, size_t dataLength, unsigned char * SHA256HashResultArray){
    //SHA256 structure
    SHA256_CTX sha256;

    //initialize the structure
    SHA256_Init(& sha256);

    //update the structure with the frame data
    SHA256_Update(& sha256, audioMatrixData, dataLength);

    //finalize the hash and store the result
    SHA256_Final(SHA256HashResultArray, & sha256);
}

//generate initial condition or control parameter using the SHA256 hash of the original frame ann the quantum bits
double constructParameters(unsigned char * SHA256HashResultArray, FILE * fp){
    //fetch 64 quantum bits to construct a 64-bit value
    uint64_t quantumValue = 0;
    for(int i = 0; i < 64; i++){
        quantumValue = quantumValue << 1;

        char bit = fgetc(fp);
        if(bit == '1')
            quantumValue = quantumValue | 0b1;

        else if(bit == EOF){
            printf("insufficient quantum bits\n");
            exit(1);
        }
    }

    //fetch 8 bytes from the SHA256 hash to construct a 64-bit value
    uint64_t SHA256HashValue = 0;
    memcpy(&SHA256HashValue, SHA256HashResultArray, 8);

    //perform XOR operations
    uint64_t data = quantumValue ^ SHA256HashValue;

    double result = (double)data / 0x1p64;

    quantumBitsConsumed = quantumBitsConsumed + 64;

    return result;
}


//iterate the PLCM and return the iteration result
double PLCM(double initialCondition, double controlParameter){
    double iterationResult = 0;

    if(initialCondition >= 0 && initialCondition <= controlParameter)
        iterationResult = initialCondition / controlParameter;
    
    else if(initialCondition > controlParameter && initialCondition <= 0.5)
        iterationResult = (initialCondition - controlParameter) / (0.5 - controlParameter);
    
    else
        iterationResult = PLCM(controlParameter, 1 - initialCondition);

    return iterationResult;
}

//iterate the PLCM and store results
double IteratePLCM(double controlParameter, double initialCondition, int iterations, double * iterationResultArray){
    double iterationResult = 0;

    for(int i = 0; i < iterations; i ++){
        iterationResult = PLCM(controlParameter, initialCondition);
        initialCondition = iterationResult;
        iterationResultArray[i] = iterationResult;
    }

    return initialCondition;
}

//convert an iteration result into int16_t data
void ConvertIterationResultToInt16t(double * iterationResultArray, int16_t * int16tArray, int iterations){
    int16_t * p;
    for(int i = 0; i < iterations; i++){
        p = & int16tArray[i * INT16_RESERVED];
        memcpy(p, (int16_t *)&iterationResultArray[i], INT16_RESERVED * sizeof(int16_t));
    }
}

//convert the linear buffer into a two-dimensional matrix
void convertLinearBufferToMatrix(int16_t * buffer, int16_t matrix[AUDIO_MATRIX_ROWS_COLUMNS][AUDIO_MATRIX_ROWS_COLUMNS]){
    int idx = 0;

    for(int i = 0; i < AUDIO_MATRIX_ROWS_COLUMNS; i++)
        for(int j = 0; j < AUDIO_MATRIX_ROWS_COLUMNS; j++)
            matrix[i][j] = buffer[idx ++];
}

//fetch a set of quantum bits to generate a uint16_t data for confusion or diffusion operations
uint16_t GenerateUint16tDataFromQuantumBits(FILE * fp){
    uint16_t data = 0;

    for(int i = 0; i < 16; i ++){
        char bit = fgetc(fp);

        if(bit == '1')
            data = data | 0b1;
        else if(bit == EOF){
            printf("insufficient qubits\n");
            exit(1);
        }

        data = data << 1;
    }

    quantumBitsConsumed = quantumBitsConsumed + 16;

    return data;
}

//perform confusion operation on the matrix
void Confusion(int16_t matrix[AUDIO_MATRIX_ROWS_COLUMNS][AUDIO_MATRIX_ROWS_COLUMNS], uint16_t confusionSeed){
    int16_t tempMatrix[AUDIO_MATRIX_ROWS_COLUMNS][AUDIO_MATRIX_ROWS_COLUMNS];
    for(int i = 0; i < AUDIO_MATRIX_ROWS_COLUMNS; i++)
        for(int j = 0; j < AUDIO_MATRIX_ROWS_COLUMNS; j++)
            tempMatrix[i][j] = matrix[i][j];

    for(int i = 0; i < AUDIO_MATRIX_ROWS_COLUMNS; i++)
        for(int j = 0; j < AUDIO_MATRIX_ROWS_COLUMNS; j++){
            int newi = (i + j) % AUDIO_MATRIX_ROWS_COLUMNS;
            int temp = round(confusionSeed * sin(2 * PI * newi / AUDIO_MATRIX_ROWS_COLUMNS));
            int newj = ((j + temp) % AUDIO_MATRIX_ROWS_COLUMNS + AUDIO_MATRIX_ROWS_COLUMNS) % AUDIO_MATRIX_ROWS_COLUMNS;

            matrix[newi][newj] = tempMatrix[i][j];
    }
}

//fetch a set of quantum bits to generate a uint16_t data for diffusion operations
int16_t GenerateInt16tDataFromQuantumBits(FILE * fp){
    int16_t data = 0;

    for(int i = 0; i < 16; i ++){
        char bit = fgetc(fp);

        if(bit == '1')
            data = data | 0b1;
        else if(bit == EOF){
            printf("insufficient qubits\n");
            exit(1);
        }

        data = data << 1;
    }

    quantumBitsConsumed = quantumBitsConsumed + 16;

    return data;
}

//perform diffusion operations on the matrix
int Diffusion(int16_t matrix[AUDIO_MATRIX_ROWS_COLUMNS][AUDIO_MATRIX_ROWS_COLUMNS], int16_t * finalKeyArray, int16_t diffusionSeed, int idx){
    int prei, prej;
    int16_t tempMatrix[AUDIO_MATRIX_ROWS_COLUMNS][AUDIO_MATRIX_ROWS_COLUMNS];
    for(int i = 0; i < AUDIO_MATRIX_ROWS_COLUMNS; i++)
        for(int j = 0; j < AUDIO_MATRIX_ROWS_COLUMNS; j++)
            tempMatrix[i][j] = matrix[i][j];
    
    for(int i = 0; i < AUDIO_MATRIX_ROWS_COLUMNS; i++)
        for(int j = 0; j < AUDIO_MATRIX_ROWS_COLUMNS; j++){
            if(j != 0){
                prei = i;
                prej = j - 1;
                matrix[i][j] = finalKeyArray[idx] ^ ((tempMatrix[i][j] + finalKeyArray[idx])) ^ matrix[prei][prej];
            }

            else if(i != 0 && j == 0){
                prei = i - 1;
                prej = AUDIO_MATRIX_ROWS_COLUMNS - 1;
                matrix[i][j] = finalKeyArray[idx] ^ ((tempMatrix[i][j] + finalKeyArray[idx])) ^ matrix[prei][prej];
            }

            else if(i == 0 & j == 0){
                matrix[i][j] = finalKeyArray[idx] ^ ((tempMatrix[i][j] + finalKeyArray[idx])) ^ diffusionSeed;
            }

            idx = idx + 1;
    }
    
    return idx;
}

//perform inverse confusion operation on the matrix
void InverseConfusion(int16_t matrix[AUDIO_MATRIX_ROWS_COLUMNS][AUDIO_MATRIX_ROWS_COLUMNS], uint16_t confusionSeed){
    int16_t tempMatrix[AUDIO_MATRIX_ROWS_COLUMNS][AUDIO_MATRIX_ROWS_COLUMNS];
    for(int i = 0; i < AUDIO_MATRIX_ROWS_COLUMNS; i++)
        for(int j = 0; j < AUDIO_MATRIX_ROWS_COLUMNS; j++)
            tempMatrix[i][j] = matrix[i][j];

    for(int i = 0; i < AUDIO_MATRIX_ROWS_COLUMNS; i++)
        for(int j = 0; j < AUDIO_MATRIX_ROWS_COLUMNS; j++){
            int temp = round(confusionSeed * sin(2 * PI * i / AUDIO_MATRIX_ROWS_COLUMNS));
            int newi = ((i - j + temp) % AUDIO_MATRIX_ROWS_COLUMNS + AUDIO_MATRIX_ROWS_COLUMNS) % AUDIO_MATRIX_ROWS_COLUMNS;
            int newj = ((j - temp) % AUDIO_MATRIX_ROWS_COLUMNS + AUDIO_MATRIX_ROWS_COLUMNS) % AUDIO_MATRIX_ROWS_COLUMNS;

            matrix[newi][newj] = tempMatrix[i][j];
    }
}

//perform inverse diffusion operation on the matrix
int inverseDiffusion(int16_t matrix[AUDIO_MATRIX_ROWS_COLUMNS][AUDIO_MATRIX_ROWS_COLUMNS], int16_t * finalKeyArray, int16_t diffusionSeed, int idx){
    int prei, prej;
    int16_t tempMatrix[AUDIO_MATRIX_ROWS_COLUMNS][AUDIO_MATRIX_ROWS_COLUMNS];
    for(int i = 0; i < AUDIO_MATRIX_ROWS_COLUMNS; i++)
        for(int j = 0; j < AUDIO_MATRIX_ROWS_COLUMNS; j++)
            tempMatrix[i][j] = matrix[i][j];

    for(int i = AUDIO_MATRIX_ROWS_COLUMNS - 1; i >= 0; i--)
        for(int j = AUDIO_MATRIX_ROWS_COLUMNS - 1; j >= 0; j--){

            idx = idx - 1;

            if(j != 0){
                prei = i;
                prej = j - 1;
                matrix[i][j] = ((finalKeyArray[idx] ^ tempMatrix[i][j] ^ tempMatrix[prei][prej]) - finalKeyArray[idx]);
            }

             else if(i != 0 && j == 0){
                prei = i - 1;
                prej = AUDIO_MATRIX_ROWS_COLUMNS - 1;
                matrix[i][j] = ((finalKeyArray[idx] ^ tempMatrix[i][j] ^ tempMatrix[prei][prej]) - finalKeyArray[idx]);
            }

            else if(i == 0 & j == 0){
                matrix[i][j] = ((finalKeyArray[idx] ^ tempMatrix[i][j] ^ diffusionSeed) - finalKeyArray[idx]);
            }
    }

    return idx;
}

//convert the two-dimensional matrix into a linear buffer
void convertMatrixToLinearBuffer(int16_t * buffer, int16_t matrix[AUDIO_MATRIX_ROWS_COLUMNS][AUDIO_MATRIX_ROWS_COLUMNS]){
    int idx = 0;

    for(int i = 0; i < AUDIO_MATRIX_ROWS_COLUMNS; i++)
        for(int j = 0; j < AUDIO_MATRIX_ROWS_COLUMNS; j++)
            buffer[idx ++] = matrix[i][j];
}

