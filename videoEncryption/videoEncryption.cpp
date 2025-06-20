#include "videoEncryption.hpp"

int frameIdx, quantumBitsConsumed, confusionSeed;
Mat originalFrame, encryptedFrame, decryptedFrame, tempFrame;
sem_t wakeAssistantThreadMutex[NUMBER_OF_THREADS], waitAssistantThreadToCompleteTaskMutex[NUMBER_OF_THREADS];

int main(int argc, const char ** argv){
    //open the video file
    VideoCapture capture;
    capture.open(ORIGINAL_VIDEO_FILE);
    if(!capture.isOpened()){
        printf("failed to open the video file\n");
        exit(1);
    }

    //open the file that stores generated quantum bits
    FILE * quantumBitFilePointer = fopen(QUANTUM_BIT_FILE_POINTER, "r");
    if(quantumBitFilePointer == NULL){
        printf("failed to open the quantum bit file\n");
        exit(1);
    }

    //SHA256 hash of the original frame is stored in this array
    unsigned char SHA256HashResultArray[SHA256_DIGEST_LENGTH];

    //iterations of PLCMs for processing a sub-frame
    int iterations              = int((3 * FRAME_WIDTH * FRAME_HEIGHT * DIFFUSION_CONFUSION_ROUNDS) / (NUMBER_OF_THREADS * BYTES_RESERVED)) + 1;
    //this array stores parameters for initializing PLCMs of assistant threads
    double * initParameterArray = (double *)malloc(4 * NUMBER_OF_THREADS * sizeof(double));

    //initialize the semaphores
    for(int i = 0; i < NUMBER_OF_THREADS; i++){
        sem_init(&wakeAssistantThreadMutex[i], 0, 0);
        sem_init(&waitAssistantThreadToCompleteTaskMutex[i], 0, 0);
    }

    //create assistant threads
    struct assistantThreadParameter tp[NUMBER_OF_THREADS];
    for(int i = 0; i < NUMBER_OF_THREADS; i++){
        tp[i].threadIdx          = i;
        tp[i].iterations         = iterations;
        tp[i].initParameterArray = initParameterArray;
    }
    pthread_t th[NUMBER_OF_THREADS];
    for(int i = 0; i < NUMBER_OF_THREADS; i++)
        pthread_create(&th[i], NULL, assistantThreadFunc, (void *)&tp[i]);

    double initialCondition1, controlParameter1;
    double initialCondition2, controlParameter2;
    double startTime, endTime;
    double totalEncryptionTime = 0;
    double delayedFrameCount   = 0;
    frameIdx                   = 0;
    quantumBitsConsumed        = 0;

    while(1){
        //fetch an original frame from the camera
        capture >> originalFrame;
        if(originalFrame.empty())
            break;

        tempFrame      = originalFrame.clone();
        encryptedFrame = originalFrame.clone();

        startTime      = GetCPUSecond();

    
        //calculate the SHA256 hash of the original frame
        calculateSHA256Hash(originalFrame.data, FRAME_WIDTH * FRAME_HEIGHT * 3, SHA256HashResultArray);

        //construct initial conditions and control parameters using the SHA256 hash of the original frame and the quantum bits
        initialCondition1 = constructParameters(&SHA256HashResultArray[0], quantumBitFilePointer);
        controlParameter1 = constructParameters(&SHA256HashResultArray[8], quantumBitFilePointer);
        if(controlParameter1 > 0.5) 
            controlParameter1 = controlParameter1 - 0.5;

        initialCondition2 = constructParameters(&SHA256HashResultArray[16], quantumBitFilePointer);
        controlParameter2 = constructParameters(&SHA256HashResultArray[24], quantumBitFilePointer);
        if(controlParameter2 > 0.5)
            controlParameter2 = controlParameter2 - 0.5;
        
        //pre-iterate PLCMs of the main thread
        for(int i = 0; i < PRE_ITERATIONS; i++){
            initialCondition1 = PLCM(initialCondition1, controlParameter1);
            initialCondition2 = PLCM(initialCondition2, controlParameter2);
        }

        //generate parameters for initializing PLCMs of assistant threads
        GenerateParameters(controlParameter1, &initialCondition1, controlParameter2, &initialCondition2, initParameterArray);

        //wake assistant threads to (re)initialize their PLCMs
        for(int i = 0; i < NUMBER_OF_THREADS; i++)
            sem_post(&wakeAssistantThreadMutex[i]);

        //wait assistant threads to complete (re)initialization
        for(int i = 0; i < NUMBER_OF_THREADS; i++)
            sem_wait(&waitAssistantThreadToCompleteTaskMutex[i]);

        //generate a confusion seed for confusion operations
        confusionSeed = abs(GenerateConfusionSeedFromQuantumBits(quantumBitFilePointer)) % CONFUSION_SEED_UPPER_BOUND + CONFUSION_SEED_LOWWER_BOUND;

        //perform confusion operations
        for(int i = 0; i < CONFUSION_ROUNDS; i++){
            //wake assistant threads to perform confusion operations
            for(int i = 0; i < NUMBER_OF_THREADS; i++)
                sem_post(&wakeAssistantThreadMutex[i]);

            //wait assistant threads to complete confusion operations
            for(int i = 0; i < NUMBER_OF_THREADS; i++)
                sem_wait(&waitAssistantThreadToCompleteTaskMutex[i]);

            tempFrame = encryptedFrame.clone();
        }

        //perform diffusion and confusion operations
        for(int i = 0; i < DIFFUSION_CONFUSION_ROUNDS; i++){
            //wake assistant threads to perform diffusion operations
            for(int i = 0; i < NUMBER_OF_THREADS; i++)
                sem_post(&wakeAssistantThreadMutex[i]);

            //wait assistant threads to complete diffusion operations
            for(int i = 0; i < NUMBER_OF_THREADS; i++)
                sem_wait(&waitAssistantThreadToCompleteTaskMutex[i]);

            tempFrame = encryptedFrame.clone();

            //wake assistant threads to perform confusion operations
            for(int i = 0; i < NUMBER_OF_THREADS; i++)
                sem_post(&wakeAssistantThreadMutex[i]);

            //wait assistant threads to complete confusion operations
            for(int i = 0; i < NUMBER_OF_THREADS; i++)
                sem_wait(&waitAssistantThreadToCompleteTaskMutex[i]);

            tempFrame = encryptedFrame.clone();
        }

        endTime = GetCPUSecond();
        totalEncryptionTime = totalEncryptionTime + (endTime - startTime) * 1000;
        if(int((endTime - startTime) * 1000) > int(1000 / VIDEO_FPS))
            delayedFrameCount ++;
        
        //decrypt the encrypted frame
        decryptedFrame = encryptedFrame.clone();

        //perform inverse confusion and diffusion operations 
        for(int i = 0; i < DIFFUSION_CONFUSION_ROUNDS; i++){
            //wake assistant threads to perform inverse confusion operations
            for(int i = 0; i < NUMBER_OF_THREADS; i++)
                sem_post(&wakeAssistantThreadMutex[i]);

            //wait assistant threads to complete inverse confusion operations
            for(int i = 0; i < NUMBER_OF_THREADS; i++)
                sem_wait(&waitAssistantThreadToCompleteTaskMutex[i]);

            tempFrame = decryptedFrame.clone();

            //wake assistant threads to perform inverse diffusion operations
            for(int i = 0; i < NUMBER_OF_THREADS; i++)
                sem_post(&wakeAssistantThreadMutex[i]);

            //wait assistant threads to complete inverse diffusion operations
            for(int i = 0; i < NUMBER_OF_THREADS; i++)
                sem_wait(&waitAssistantThreadToCompleteTaskMutex[i]);

            tempFrame = decryptedFrame.clone();
        }

        //perform inverse confusion operations
        for(int i = 0; i < CONFUSION_ROUNDS; i++){
            //wake assistant threads to perform inverse confusion operations
            for(int i = 0; i < NUMBER_OF_THREADS; i++)
                sem_post(&wakeAssistantThreadMutex[i]);

            //wait assistant threads to complete inverse confusion operations
            for(int i = 0; i < NUMBER_OF_THREADS; i++)
                sem_wait(&waitAssistantThreadToCompleteTaskMutex[i]);

            tempFrame = decryptedFrame.clone();
        }

        //display frames
        imshow("original frame", originalFrame);
        imshow("encrypted frame", encryptedFrame);
        imshow("decrypted frame", decryptedFrame);

        endTime = GetCPUSecond();

        if(frameIdx % FRAMES_TO_PROCESS == 0){
            system("clear");
            if(frameIdx != 0){
                printf("\033[1mvideo encrytpion and decryption using quantum bits\033[0m\n");
                printf("frame width: %d frame height: %d video FPS: %d\n", FRAME_WIDTH, FRAME_HEIGHT, VIDEO_FPS);
                printf("rounds of confusion operations: %d\n", CONFUSION_ROUNDS + DIFFUSION_CONFUSION_ROUNDS);
                printf("rounds of diffusion operations: %d\n", DIFFUSION_CONFUSION_ROUNDS);
                printf("number of assistant threads: %d\n", NUMBER_OF_THREADS);
                printf("initial condition 1: %f control parameter 1: %f\n", initialCondition1, controlParameter1);
                printf("initial condition 2: %f control parameter 2: %f\n", initialCondition2, controlParameter2);
                printf("confusion seed: %d\n", confusionSeed);
                printf("quantum bits consumed: %.3f (kb)\n", double(quantumBitsConsumed) / 1024);
                printf("video frames that has been processed: %d\n", frameIdx);
                printf("time for encrypting and decrypting the frame: %.3f (ms)\n", (endTime - startTime) * 1000);
            }
        }
        frameIdx ++;
        
        int waitTime = (int(1000 / VIDEO_FPS) - int((endTime - startTime) * 1000));
        if(waitTime > 1)
            waitKey(waitTime);
        else{
            waitKey(1);
        }
    }

    printf("average encryption time: %f latency rate: %f\n", totalEncryptionTime / frameIdx, delayedFrameCount / frameIdx);

    capture.release();
    destroyAllWindows();
    fclose(quantumBitFilePointer);
    free(initParameterArray);
    for(int i = 0; i < NUMBER_OF_THREADS; i++){
        sem_destroy(&wakeAssistantThreadMutex[i]);
        sem_destroy(&waitAssistantThreadToCompleteTaskMutex[i]);
    }
    for(int i = 0; i < NUMBER_OF_THREADS; i++)
        pthread_cancel(th[i]);

    return 0;
}

//assistant thread execute this function to encrypt and decrypt the video frames
static void * assistantThreadFunc(void * arg){
    struct assistantThreadParameter * p = (struct assistantThreadParameter *)arg;
    int threadIdx                      = p->threadIdx;
    int iterations                     = p->iterations;
    double * initParameterArray        = p->initParameterArray;

    int cols        = FRAME_WIDTH;
    int rows        = FRAME_HEIGHT / NUMBER_OF_THREADS;
    int startingRow = threadIdx   * rows;
    int endingRow   = startingRow + rows;

    double * iterationResultArray1     = (double *)malloc(iterations * sizeof(double));
    double * iterationResultArray2     = (double *)malloc(iterations * sizeof(double));
    unsigned char * uCharResultArray1  = (unsigned char *)malloc(iterations * BYTES_RESERVED * sizeof(unsigned char));
    unsigned char * uCharResultArray2  = (unsigned char *)malloc(iterations * BYTES_RESERVED * sizeof(unsigned char));
    unsigned char * byteSequence       = (unsigned char *)malloc(iterations * BYTES_RESERVED * sizeof(unsigned char));
    unsigned char * diffusionSeedArray = (unsigned char *)malloc(3 * DIFFUSION_CONFUSION_ROUNDS * sizeof(unsigned char));


    double initialCondition1, controlParameter1;
    double initialCondition2, controlParameter2;
    unsigned char diffusionSeed[3];

    while(true){
        //wait to be awakened by the main thead
        sem_wait(&wakeAssistantThreadMutex[threadIdx]);

        controlParameter1 = initParameterArray[threadIdx * 4];
        if(controlParameter1 > 0.5)
            controlParameter1 = 1 - controlParameter1;
        initialCondition1 = initParameterArray[threadIdx * 4 + 1];

        controlParameter2 = initParameterArray[threadIdx * 4 + 2];
        if(controlParameter2 > 0.5)
            controlParameter2 = 1 - controlParameter2;
        initialCondition2 = initParameterArray[threadIdx * 4 + 3];

        //pre-iterate PLCMs
        for(int i = 0; i < PRE_ITERATIONS; i++){
            initialCondition1 = PLCM(controlParameter1, initialCondition1);
            initialCondition2 = PLCM(controlParameter2, initialCondition2);
        }

        //tell the main thread that (re)initialization is completed
        sem_post(&waitAssistantThreadToCompleteTaskMutex[threadIdx]);

        //generate byte sequence for diffusion operations
        initialCondition1 = IteratePLCM(controlParameter1, initialCondition1, iterations, iterationResultArray1);
        initialCondition2 = IteratePLCM(controlParameter2, initialCondition2, iterations, iterationResultArray2);
        ConvertResultToByte(iterationResultArray1, uCharResultArray1, iterations);
        ConvertResultToByte(iterationResultArray2, uCharResultArray2, iterations);
        GenerateBytes(iterations, uCharResultArray1, uCharResultArray2, byteSequence);

        //generate diffusion seeds
        GenerateDiffusionSeeds(controlParameter1, &initialCondition1, controlParameter2, &initialCondition2,diffusionSeedArray);

        //perform confusion operations on the sub-frame
        for(int i = 0; i < CONFUSION_ROUNDS; i++){
            //wait to be awakened by the main thread
            sem_wait(&wakeAssistantThreadMutex[threadIdx]);

            //perform confusion operations
            Confusion(startingRow, endingRow);

            //notify the main thread that the confusion operations are completed
            sem_post(&waitAssistantThreadToCompleteTaskMutex[threadIdx]);
        }

        int byteSequenceIdx       = 0;    
        int diffusionSeedArrayIdx = 0; 

        //perform diffusion and confusion operation on the sub-frame
        for(int i = 0; i < DIFFUSION_CONFUSION_ROUNDS; i++){
            //wait to be awakened by the main thread
            sem_wait(&wakeAssistantThreadMutex[threadIdx]);

            //fetch the diffusion seeds
            diffusionSeed[0] = diffusionSeedArray[diffusionSeedArrayIdx ++];
            diffusionSeed[1] = diffusionSeedArray[diffusionSeedArrayIdx ++];
            diffusionSeed[2] = diffusionSeedArray[diffusionSeedArrayIdx ++];

            byteSequenceIdx = Diffusion(startingRow, endingRow, diffusionSeed, byteSequence, byteSequenceIdx);

            //notify the main thread that the diffusion operations are completed
            sem_post(&waitAssistantThreadToCompleteTaskMutex[threadIdx]);

            //wait to be awakened by the main thread
            sem_wait(&wakeAssistantThreadMutex[threadIdx]);

            //perform confusion operations
            Confusion(startingRow, endingRow);

            //notify the main thread that the confusion operations are completed
            sem_post(&waitAssistantThreadToCompleteTaskMutex[threadIdx]);
        }

        //decrypt the sub-frame
        //perform inverse confusion and diffusion operations
        for(int i = 0; i < DIFFUSION_CONFUSION_ROUNDS; i++){
            //wait to be awakened by the main thread
            sem_wait(&wakeAssistantThreadMutex[threadIdx]);

            //perform inverse confusion operations
            InverseConfusion(startingRow, endingRow);

            //notify the main thread that the inverse confusion operations are completed
            sem_post(&waitAssistantThreadToCompleteTaskMutex[threadIdx]);

            //wait to be awakened by the main thread
            sem_wait(&wakeAssistantThreadMutex[threadIdx]);

            //fetch the diffusion seeds
            diffusionSeed[2] = diffusionSeedArray[--diffusionSeedArrayIdx];
            diffusionSeed[1] = diffusionSeedArray[--diffusionSeedArrayIdx];
            diffusionSeed[0] = diffusionSeedArray[--diffusionSeedArrayIdx];

            //perform inverse diffusion operations
            byteSequenceIdx = InverseDiffusion(startingRow, endingRow, diffusionSeed, byteSequence, byteSequenceIdx);

            //notify the main thread that the inverse diffusion operations are completed
            sem_post(&waitAssistantThreadToCompleteTaskMutex[threadIdx]);
        }

        //perform inverse confusion operations
        for(int i = 0; i < CONFUSION_ROUNDS; i++){
            //wait to be awakened by the main thread
            sem_wait(&wakeAssistantThreadMutex[threadIdx]);

            //perform inverse confusion operations
            InverseConfusion(startingRow, endingRow);

            //notify the main thread that the inverse confusion operations are completed
            sem_post(&waitAssistantThreadToCompleteTaskMutex[threadIdx]);
        }

    }

    free(iterationResultArray1);
    free(iterationResultArray2);
    free(uCharResultArray1);  
    free(uCharResultArray2);
    free(byteSequence);
    free(diffusionSeedArray);

    return NULL;
}

double GetCPUSecond(void){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

//calculate SHA256 hash of the original frame
void calculateSHA256Hash(unsigned char * frameData, size_t dataLength, unsigned char * SHA256HashResultArray){
    //SHA256 structure
    SHA256_CTX sha256;

    //initialize the structure
    SHA256_Init(& sha256);

    //update the structure with the frame data
    SHA256_Update(& sha256, frameData, dataLength);

    //finalize the hash and store the result
    SHA256_Final(SHA256HashResultArray, & sha256);
}

//generate initial condition or control parameter using the SHA256 hash of the original frame and the quantum bits
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

//generate parameters for initializing PLCMs of assistant threads
void GenerateParameters(double controlParameter1, double * initialCondition1, 
                        double controlParameter2, double * initialCondition2,
                        double * initParameterArray){

    for(int i = 0; i < NUMBER_OF_THREADS * 4; i++){
        if(i % 2 == 0){
            initParameterArray[i] = PLCM(controlParameter1, * initialCondition1);
            * initialCondition1   = initParameterArray[i];
        }

        else
            initParameterArray[i] = PLCM(controlParameter2, * initialCondition2);
            * initialCondition2   = initParameterArray[i];
    }
}

//fetch 16 quantum bits to construct a confusion seed
int GenerateConfusionSeedFromQuantumBits(FILE * fp){
    int data = 0;

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

//convert iteration results to byte sequence
void ConvertResultToByte(double * resultArray, unsigned char * byteSequence, int elems){
    unsigned char * p;
    for(int i = 0; i < elems; i++){
        p = &byteSequence[i * BYTES_RESERVED];
        memcpy(p, (unsigned char *)&resultArray[i], BYTES_RESERVED);   
    }
}

//Act XOR operation and store the bytes for encryption
void GenerateBytes(int iterations, unsigned char * uCharResultArray1, unsigned char * uCharResultArray2, unsigned char * byteSequence){
     int n = iterations * BYTES_RESERVED;
    
    for(int i = 0; i < n; i++ )
        byteSequence[i] = uCharResultArray1[i] ^ uCharResultArray2[i];
}

//use the chaotic map to generate diffusion seeds for encryption a frame
void GenerateDiffusionSeeds(double controlParameter1, double * initialCondition1, 
                            double controlParameter2, double * initialCondition2,
                            unsigned char * diffusionSeedArray){

    for(int i = 0; i < 3 * DIFFUSION_CONFUSION_ROUNDS; i++){
        double iterationResult1 = PLCM(* initialCondition1, controlParameter1);
        * initialCondition1     = iterationResult1;

        double iterationResult2 = PLCM(* initialCondition2, controlParameter2);
        * initialCondition2     = iterationResult2;

        unsigned char temp1, temp2;
        memcpy(&temp1, (unsigned char *) &iterationResult1, 1);
        memcpy(&temp2, (unsigned char *) &iterationResult2, 1);

        diffusionSeedArray[i] = temp1 ^ temp2;
    }
}

//confusion function
void Confusion(int startingRow, int endingRow){
    for(int r = startingRow; r < endingRow; r++)
        for(int c = 0; c < FRAME_WIDTH; c++){
            int nr = (r + c) % FRAME_HEIGHT;
            int temp = round(confusionSeed * sin(2 * PI * nr / FRAME_HEIGHT));
            int nc = ((c + temp) % FRAME_WIDTH + FRAME_WIDTH) % FRAME_WIDTH;

            encryptedFrame.at<Vec3b>(nr, nc)[0] = tempFrame.at<Vec3b>(r, c)[0];
            encryptedFrame.at<Vec3b>(nr, nc)[1] = tempFrame.at<Vec3b>(r, c)[1];
            encryptedFrame.at<Vec3b>(nr, nc)[2] = tempFrame.at<Vec3b>(r, c)[2];
        }
}

//diffusion function
int Diffusion(int startingRow, int endingRow, unsigned char * diffusionSeed, unsigned char * byteSequence, int idx){
    int prei, prej;

    for(int i = startingRow ; i < endingRow; i++)
        for(int j = 0; j < FRAME_WIDTH; j++){
            if(j != 0){
                prei = i;
                prej = j - 1;
                encryptedFrame.at<Vec3b>(i, j)[0] = byteSequence[idx] ^ ((tempFrame.at<Vec3b>(i,j)[0] + byteSequence[idx]) % 256) ^ encryptedFrame.at<Vec3b>(prei, prej)[0];
                idx = idx + 1;
                encryptedFrame.at<Vec3b>(i, j)[1] = byteSequence[idx] ^ ((tempFrame.at<Vec3b>(i,j)[1] + byteSequence[idx]) % 256) ^ encryptedFrame.at<Vec3b>(prei, prej)[1];
                idx = idx + 1;
                encryptedFrame.at<Vec3b>(i, j)[2] = byteSequence[idx] ^ ((tempFrame.at<Vec3b>(i,j)[2] + byteSequence[idx]) % 256) ^ encryptedFrame.at<Vec3b>(prei, prej)[2];
                idx = idx + 1;
            }

            else if(i != startingRow && j == 0){
                prei = i - 1;
                prej = FRAME_WIDTH - 1;
                encryptedFrame.at<Vec3b>(i, j)[0] = byteSequence[idx] ^ ((tempFrame.at<Vec3b>(i,j)[0] + byteSequence[idx]) % 256) ^ encryptedFrame.at<Vec3b>(prei, prej)[0];
                idx = idx + 1;
                encryptedFrame.at<Vec3b>(i, j)[1] = byteSequence[idx] ^ ((tempFrame.at<Vec3b>(i,j)[1] + byteSequence[idx]) % 256) ^ encryptedFrame.at<Vec3b>(prei, prej)[1];
                idx = idx + 1;
                encryptedFrame.at<Vec3b>(i, j)[2] = byteSequence[idx] ^ ((tempFrame.at<Vec3b>(i,j)[2] + byteSequence[idx]) % 256) ^ encryptedFrame.at<Vec3b>(prei, prej)[2];
                idx = idx + 1;
            }

            else if(i == startingRow && j == 0){
                encryptedFrame.at<Vec3b>(i, j)[0] = byteSequence[idx] ^ ((tempFrame.at<Vec3b>(i,j)[0] + byteSequence[idx]) % 256) ^ diffusionSeed[0];
                idx = idx + 1;
                encryptedFrame.at<Vec3b>(i, j)[1] = byteSequence[idx] ^ ((tempFrame.at<Vec3b>(i,j)[1] + byteSequence[idx]) % 256) ^ diffusionSeed[1];
                idx = idx + 1;
                encryptedFrame.at<Vec3b>(i, j)[2] = byteSequence[idx] ^ ((tempFrame.at<Vec3b>(i,j)[2] + byteSequence[idx]) % 256) ^ diffusionSeed[2];
                idx = idx + 1;
            }
        }
    
    return idx;
}


//inverse confusion function
void InverseConfusion(int startingRow, int endingRow){
    for(int r = startingRow; r < endingRow; r++)
        for(int c = 0; c < FRAME_WIDTH; c++){
            int temp = round(confusionSeed * sin(2 * PI * r / FRAME_HEIGHT));
            int nr = ((r - c + temp)% FRAME_HEIGHT + FRAME_HEIGHT) % FRAME_HEIGHT;
            int nc = ((c - temp) % FRAME_WIDTH + FRAME_WIDTH) % FRAME_WIDTH;

            decryptedFrame.at<Vec3b>(nr, nc)[0] = tempFrame.at<Vec3b>(r, c)[0];
            decryptedFrame.at<Vec3b>(nr, nc)[1] = tempFrame.at<Vec3b>(r, c)[1];
            decryptedFrame.at<Vec3b>(nr, nc)[2] = tempFrame.at<Vec3b>(r, c)[2];
        }
}

//inverse diffusion function
int InverseDiffusion(int startingRow, int endingRow, unsigned char * diffusionSeed, unsigned char * byteSequence, int idx){
    int prei, prej;

    for(int i = endingRow - 1; i >= startingRow; i--)
        for(int j = FRAME_WIDTH - 1; j >= 0; j--){
            if(j != 0){
                prei = i;
                prej = j - 1;

                idx = idx - 1;
                decryptedFrame.at<Vec3b>(i, j)[2] = ((byteSequence[idx] ^ tempFrame.at<Vec3b>(i, j)[2] ^ tempFrame.at<Vec3b>(prei, prej)[2]) + 256 - byteSequence[idx]);
                idx = idx - 1;
                decryptedFrame.at<Vec3b>(i, j)[1] = ((byteSequence[idx] ^ tempFrame.at<Vec3b>(i, j)[1] ^ tempFrame.at<Vec3b>(prei, prej)[1]) + 256 - byteSequence[idx]);
                idx = idx - 1;
                decryptedFrame.at<Vec3b>(i, j)[0] = ((byteSequence[idx] ^ tempFrame.at<Vec3b>(i, j)[0] ^ tempFrame.at<Vec3b>(prei, prej)[0]) + 256 - byteSequence[idx]);
            }

            else if(i != startingRow && j == 0){
                prei = i - 1;
                prej = FRAME_WIDTH - 1;

                idx = idx - 1;
                decryptedFrame.at<Vec3b>(i, j)[2] = ((byteSequence[idx] ^ tempFrame.at<Vec3b>(i, j)[2] ^ tempFrame.at<Vec3b>(prei, prej)[2]) + 256 - byteSequence[idx]);
                idx = idx - 1;
                decryptedFrame.at<Vec3b>(i, j)[1] = ((byteSequence[idx] ^ tempFrame.at<Vec3b>(i, j)[1] ^ tempFrame.at<Vec3b>(prei, prej)[1]) + 256 - byteSequence[idx]);
                idx = idx - 1;
                decryptedFrame.at<Vec3b>(i, j)[0] = ((byteSequence[idx] ^ tempFrame.at<Vec3b>(i, j)[0] ^ tempFrame.at<Vec3b>(prei, prej)[0]) + 256 - byteSequence[idx]);
            }

            else if(i == startingRow && j == 0){
                idx = idx - 1;
                decryptedFrame.at<Vec3b>(i, j)[2] = ((byteSequence[idx] ^ tempFrame.at<Vec3b>(i, j)[2] ^ diffusionSeed[2]) + 256 - byteSequence[idx]);
                idx = idx - 1;
                decryptedFrame.at<Vec3b>(i, j)[1] = ((byteSequence[idx] ^ tempFrame.at<Vec3b>(i, j)[1] ^ diffusionSeed[1]) + 256 - byteSequence[idx]);
                idx = idx - 1;
                decryptedFrame.at<Vec3b>(i, j)[0] = ((byteSequence[idx] ^ tempFrame.at<Vec3b>(i, j)[0] ^ diffusionSeed[0]) + 256 - byteSequence[idx]);
            }

        }

    return idx;
}
