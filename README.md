## A Real-Time Audio and Video Encryption Framework for Low-Bit-Rate Quantum Networks and Its industrial Application

### Software Description

Both the audio and video encryption softwares are developed on a workstation equipped with an Intel Xeon Gold 6226R CPU and 64 GB of memory, running on Ubuntu 22.04.

- audio encryption : The Advanced Linux Sound Architecture (ALSA) driver, version k6.8.0-45-generic, is employed for processing audio samples. The audio encryption demonstration software captures audio samples from the original audio file, encrypts the captured samples, decrypts the resulting data, and ultimately plays back the decrypted audio samples through headphones.

- video encryption : The OpenCV version 4.6.0 is utilized for processing video frames. The video encryption demonstration software captures frames from the original video file, encrypts the captured frames, decrypts the resulting frames, and ultimately displays the decrypted video frames.

### File Description

- audioEncryption directory : Source files for the audio encryption utilized within the proposed framework. Executing the setup.sh script directly compiles the source files.

- videoEncryption directory : Source files for the video encryption utilized within the proposed framework. Executing the setup.sh script directly compiles the source files.

- quantumBits directory : Quantum bits generated in the laboratory are stored in this directory.

- originalData directory: original audio and video files are stored in this directory.

- videoEncryption.webm : Demonstration video of real-time video encryption and decryption.

### Software Configuration Instructions

For audio encryption, the parameters such as sampling rate, sample width of the audio, ronds of confusion operations $r_c$, and rounds of combined diffusion and confusion operations $r_d$ are defined in audioEncryption.h

 ```
//macros for the initialization of capture and playbcak devices
#define PCM_DEVICE                  "default"
#define CAPTURE_SAMPLE_RATE         44100
#define PLAYBACK_SAMPLE_RATE        CAPTURE_SAMPLE_RATE - 100
#define CHANNELS                    1

//memory allocated for stroing the captured audio samples
#define AUDIO_MATRIX_ROWS_COLUMNS   64
#define AUDIO_BUFFER_SIZE           AUDIO_MATRIX_ROWS_COLUMNS * AUDIO_MATRIX_ROWS_COLUMNS

//macros for audio encryption and decryption
#define CONFUSION_ROUNDS            3
#define CONFUSION_DIFFUSION_ROUNDS  3
```

Similarly, the parameters for video encryption are defined in videoEncryption. h

```
#define FRAME_WIDTH                 512
#define FRAME_HEIGHT                512
#define VIDEO_FPS                   24

#define CONFUSION_ROUNDS            3
#define DIFFUSION_CONFUSION_ROUNDS  3
#define NUMBER_OF_THREADS           32
```

The software can be executed with the new configuration by modifying these parameters and recompiling the code.

### Special Instructions

- The software uses the default device for the headphones. Please ensure that the device selection is correct and functioning properly when running the software.

- To ensure proper software functionality, the audio matrix must have equal length and width, and the video frame must also have equal length and width.

- The number of worker threads for video encryption must be divisible by either the length or the width of the video frame.

- Using audio sample buffering in the software may cause inconsistencies with the ALSA buffer speed, resulting in playback noise and stuttering. To resolve this, adjust the PLAYBACK_SAMPLE_RATE macro in audioEncryption.h.