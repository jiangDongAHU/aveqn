g++ videoEncryption.cpp -o out -lpthread -lm -lssl -lcrypto `pkg-config opencv4 --libs --cflags` -Wno-deprecated-declarations
./out
rm out
