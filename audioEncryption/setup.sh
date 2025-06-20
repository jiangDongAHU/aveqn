g++ audioEncryption.c -o out -lasound -lm -lssl -lcrypto -Wno-deprecated-declarations
./out
rm out
