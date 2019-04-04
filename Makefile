compile_gc:
	g++ -std=c++11 -g -O3 -I/usr/local/include/ -I../graphchi-cpp/src/ -fopenmp -Wall -Wno-strict-aliasing -lpthread -DSKETCH_SIZE=2000 -DK_HOPS=3 -DDEBUG -DPREGEN=10000 -DMEMORY=1 -g -I../graphchi-cpp/streaming/ ../graphchi-cpp/streaming/main.cpp -o ../graphchi-cpp/bin/streaming/main -lz
