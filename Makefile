# Simple Makefile for Vector Search

CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall
TARGET = search
SOURCE = main.cpp

# Build the program
build: $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)

# Build and run
run: build
	./$(TARGET)

# Build and run with specified vector count (e.g., make run-1000)
run-%: build
	./$(TARGET) $*

.PHONY: build run
