CXX = g++
CXXFLAGS = -std=c++17 -O2

.PHONY: all run clean

all: tilda

tilda: TildaModel.cpp Functii.h
	$(CXX) $(CXXFLAGS) TildaModel.cpp -o tilda

run: tilda
	./tilda

clean:
	@if [ -f tilda ]; then rm tilda; fi
