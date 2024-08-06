CXX = g++-13
CXXFLAGS = -std=c++17 -g -O3 -march=native -funroll-loops -ftree-vectorize -fopenmp
LDFLAGS = -Wl,-ld_classic -fopenmp

# Source files for each target
TRAIN_SOURCES = \
	./RevGrad/model/Model.cpp \
	./RevGrad/tensor/Tensor.cpp \
	./RevGrad/utill/Print.cpp \
	./model/Model.cpp \
	./player/Trainer.cpp \
	./game/Game.cpp \
    ./Train.cpp

PLAY_SOURCES = \
	./RevGrad/model/Model.cpp \
	./RevGrad/tensor/Tensor.cpp \
	./RevGrad/utill/Print.cpp \
	./model/Model.cpp \
	./player/Human.cpp \
	./player/AI.cpp \
	./game/Game.cpp \
    ./Play.cpp

# Object files for each target
TRAIN_OBJS = $(TRAIN_SOURCES:.cpp=.o)
PLAY_OBJS = $(PLAY_SOURCES:.cpp=.o)

# Targets
TRAIN_TARGET = ./Train
PLAY_TARGET = ./Play

all: $(TRAIN_TARGET) $(PLAY_TARGET)

# Build TRAIN
$(TRAIN_TARGET): $(TRAIN_OBJS)
	$(CXX) -o $@ $(TRAIN_OBJS) $(LDFLAGS)

# Build PLAY
$(PLAY_TARGET): $(PLAY_OBJS)
	$(CXX) -o $@ $(PLAY_OBJS) $(LDFLAGS)

# Rule to compile .cpp files to .o files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Clean up build files
clean:
	rm -f \
        $(TRAIN_TARGET) $(PLAY_TARGET) \
        $(TRAIN_OBJS) $(PLAY_OBJS)
