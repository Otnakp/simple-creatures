CC      = cc
CFLAGS  = -Wall -Wextra -O2 -DGL_SILENCE_DEPRECATION -I/opt/homebrew/Cellar/glfw/3.4/include
LDFLAGS = -L/opt/homebrew/Cellar/glfw/3.4/lib -lglfw -framework OpenGL -lm

NET_DIR = network
SIM_DIR = sim

NET_SRC  = $(NET_DIR)/network.c
SIM_SRC  = $(SIM_DIR)/world.c $(SIM_DIR)/agent.c $(SIM_DIR)/main.c $(NET_SRC)
TEST_SRC = $(NET_DIR)/network.c $(NET_DIR)/test_network.c

all: sim headless test_network

sim: $(SIM_SRC)
	$(CC) $(CFLAGS) $(SIM_SRC) -o $(SIM_DIR)/sim $(LDFLAGS)

headless: $(SIM_DIR)/main_headless.c $(SIM_DIR)/world.c $(SIM_DIR)/agent.c $(NET_SRC)
	$(CC) $(CFLAGS) $(SIM_DIR)/main_headless.c $(SIM_DIR)/world.c $(SIM_DIR)/agent.c $(NET_SRC) -o $(SIM_DIR)/headless -lm

run_headless: headless
	./$(SIM_DIR)/headless

test_network: $(TEST_SRC)
	$(CC) $(CFLAGS) $(TEST_SRC) -o $(NET_DIR)/test_network -lm

run: sim
	./$(SIM_DIR)/sim

python_test:
	python3 python/test.py

clean:
	rm -f $(SIM_DIR)/sim $(SIM_DIR)/headless $(NET_DIR)/test_network

.PHONY: all sim run headless run_headless python_test clean