# Real Engine Platformer prototype.
# Builds the game on top of the engine sources in ../src and compiles the
# engine shaders into bin/shaders, where the renderer looks for them.
CC      := clang
ENGINE  := ..
CFLAGS  := -std=c23 -O2 -g -Wall -Wextra -Werror -pedantic -Wconversion \
           -ffp-contract=off -I$(ENGINE)/src -Isrc
LDFLAGS :=
LIBS    := -lm -lpthread -ldl

ifeq ($(shell test -f /usr/include/vulkan/vulkan.h && echo yes),yes)
CFLAGS += -DRE_HAS_VULKAN
endif
ifeq ($(shell test -f /usr/include/X11/Xlib.h && echo yes),yes)
CFLAGS += -DRE_HAS_X11
endif
ifeq ($(shell test -f /usr/include/X11/Xlib.h && test -f /usr/include/vulkan/vulkan.h && echo yes),yes)
LIBS += -lX11
endif

GAME_HEADERS := $(wildcard src/*.h) $(wildcard $(ENGINE)/src/*.h)
GAME_SRCS := src/main.c src/game.c src/scene.c src/character.c
GAME_OBJS := $(patsubst src/%.c,build/%.o,$(GAME_SRCS))

ENGINE_SRCS := $(filter-out $(ENGINE)/src/main.c,$(wildcard $(ENGINE)/src/*.c))
ENGINE_OBJS := $(patsubst $(ENGINE)/src/%.c,build/engine/%.o,$(ENGINE_SRCS))

PHYS_SRCS := re_core re_thread re_mem re_vec re_bvh re_gjk re_solver re_world
PHYS_OBJS := $(patsubst %,build/engine/%.o,$(PHYS_SRCS))

TEST_OBJS := build/test_game.o build/game.o build/scene.o build/character.o $(PHYS_OBJS)

TARGET  := bin/platformer
TEST    := bin/test_game
SHADERS := cull.comp pbr_clustered.vert pbr_clustered.frag shadow.vert tonemap.frag tonemap.vert
SPVS    := $(patsubst %,bin/shaders/%.spv,$(SHADERS))

all: folders $(TARGET) $(TEST) $(SPVS)

$(TARGET): $(GAME_OBJS) $(ENGINE_OBJS) Makefile
	$(CC) $(LDFLAGS) -o $@ $(GAME_OBJS) $(ENGINE_OBJS) $(LIBS)

$(TEST): $(TEST_OBJS) Makefile
	$(CC) $(LDFLAGS) -o $@ $(TEST_OBJS) -lm -lpthread

build/%.o: src/%.c $(GAME_HEADERS) Makefile
	$(CC) $(CFLAGS) -c $< -o $@

build/engine/%.o: $(ENGINE)/src/%.c $(GAME_HEADERS) Makefile
	@mkdir -p build/engine
	$(CC) $(CFLAGS) -c $< -o $@

build/test_game.o: tests/test_game.c $(GAME_HEADERS) Makefile
	$(CC) $(CFLAGS) -c $< -o $@

bin/shaders/%.spv: $(ENGINE)/shaders/%
	@mkdir -p bin/shaders
	glslangValidator -V --target-env vulkan1.3 $< -o $@

folders:
	mkdir -p build/engine bin bin/shaders

test: $(TEST)
	./$(TEST)

run: $(TARGET) $(SPVS)
	./$(TARGET)

clean:
	rm -rf build bin

.PHONY: all test run folders clean
