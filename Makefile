# Real Engine - C23 data-oriented 3D game engine.
# Build layout follows the Laboratorio di Algoritmi course convention:
#   src/    sources and headers
#   build/  object files
#   bin/    executables and compiled shaders

CC      := clang
CFLAGS  := -std=c23 -O2 -g -Wall -Wextra -Werror -pedantic -Wconversion -ffp-contract=off
LDFLAGS :=
LIBS    := -lm -lpthread -ldl

# Vulkan headers present? -> compile the Vulkan backend (still loaded at
# runtime via dlopen, never linked).
ifeq ($(shell test -f /usr/include/vulkan/vulkan.h && echo yes),yes)
CFLAGS += -DRE_HAS_VULKAN
endif
# X11 headers present? -> windowed path.
ifeq ($(shell test -f /usr/include/X11/Xlib.h && echo yes),yes)
CFLAGS += -DRE_HAS_X11
endif
ifeq ($(shell test -f /usr/include/X11/Xlib.h && test -f /usr/include/vulkan/vulkan.h && echo yes),yes)
LIBS += -lX11
endif

# Optional AVX2 build (deterministic-per-build; scalar is the reference).
ifeq ($(RE_AVX2),1)
CFLAGS += -mavx2 -DRE_USE_AVX2
endif

INCLUDES    := $(wildcard src/*.h)
SRCS        := $(wildcard src/*.c)
OBJS        := $(patsubst src/%.c,build/%.o,$(SRCS))
COMMON_DEPS := $(INCLUDES) Makefile folders

TARGET      := bin/real_engine
SHADER_DIR  := bin/shaders
SHADERS     := cull.comp pbr_clustered.vert pbr_clustered.frag shadow.vert tonemap.frag tonemap.vert
SPVS        := $(patsubst %,$(SHADER_DIR)/%.spv,$(SHADERS))

all: $(TARGET)

$(TARGET): $(OBJS) $(COMMON_DEPS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

build/%.o: src/%.c $(COMMON_DEPS)
	$(CC) $(CFLAGS) -c $< -o $@

shaders: $(SPVS)

$(SHADER_DIR)/%.spv: shaders/%
	@mkdir -p $(SHADER_DIR)
	glslangValidator -V --target-env vulkan1.3 $< -o $@

folders:
	mkdir -p build/ bin/

verify: all
	./verify.sh

doc:
	doxygen Doxyfile

clean:
	rm -rf build/ bin/

rebuild: clean all

.PHONY: all shaders verify doc folders clean rebuild
