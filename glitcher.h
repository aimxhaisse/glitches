#ifndef GLITCHER_H
#define GLITCHER_H

#include <stdint.h>

#define GLITCH_WIDTH		1440
#define GLITCH_HEIGHT		900
#define GLITCH_AREA		(GLITCH_WIDTH * GLITCH_HEIGHT) * sizeof(uint32_t)
#define MAX_GENERATIONS		4096

int buffer_to_jpeg(void *buffer, size_t buffer_size, char *output);

#endif /* GLITCHER_H */
