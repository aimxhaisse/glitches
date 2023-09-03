#ifndef MEM_GLITCHER_H
#define MEM_GLITCHER_H

#include <stdio.h>
#include <stdlib.h>

/* Main structure for the memory glitcher */
typedef struct mem_glitcher_s {
  FILE*		fh;
  size_t	size;
  char		buffer[GLITCH_AREA];
} mem_glitcher_t;

mem_glitcher_t *mem_glitcher_new();
int mem_glitch(mem_glitcher_t *glitcher, char *output);
void mem_glitcher_free(mem_glitcher_t *glitcher);

#endif /* MEM_GLITCHER_H */
