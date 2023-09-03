#include <stdio.h>
#include <stdlib.h>
#include <sys/sysinfo.h>

#include "glitcher.h"
#include "mem_glitcher.h"

mem_glitcher_t *mem_glitcher_new() {
  mem_glitcher_t *	glitcher;

  glitcher = malloc(sizeof(*glitcher));
  if (glitcher == NULL) {
    fprintf(stderr, "malloc: unable to allocate memory\n");
    return NULL;
  }

  struct sysinfo meminfo;
  int res = sysinfo(&meminfo);
  if (res != 0) {
    fprintf(stdout, "sysinfo: unable to get memory informatipn");
    return NULL;
  }

  glitcher->size = meminfo.totalram;

  fprintf(stdout, "size of memory: %ld\n", glitcher->size);
  
  glitcher->fh = fopen("/dev/mem", "r");
  if (glitcher->fh == NULL) {
    fprintf(stderr, "open: unable to open /dev/mem\n");
    free(glitcher);
    return NULL;
  }

  return glitcher;
}

int mem_glitch(mem_glitcher_t *glitcher, char *output) {
  size_t peek_at;

  peek_at = random() % (glitcher->size - GLITCH_AREA);
  if (fseek(glitcher->fh, peek_at, SEEK_SET) != 0) {
    fprintf(stderr, "fseek: unable to reach position %ld\n", peek_at);
    return -1;
  }

  if (fread(glitcher->buffer, 1, GLITCH_AREA, glitcher->fh) != GLITCH_AREA) {
    fprintf(stderr, "fseek: unable to read %ld bytes from /dev/mem\n", GLITCH_AREA);
    return -1;
  }

  if (buffer_to_jpeg(glitcher->buffer, GLITCH_AREA, output)) {
    fprintf(stderr, "buffer_to_jpeg: unable to create JPEG from glitch buffer\n");
    return -1;
  }

  return 0;
}

void mem_glitcher_free(mem_glitcher_t *glitcher) {
  fclose(glitcher->fh);
  free(glitcher);
}
