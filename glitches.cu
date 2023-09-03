#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

/* Output file of the glitch */
#define GLITCH_WIDTH		1440
#define GLITCH_HEIGHT		900
#define GLITCH_AREA		(GLITCH_WIDTH * GLITCH_HEIGHT) * sizeof(uint32_t)

/* Main structure for the glitcher */
typedef struct glitcher_s {
  void *	dev_buffer;		/* memory to allocate on the GFX card to seek */
  size_t	dev_buffer_size;	/* size of the memory buffer */
  size_t	peek_at;		/* where to start looking at for our glitch */
} glitcher_t;

/* Creates a new Glitcher */
glitcher_t *new_glitcher() {
  int		err;
  size_t	available_m, total_m;
  glitcher_t *	glitcher;

  glitcher = (glitcher_t *) malloc(sizeof(glitcher_t));
  if (glitcher == NULL) {
    fprintf(stderr, "malloc: unable to allocate memory for new glitcher\n");
    return NULL;
  }

  /*
    This is a bold assumption, we try to allocate 2/3 of the memory
    available in the graphic card. We don't know exactly how it
    allocates memory, it likely has a bucket approach or something,
    and this is what has the biggest impact on the final glitch. We'll
    likely need to implement different strategies here at some point.
  */

  err = cudaMemGetInfo(&available_m, &total_m);
  if (err != 0) {
    fprintf(stderr, "cudaMemGetInfo: unable to get memory info (%d)\n", err);
    free(glitcher);
    return NULL;
  }

  fprintf(stdout, "memory available: %ld / %ld\n", available_m, total_m);

  glitcher->dev_buffer_size = (available_m / 2) * 3;
  fprintf(stdout, "trying to allocate %ld bytes on device\n", glitcher->dev_buffer_size);

  err = cudaMalloc(&glitcher->dev_buffer, glitcher->dev_buffer_size);
  if (err != 0) {
    fprintf(stderr, "cudaMalloc: unable to allocate device memory (%d)\n", err);
    free(glitcher);
    return NULL;
  }

  return glitcher;
}

/* Releases a Glitcher */
void free_glitcher(glitcher_t *glitcher) {
  int err;

  err = cudaFree(glitcher->dev_buffer);
  if (err != 0) {
    fprintf(stderr, "cudaFree: unable to free device memory (%d)\n", err);
  }
}

/* Creates a new glitch out from GFX memory */
int glitch(glitcher_t *glitcher, char *output) {
  return 0;
}

/* Entrypoint */
int main(int ac, char **av) {
  char *	output_file;
  glitcher_t *	glitcher;
  
  if (ac != 2) {
    fprintf(stderr, "usage: %s OUTPUT\n", av[0]);
    return -1;
  }

  output_file = av[1];

  glitcher = new_glitcher();
  if (glitcher == NULL) {
    fprintf(stderr, "unable to create glitcher\n");
    return -1;
  }

  /* Here we can maybe loop a hundred of times or so to see what we have */
  if (glitch(glitcher, output_file) != 0) {
    fprintf(stderr, "unable to glitch\n");
    free_glitcher(glitcher);
    return -1;
  }

  free_glitcher(glitcher);
  
  return 0;
}
