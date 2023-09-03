/*
  This is merely for documentation, this is a failed attempt using
  CUDA. Memory of the graphic card is likely cleared at some level,
  need to deep dive.
*/

#include <stdio.h>
#include <stdlib.h>

#include "glitcher.h"
#include "cuda_glitcher.h"

/* Creates a new CUDA Glitcher */
cuda_glitcher_t *cuda_glitcher_new() {
  int		err;
  size_t	available_m, total_m;
  cuda_glitcher_t *	glitcher;

  /* Cuda init */
  cuInit(0);

  int device_count = 0;
  cuDeviceGetCount(&device_count);
  if (device_count == 0) {
    fprintf(stderr, "cuDeviceGetCount: no device supporting cuda.\n");
    return NULL;
  }
  
  glitcher = (cuda_glitcher_t *) malloc(sizeof(cuda_glitcher_t));
  if (glitcher == NULL) {
    fprintf(stderr, "malloc: unable to allocate memory for new glitcher\n");
    return NULL;
  }

  cuDeviceGet(&glitcher->cu_device, 0);
  cuCtxCreate(&glitcher->cu_context, 0, glitcher->cu_device);

  err = cuMemGetInfo(&available_m, &total_m);
  if (err != 0) {
    fprintf(stderr, "cudaMemGetInfo: unable to get memory info (%d)\n", err);
    free(glitcher);
    return NULL;
  }

  fprintf(stdout, "memory available: %ld / %ld\n", available_m, total_m);

  /*
    This is a bold assumption, we try to allocate most of the memory
    available in the graphic card. We don't know exactly how it
    allocates memory, it likely has a bucket approach or something,
    and this is what has the biggest impact on the final glitch. We'll
    likely need to implement different strategies here at some point.
  */

  glitcher->dev_buffer_size = (available_m * 95) / 100;
  fprintf(stdout, "trying to allocate %ld bytes on device\n", glitcher->dev_buffer_size);

  if (glitcher->dev_buffer_size < GLITCH_AREA) {
    fprintf(stderr, "not enough memory on device (%ld < %ld)\n", glitcher->dev_buffer_size, GLITCH_AREA);
    free(glitcher);
    return NULL;
  }

  err = cuMemAlloc(&glitcher->dev_buffer, glitcher->dev_buffer_size);
  if (err != 0) {
    fprintf(stderr, "cuMemAlloc: unable to allocate device memory (%d)\n", err);
    free(glitcher);
    return NULL;
  }

  return glitcher;
}

/* Releases a Glitcher */
void cuda_glitcher_free(cuda_glitcher_t *glitcher) {
  int err;

  err = cuMemFree(glitcher->dev_buffer);
  if (err != 0) {
    fprintf(stderr, "cuMemFree: unable to free device memory (%d)\n", err);
  }

  free(glitcher);
}

/* Creates a new glitch out from GFX memory */
int cuda_glitch(cuda_glitcher_t *glitcher, char *output) {
  int		err;
  size_t	peek_at;
  void *	buffer;

  peek_at = random() % (glitcher->dev_buffer_size - GLITCH_AREA);

  buffer = malloc(GLITCH_AREA);
  if (buffer == NULL) {
    fprintf(stderr, "malloc: unable to allocate memory\n");
    return -1;
  }

  fprintf(stdout, "peeking at offset %ld\n", peek_at);

  err = cuMemcpyDtoH(buffer, glitcher->dev_buffer + peek_at, GLITCH_AREA);
  if (err != 0) {
    fprintf(stderr, "cuMemcpy: unable to copy memory from device to host (%d)\n", err);
    return -1;
  }

  err = buffer_to_jpeg(buffer, GLITCH_AREA, output);
  if (err != 0) {
    fprintf(stderr, "buffer_to_jpeg: unable to convert glitch to jpeg\n");
    return -1;
  }

  free(buffer);

  return 0;
}
