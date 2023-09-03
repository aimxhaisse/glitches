#ifndef CUDA_GLITCHER_H
#define CUDA_GLITCHER_H

/*
 * TODO: Figure out a way to have non-cleared memory from the graphic
 * card. Likely implies admin mode or something.
 */

#include <stdint.h>
#include <cuda.h>

/* Main structure for the CUDA glitcher */
typedef struct cuda_glitcher_s {
  CUdeviceptr 	dev_buffer;		/* memory to allocate on the GFX card to seek */
  size_t	dev_buffer_size;	/* size of the memory buffer */
  CUdevice	cu_device;		/* CUDA device */
  CUcontext	cu_context;		/* CUDA context */
} cuda_glitcher_t;

cuda_glitcher_t *cuda_glitcher_new();
int cuda_glitch(cuda_glitcher_t *glitcher, char *output);
void cuda_glitcher_free(cuda_glitcher_t *glitcher);

#endif /* CUDA_GLITCHER_H */
