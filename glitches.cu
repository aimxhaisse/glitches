#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include <jpeglib.h>

/* Output file of the glitch */
#define GLITCH_WIDTH		1440
#define GLITCH_HEIGHT		900
#define GLITCH_AREA		(GLITCH_WIDTH * GLITCH_HEIGHT) * sizeof(uint32_t)
#define MAX_GENERATIONS		4096

/* Main structure for the glitcher */
typedef struct glitcher_s {
  void *	dev_buffer;		/* memory to allocate on the GFX card to seek */
  size_t	dev_buffer_size;	/* size of the memory buffer */
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

  err = cudaMemGetInfo(&available_m, &total_m);
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

/* Converts a buffer to a JPEG */
int buffer_to_jpeg(void *buffer, size_t buffer_size, char *output) {
  struct jpeg_compress_struct	cinfo;
  struct jpeg_error_mgr		jerr;
  FILE *			outfile;
  JSAMPROW			lines[1];

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  if ((outfile = fopen(output, "wb")) == NULL) {
    fprintf(stderr, "fopen: can't open %s\n", output);
    return -1;
  }
  jpeg_stdio_dest(&cinfo, outfile);

  cinfo.image_width = GLITCH_WIDTH;
  cinfo.image_height = GLITCH_HEIGHT;
  cinfo.input_components = 4;
  cinfo.in_color_space = JCS_EXT_RGBX;
  
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, 100, TRUE);

  jpeg_start_compress(&cinfo, TRUE);

  unsigned int sum = 0;

  for (int i = 0; i < GLITCH_HEIGHT; i++) {
    lines[0] = (JSAMPROW) ((char *) buffer + i * GLITCH_WIDTH);
    (void) jpeg_write_scanlines(&cinfo, lines, 1);
    for (int j = 0; j < GLITCH_WIDTH; j++) {
      sum += *((char *) buffer + i * GLITCH_WIDTH + j);
    }
  }

  fprintf(stdout, "checksum: %d\n", sum);

  jpeg_finish_compress(&cinfo);
  fclose(outfile);
  jpeg_destroy_compress(&cinfo);
  
  return 0;
}

/* Creates a new glitch out from GFX memory */
int glitch(glitcher_t *glitcher, char *output) {
  int		err;
  size_t	peek_at;
  void *	buffer;

  peek_at = random() % (glitcher->dev_buffer_size - GLITCH_AREA);

  buffer = malloc(GLITCH_AREA);
  if (buffer == NULL) {
    fprintf(stderr, "malloc: unable to allocate memory\n");
    return -1;
  }

  err = cudaMemcpy(buffer, (void *) (((char *) glitcher->dev_buffer) + peek_at), GLITCH_AREA, cudaMemcpyDeviceToHost);
  if (err != 0) {
    fprintf(stderr, "cudaMemcpy: unable to copy memory from device to host (%d)\n", err);
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

/* Entrypoint */
int main(int ac, char **av) {
  int		number;
  char *	output_dir;
  char *	output_file;
  glitcher_t *	glitcher;

  if (ac != 3) {
    fprintf(stderr, "usage: %s OUTPUT_DIR NUMBER\n", av[0]);
    return -1;
  }

  output_dir = av[1];
  number = atoi(av[2]);

  if (number <= 0) {
    fprintf(stderr, "number should be in [0, %d] (%d)\n", MAX_GENERATIONS, number);
    return -1;
  }

  glitcher = new_glitcher();
  if (glitcher == NULL) {
    fprintf(stderr, "unable to create glitcher\n");
    return -1;
  }

  srand(time(NULL));

  for (int i = 0; i < number; i++) {
    int size = (strlen(output_dir) + 1 + 16 + 4);
    output_file = (char *) malloc(size * sizeof(*output_file));
    snprintf(output_file, size, "%s/%d.jpeg", output_dir, i);

    fprintf(stdout, "%s...\n", output_file);

    if (glitch(glitcher, output_file) != 0) {
      fprintf(stderr, "unable to glitch\n");
      free_glitcher(glitcher);
      return -1;
    }

    fprintf(stdout, "OK!\n");

    free(output_file);
  }
    
  free_glitcher(glitcher);
  
  return 0;
}
