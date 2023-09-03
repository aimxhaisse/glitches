/*
  We don't use the Cuda runtime API but instead rely on the Kernel
  API.  This is to be as low-level as possible and have the ability to
  get dirty memory.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <jpeglib.h>

#include "glitcher.h"
#include "mem_glitcher.h"

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

/* Entrypoint */
int main(int ac, char **av) {
  int			number;
  char *		output_dir;
  char *		output_file;
  mem_glitcher_t *	mem_glitcher;

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
  
  mem_glitcher = mem_glitcher_new();
  if (mem_glitcher == NULL) {
    fprintf(stderr, "unable to create memory glitcher\n");
    return -1;
  }

  srand(time(NULL));

  for (int i = 0; i < number; i++) {
    int size = (strlen(output_dir) + 1 + 16 + 4);
    output_file = (char *) malloc(size * sizeof(*output_file));
    snprintf(output_file, size, "%s/%d.jpeg", output_dir, i);

    fprintf(stdout, "%s...\n", output_file);

    if (mem_glitch(mem_glitcher, output_file) != 0) {
      fprintf(stderr, "unable to glitch\n");
      mem_glitcher_free(mem_glitcher);
      return -1;
    }

    fprintf(stdout, "OK!\n");

    free(output_file);
  }
    
  mem_glitcher_free(mem_glitcher);
  
  return 0;
}
