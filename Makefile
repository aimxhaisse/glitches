COMPILER 	:= nvcc
SOURCE		:= glitches.cu
OUTPUT		:= glitches


all: help

.PHONY: help
help:	## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: build
build:  ## Build the Glitches explorer
	$(COMPILER) $(SOURCE) -o $(OUTPUT)

.PHONY: clean
clean: 	## Cleans the last glitches build
	rm -f $(OUTPUT)

.PHONY: re
re: clean build	## Rebuilds everything
