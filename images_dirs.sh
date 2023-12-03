#!/bin/bash

# Set the root directory where the .flac files are located
CORPUS_ROOT="other_voice_osr"

# Set the directory where the image directories will be created
IMAGE_ROOT="images_osr/other_voice_osr"

# Create the images directory if it doesn't exist
mkdir -p "${IMAGE_ROOT}"

# Navigate to the corpus root directory
cd "${CORPUS_ROOT}"

# Find all .flac files and create corresponding directories in the images folder
find . -type f -name "*.flac" | while read -r file; do
    # Extract the directory path without the file name and leading ./
    dir_path=$(dirname "${file}" | cut -c 3-)
    # Create the corresponding directory structure in the images directory
    mkdir -p "../${IMAGE_ROOT}/${dir_path}"
done

# Move back to the original directory
cd -
