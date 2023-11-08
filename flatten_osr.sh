#!/bin/bash

# The source directory containing the FLAC files
SOURCE_DIR="./osr/"

# The target directory where FLAC files will be copied
TARGET_DIR="./other_voice_osr/"

# Check if TARGET_DIR exists, if not, create it
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
fi

# Find all .flac files in SOURCE_DIR and copy them to TARGET_DIR
find "$SOURCE_DIR" -type f -name '*.flac' -exec cp {} "$TARGET_DIR" \;

echo "All FLAC files have been copied to $TARGET_DIR"
