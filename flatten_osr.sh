#!/bin/bash

# The source directory containing the FLAC files
SOURCE_DIR="./all_voice_data/"

# The target directory where FLAC files will be copied
TARGET_DIR="./all_voice/"

# Check if TARGET_DIR exists, if not, create it
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
fi

# Find all .flac files in SOURCE_DIR and copy them to TARGET_DIR
find "$SOURCE_DIR" -type f -name *.wav -exec cp {} "$TARGET_DIR" \;

echo "All FLAC files have been copied to $TARGET_DIR"

# Find all .flac files in SOURCE_DIR and copy them to TARGET_DIR
find "$SOURCE_DIR" -type f -name *.wav -exec cp {} "$TARGET_DIR" \;

echo "All FLAC files have been copied to $TARGET_DIR"
