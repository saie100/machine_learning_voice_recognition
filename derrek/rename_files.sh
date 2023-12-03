#!/bin/bash

# Set the directory to search for .wav files
DIRECTORY="./"

# Set the output CSV file name
OUTPUT_CSV="derrek_metadata.csv"

# Find .wav files and write to CSV
find "$DIRECTORY" -type f -name "*.wav" > "$OUTPUT_CSV"
