#!/bin/bash

# Loop through .wav files with the pattern '* *.wav'
for file in *.wav; do
    # Extract the number before the space and the last digit of the 4-digit number
    number=${BASH_REMATCH[1]}
    last_digit=${BASH_REMATCH[2]: -1}
    
    # Form the new file name
    new_file="target${number}_${last_digit}.wav"
    
    # Rename the file
    mv "$file" "$new_file"
    
    echo "Renamed $file to $new_file"
done