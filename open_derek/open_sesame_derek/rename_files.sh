#!/bin/bash
rm *.asd
# Loop through .wav files with the pattern '* *.wav'
count=1
for file in *.wav; do
    # Extract the number before the space and the last digit of the 4-digit number
    
    # Form the new file name
    new_file="open_sesame_${count}.wav"
    
    # Rename the file
    mv "$file" "$new_file"
    
    echo "Renamed $file to $new_file"
    ((count++))
done