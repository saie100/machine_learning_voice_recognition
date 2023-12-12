#!/bin/bash

# Navigate to the corpus root directory
cd other_voice_osr

# Create or clear the meta_data.csv file
echo "relative_path,classID" > ../metadata_osr.csv

# Declare an associative array to hold the unique classIDs
declare -A classID_map
next_classID=0

# Find all .flac files and process their paths
while read -r file; do
    # Extract the reader ID by cutting the directory path
    reader_id=$(echo "$file" | cut -d'/' -f2 | cut -d'-' -f1)
    
    # Check if the reader_id is already in the map
    if [[ ! ${classID_map[$reader_id]+_} ]]; then
        # If not, add it with the next available unique number
        classID_map[$reader_id]=$next_classID
        let next_classID++
    fi

    # Write the relative file path and mapped class ID to the meta_data.csv file
    echo "other_voice_osr/${file:2},${classID_map[$reader_id]}" >> ../metadata_osr.csv
done < <(find . -type f -name "*.flac")

cd -

# Move to the target_voice directory
cd target_voice

# Use next_classID directly
# Find all .wav files and process their paths
while read -r file; do
    file_name=$(echo "$file" | cut -d'/' -f2)
    echo "target_voice/${file_name},${next_classID}" >> ../metadata_osr.csv
done < <(find . -type f -name "*.wav")

cd -
