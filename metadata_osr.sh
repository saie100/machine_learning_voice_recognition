#!/bin/bash

# Navigate to the corpus root directory
cd other_voice_osr


# Create or clear the meta_data.csv file
echo "relative_path,classID" > ../metadata_osr.csv

# Declare an associative array to hold the unique classIDs
declare -A classID_map
next_classID=0

# Find all .flac files and process their paths
find . -type f -name "*.flac" | while read -r file; do
    # Extract the reader ID by cutting the directory path
    reader_id=$(echo "$file" | cut -d'/' -f2 | cut -d'-' -f1)
    
    # Check if the reader_id is already in the map
    if [[ ! ${classID_map[$reader_id]+_} ]]; then
        # If not, add it with the next available unique number
        classID_map[$reader_id]=$next_classID
        let next_classID++
    fi

    # Write the relative file path and mapped class ID to the meta_data.csv file
    if [[ ${classID_map[$reader_id]} -lt 3 ]]; then
        # Write the relative file path and mapped class ID to the meta_data.csv file
        # echo "other_voice_osr/${file:2},${classID_map[$reader_id]}" >> ../metadata_osr.csv
        echo "other_voice_osr/${file:2},0" >> ../metadata_osr.csv
    fi
    # echo "other_voice_osr/${file:2},0" >> ../metadata_osr.csv
    # echo "other_voice_osr/${file:2},${classID_map[$reader_id]}" >> ../metadata_osr.csv
done

cd -
cd target_voice
# Find all .flac files and process their paths
find . -type f -name "*.wav" | while read -r file; do
    file_name=$(echo "$file" | cut -d'/' -f2)
    echo "target_voice/${file_name},1" >> ../metadata_osr.csv
done

# Move back to the original directory
cd -
