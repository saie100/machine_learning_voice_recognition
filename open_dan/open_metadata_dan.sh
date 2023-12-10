#!/bin/bash

# Navigate to the corpus root directory
cd open_other_dan


# Create or clear the meta_data.csv file
echo "relative_path,classID" > ../metadata_open_dan.csv



# Find all .flac files and process their paths
find . -type f -name "*.wav" | while read -r file; do
    echo "open_dan/open_other_dan/${file:2},1" >> ../metadata_open_dan.csv
done
cd -

cd open_sesame_dan

find . -type f -name "*.wav" | while read -r file; do
    echo "open_dan/open_sesame_dan/${file:2},0" >> ../metadata_open_dan.csv
done

# Move back to the original directory
cd -
