#!/bin/bash

# Navigate to the corpus root directory
cd train_other_voice_osr


# Create or clear the meta_data.csv file
echo "relative_path,classID" > ../metadata_train_500.csv



# Find all .flac files and process their paths
find . -type f -name "*.flac" | while read -r file; do
    echo "train_other_voice_osr/${file:2},0" >> ../metadata_train_500.csv
done
cd -

# cd open_sesame_dan

# find . -type f -name "*.wav" | while read -r file; do
#     echo "open_dan/open_sesame_dan/${file:2},0" >> ../metadata_train_500.csv
# done

# # Move back to the original directory
# cd -
