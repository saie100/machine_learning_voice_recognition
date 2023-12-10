#!/bin/bash

# Navigate to the corpus root directory
cd open_other_derek


# Create or clear the meta_data.csv file
echo "relative_path,classID" > ../metadata_open_derek.csv



# Find all .flac files and process their paths
find . -type f -name "*.wav" | while read -r file; do
    echo "open_derek/open_other_derek/${file:2},3" >> ../metadata_open_derek.csv
done

cd -
cd open_sesame_derek

find . -type f -name "*.wav" | while read -r file; do
    echo "open_derek/open_sesame_derek/${file:2},2" >> ../metadata_open_derek.csv
done

# Move back to the original directory
cd -
