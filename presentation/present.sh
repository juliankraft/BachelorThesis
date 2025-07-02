#!/bin/bash

cd "$(dirname "$0")" 

# Variables
TARGET_FOLDER=./out/
SOURCE_IMAGES=./images

# Clear the target folder if it exists
if [ -d "$TARGET_FOLDER" ]; then
    rm -rf "$TARGET_FOLDER"
    echo "Cleared target folder: $TARGET_FOLDER"
else
    echo "Target folder does not exist. No need to clear."
fi

mkdir -p "$TARGET_FOLDER/images"

# Export the presentation
reveal-md presentation.md --css simple_jk.css --static "$TARGET_FOLDER"
if [ $? -eq 0 ]; then
    echo "Presentation exported successfully to $TARGET_FOLDER!"
else
    echo "Error: Failed to export presentation." >&2
    exit 1
fi

# Copy all images to the static folder
if [ -d "$SOURCE_IMAGES" ]; then
    cp -r "$SOURCE_IMAGES/"* "$TARGET_FOLDER/images/"
    echo "All images copied to $TARGET_FOLDER/images/"
else
    echo "Error: Source image directory ($SOURCE_IMAGES) not found." >&2
    exit 1
fi

echo "Script completed successfully!"
