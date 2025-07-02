#!/bin/bash

cd "$(dirname "$0")"

# Configuration
LOCAL_DIR="./out/"
REMOTE="myserver"
REMOTE_DIR="/var/www/html/presentation_BA/"

echo "Starting publish..."

# Use rsync to sync local out/ to remote directory
rsync -avz --progress --delete \
    -e ssh \
    "$LOCAL_DIR" \
    "$REMOTE:$REMOTE_DIR"

if [ $? -eq 0 ]; then
    echo "Publish completed successfully."
else
    echo "Error during publish." >&2
    exit 1
fi