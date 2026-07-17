#!/bin/bash
set -e

STORAGE_DIR="/qdrant/storage"
SNAPSHOT_NAME="qdrant_snapshot.tar.gz"

echo "Checking if Qdrant storage is already populated..."
if [ "$(ls -A $STORAGE_DIR 2>/dev/null)" ]; then
    echo "Storage directory is not empty. Skipping restore process."
    exit 0
fi

echo "Storage is empty. Preparing to restore from MinIO..."

# Configure MinIO Client
echo "Configuring MinIO client..."
mc alias set myminio $MINIO_ENDPOINT $MINIO_ACCESS_KEY $MINIO_SECRET_KEY

# Download snapshot
echo "Downloading snapshot from MinIO bucket: $MINIO_BUCKET..."
mc cp myminio/$MINIO_BUCKET/$SNAPSHOT_NAME /tmp/$SNAPSHOT_NAME

# Verify checksum if provided
if [ -n "$QDRANT_SNAPSHOT_SHA256" ]; then
    echo "Verifying SHA256 checksum..."
    echo "$QDRANT_SNAPSHOT_SHA256  /tmp/$SNAPSHOT_NAME" > /tmp/checksum.txt
    if ! sha256sum -c /tmp/checksum.txt; then
        echo "Checksum verification failed!"
        exit 1
    fi
    echo "Checksum verified successfully."
else
    echo "No SHA256 checksum provided. Skipping verification."
fi

# Extract snapshot
echo "Extracting snapshot to $STORAGE_DIR..."
tar -xzf /tmp/$SNAPSHOT_NAME -C $STORAGE_DIR

echo "Restore complete! Qdrant is ready to start."
exit 0
