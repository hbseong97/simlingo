#!/bin/bash
#SBATCH --job-name=extract_simlingo
#SBATCH --cpus-per-task=50
#SBATCH --mem=100G
#SBATCH --partition=h100
#SBATCH --output=logs/extract_simlingo_%j.out
#SBATCH --error=logs/extract_simlingo_%j.err


SOURCE_DATASET_DIR=/mnt/raid12/cache/huggingface/hub/datasets--RenzKa--simlingo/snapshots/796abab62acaddb6183d0d01c9b43a622c5c58c8/
TARGET_DATASET_DIR=database/simlingo

# Create target directory
mkdir -p $TARGET_DATASET_DIR

# Get the number of CPU cores for optimal parallelism
MAX_JOBS=$(nproc)

echo "Starting concurrent extraction with up to $MAX_JOBS parallel jobs..."

# Array to store background process IDs
pids=()

# Counter for active jobs
active_jobs=0

# Function to extract a single file
extract_file() {
    local file="$1"
    local filename=$(basename "$file")
    echo "Starting extraction of $filename..."
    tar -xzf "$file" -C "$TARGET_DATASET_DIR/"
    echo "Completed extraction of $filename"
}

# Export the function so it's available to subshells
export -f extract_file
export TARGET_DATASET_DIR

# Process each tar.gz file that contains "validation" in the name
for file in $SOURCE_DATASET_DIR/*training*.tar.gz; do
    # Wait if we've reached the maximum number of concurrent jobs
    while [ $active_jobs -ge $MAX_JOBS ]; do
        # Check if any background jobs have completed
        for i in "${!pids[@]}"; do
            if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                # Job completed, remove from array
                unset pids[$i]
                ((active_jobs--))
            fi
        done
        # Brief sleep to avoid busy waiting
        sleep 0.1
    done

    # Start extraction in background
    extract_file "$file" &
    pids+=($!)
    ((active_jobs++))
done

# Wait for all remaining background jobs to complete
echo "Waiting for all extractions to complete..."
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "All extractions completed successfully!"