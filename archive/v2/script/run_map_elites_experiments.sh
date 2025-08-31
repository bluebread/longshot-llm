#!/bin/bash

# MAP-Elites Experiments Script
# Runs map_elites_with_dataset.py with different configurations

# Fixed parameters
CELL_DENSITY=3
BATCH_SIZE=10

# Create output directories if they don't exist
mkdir -p ../output/archives
mkdir -p ../output/datasets

# Loop through num_vars from 3 to 6
for NUM_VARS in {3..6}; do
    echo "========================================="
    echo "Starting experiments for NUM_VARS=${NUM_VARS}"
    echo "========================================="
    
    # Loop through width from 2 to num_vars
    for WIDTH in $(seq 2 ${NUM_VARS}); do
        echo "----------------------------------------"
        echo "Configuration: num_vars=${NUM_VARS}, width=${WIDTH}"
        echo "----------------------------------------"
        
        # Calculate size and mutate-length
        SIZE=$(((2 ** WIDTH)))
        MUTATE_LENGTH=$((SIZE / 4))
        ITERATIONS=${SIZE}
        
        echo "Calculated parameters:"
        echo "  size=${SIZE}"
        echo "  mutate-length=${MUTATE_LENGTH}"
        echo "  iterations=${ITERATIONS}"
        
        # Clean up databases before each run
        echo "Cleaning up databases..."
        python cleanup_databases.py --force
        
        # Run with uniform strategy
        ARCHIVE_OUTPUT="../output/archives/me-n${NUM_VARS}w${WIDTH}-uniform.json"
        DATASET_OUTPUT="../output/datasets/traj-n${NUM_VARS}w${WIDTH}-uniform.json"
        
        echo ""
        echo "Running MAP-Elites with uniform strategy..."
        echo "Archive output: ${ARCHIVE_OUTPUT}"
        echo "Dataset output: ${DATASET_OUTPUT}"
        
        python map_elites_with_dataset.py \
            --num-vars ${NUM_VARS} \
            --width ${WIDTH} \
            --size ${SIZE} \
            --mutate-length ${MUTATE_LENGTH} \
            --iterations ${ITERATIONS} \
            --strategy uniform \
            --cell-density ${CELL_DENSITY} \
            --batch-size ${BATCH_SIZE} \
            --elites-only \
            --output ${ARCHIVE_OUTPUT} \
            --dataset-output ${DATASET_OUTPUT} \
            --init-strategy random
        
        # Run with performance strategy
        ARCHIVE_OUTPUT="../output/archives/me-n${NUM_VARS}w${WIDTH}-performance.json"
        DATASET_OUTPUT="../output/datasets/traj-n${NUM_VARS}w${WIDTH}-performance.json"
        
        echo ""
        echo "Running MAP-Elites with performance strategy..."
        echo "Archive output: ${ARCHIVE_OUTPUT}"
        echo "Dataset output: ${DATASET_OUTPUT}"
        
        python map_elites_with_dataset.py \
            --num-vars ${NUM_VARS} \
            --width ${WIDTH} \
            --size ${SIZE} \
            --mutate-length ${MUTATE_LENGTH} \
            --iterations ${ITERATIONS} \
            --strategy performance \
            --cell-density ${CELL_DENSITY} \
            --batch-size ${BATCH_SIZE} \
            --elites-only \
            --output ${ARCHIVE_OUTPUT} \
            --dataset-output ${DATASET_OUTPUT} \
            --init-strategy random
        
        echo ""
        echo "Completed experiments for num_vars=${NUM_VARS}, width=${WIDTH}"
        echo ""
    done
done

echo "========================================="
echo "All experiments completed!"
echo "========================================="
echo ""
echo "Results saved in:"
echo "  Archives: ../output/archives/"
echo "  Datasets: ../output/datasets/"