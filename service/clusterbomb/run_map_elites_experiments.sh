#!/bin/bash

# MAP-Elites Experiments Script for Clusterbomb Service
# Collects trajectories using MAP-Elites algorithm with specified parameter ranges
#
# Parameter Settings:
# - cell density = 3
# - iterations = 3 * width^2
# - batch size = 32
# - num-steps = 2 * (2^width)
# - init strategy = random (for uniform), warehouse (for performance)
# - enable sync = false
# - all other arguments use defaults

# Fixed parameters as specified
CELL_DENSITY=3
BATCH_SIZE=32
# Note: --enable-sync is not passed (defaults to false)

# Check if run_map_elites.py exists in the current directory
if [ ! -f "run_map_elites.py" ]; then
    echo "Error: run_map_elites.py not found in current directory."
    echo "Please run this script from the service/clusterbomb/ directory."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p output

# Function to calculate iterations based on width
calculate_iterations() {
    local width=$1
    echo $((3 * width * width))
}

# Function to calculate num-steps based on width
calculate_num_steps() {
    local width=$1
    # Calculate 2^width using a loop for maximum compatibility
    local power=1
    local i=0
    while [ $i -lt $width ]; do
        power=$((power * 2))
        i=$((i + 1))
    done
    echo $((2 * power))
}

echo "========================================="
echo "Clusterbomb MAP-Elites Trajectory Collection"
echo "========================================="
echo ""
echo "Configuration Settings:"
echo "  - Cell Density: ${CELL_DENSITY}"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - Synchronization: Disabled"
echo "  - Iterations: 3 * width^2"
echo "  - Num Steps: 2 * (2^width)"
echo ""

# Loop through num_vars from 3 to 6
for NUM_VARS in $(seq 8 8); do
    echo "========================================="
    echo "Starting experiments for NUM_VARS=${NUM_VARS}"
    echo "========================================="

    # Loop through width from 2 to num_vars
    for WIDTH in $(seq 7 ${NUM_VARS}); do
        echo "----------------------------------------"
        echo "Configuration: num_vars=${NUM_VARS}, width=${WIDTH}"
        echo "----------------------------------------"

        # Calculate dynamic parameters
        ITERATIONS=$(calculate_iterations ${WIDTH})
        NUM_STEPS=$(calculate_num_steps ${WIDTH})

        echo "Calculated parameters:"
        echo "  iterations=${ITERATIONS}"
        echo "  num-steps=${NUM_STEPS}"
        echo ""

        # Run with uniform strategy (init-strategy=random)
        ARCHIVE_OUTPUT="output/archive-n${NUM_VARS}w${WIDTH}-uniform.json"

        echo "Running MAP-Elites with uniform strategy..."
        echo "  Init strategy: random"
        echo "  Archive output: ${ARCHIVE_OUTPUT}"
        echo ""
        # echo "Command to execute:"
        # echo "python run_map_elites.py \\"
        # echo "    --num-vars ${NUM_VARS} \\"
        # echo "    --width ${WIDTH} \\"
        # echo "    --iterations ${ITERATIONS} \\"
        # echo "    --cell-density ${CELL_DENSITY} \\"
        # echo "    --batch-size ${BATCH_SIZE} \\"
        # echo "    --num-steps ${NUM_STEPS} \\"
        # echo "    --strategy uniform \\"
        # echo "    --init-strategy random \\"
        # echo "    --output ${ARCHIVE_OUTPUT}"
        # echo ""

        # Note: Not executing, just showing the command
        # Uncomment the following to actually run:
        python run_map_elites.py \
            --num-vars ${NUM_VARS} \
            --width ${WIDTH} \
            --iterations ${ITERATIONS} \
            --cell-density ${CELL_DENSITY} \
            --batch-size ${BATCH_SIZE} \
            --num-steps ${NUM_STEPS} \
            --strategy uniform \
            --init-strategy random \
            --timeout 1200 \
            --output ${ARCHIVE_OUTPUT}

        echo "Uniform strategy would complete here."
        echo ""

        # Run with performance strategy (init-strategy=warehouse)
        ARCHIVE_OUTPUT="output/archive-n${NUM_VARS}w${WIDTH}-performance.json"

        echo "Running MAP-Elites with performance strategy..."
        echo "  Init strategy: warehouse"
        echo "  Archive output: ${ARCHIVE_OUTPUT}"
        echo ""
        # echo "Command to execute:"
        # echo "python run_map_elites.py \\"
        # echo "    --num-vars ${NUM_VARS} \\"
        # echo "    --width ${WIDTH} \\"
        # echo "    --iterations ${ITERATIONS} \\"
        # echo "    --cell-density ${CELL_DENSITY} \\"
        # echo "    --batch-size ${BATCH_SIZE} \\"
        # echo "    --num-steps ${NUM_STEPS} \\"
        # echo "    --strategy performance \\"
        # echo "    --init-strategy warehouse \\"
        # echo "    --output ${ARCHIVE_OUTPUT}"
        # echo ""

        # Note: Not executing, just showing the command
        # Uncomment the following to actually run:
        python run_map_elites.py \
            --num-vars ${NUM_VARS} \
            --width ${WIDTH} \
            --iterations ${ITERATIONS} \
            --cell-density ${CELL_DENSITY} \
            --batch-size ${BATCH_SIZE} \
            --num-steps ${NUM_STEPS} \
            --strategy performance \
            --init-strategy warehouse \
            --timeout 1200 \
            --output ${ARCHIVE_OUTPUT}

        # echo "Performance strategy would complete here."
        # echo ""
        echo "Completed experiments for num_vars=${NUM_VARS}, width=${WIDTH}"
        echo ""
    done
done

echo "========================================="
echo "All experiment commands generated!"
echo "========================================="
echo ""
echo "To actually run the experiments, uncomment the python execution lines"
echo "in this script (lines marked with # python run_map_elites.py ...)"
echo ""
echo "Archives will be saved in: ./output/"
echo ""
echo "Note: This script currently shows the commands but does not execute them."
echo "This allows for review before running the actual experiments."