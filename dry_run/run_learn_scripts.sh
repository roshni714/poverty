#!/bin/bash
OUTPUT_DIR=$1
LEARN_DIR="$OUTPUT_DIR/learn"
USE_SBATCH=$2
mkdir -p $OUTPUT_DIR
python generate_sbatches.py --output-path $OUTPUT_DIR

for experiment in $LEARN_DIR/*.sh
do
    echo $experiment
    chmod u+x $experiment
    if [ "$USE_SBATCH" = true ]; then
        sbatch $experiment
    else
        chmod u+x $experiment
        $experiment
    fi
    sleep 1
done

echo "Done"
