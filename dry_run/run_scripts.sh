#!/bin/bash

rm -rf /home/users/rsahoo/zfs/projects/faculty/swager-poverty/poverty/dry_run/scripts5
mkdir /home/users/rsahoo/zfs/projects/faculty/swager-poverty/poverty/dry_run/scripts5
python generate_sbatches.py


for experiment in /home/users/rsahoo/zfs/projects/faculty/swager-poverty/poverty/dry_run/scripts5/*.sh
do
    echo $experiment
    chmod u+x $experiment
    sbatch $experiment
    #$experiment
    sleep 1
done

echo "Done"
