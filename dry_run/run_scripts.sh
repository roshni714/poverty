#!/bin/bash

rm -rf /home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/dry_run/scripts3
mkdir /home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/dry_run/scripts3
python generate_sbatches.py


for experiment in /home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/dry_run/scripts3/*.sh
do
    echo $experiment
    chmod u+x $experiment
    sbatch $experiment
    #$experiment
    sleep 1
done

echo "Done"
