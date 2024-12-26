#!/bin/bash

rm -rf /home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/dry_run/scripts
mkdir /home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/dry_run/scripts
rm -rf /home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/dry_run/pickled
mkdir /home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/dry_run/pickled
python generate_sbatches.py


for experiment in /home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/dry_run/scripts/*.sh
do
    echo $experiment
    chmod u+x $experiment
    sbatch $experiment
    sleep 1
done

echo "Done"
