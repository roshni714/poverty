#!/bin/bash

rm -rf /home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/gd/scripts
mkdir /home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/gd/scripts
python generate_sbatches.py
rm -rf /home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/gd/results
mkdir /home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/gd/results


for experiment in /home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/gd/scripts/*.sh
do
    echo $experiment
    chmod u+x $experiment
#    sbatch $experiment
    $experiment
    sleep 1
done

echo "Done"
