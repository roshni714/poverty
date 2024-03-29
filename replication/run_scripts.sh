#!/bin/bash

rm -rf /zfs/gsb/intermediate-yens/rsahoo/poverty/replication/scripts
mkdir /zfs/gsb/intermediate-yens/rsahoo/poverty/replication/scripts
python generate_sbatches.py
rm -rf /zfs/gsb/intermediate-yens/rsahoo/poverty/replication/results
mkdir /zfs/gsb/intermediate-yens/rsahoo/poverty/replication/results


for experiment in /zfs/gsb/intermediate-yens/rsahoo/poverty/replication/scripts/*.sh
do
    echo $experiment
    chmod u+x $experiment
    sbatch $experiment
#    $experiment
    sleep 1
done

echo "Done"
