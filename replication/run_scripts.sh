#!/bin/bash

rm -rf /home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/replication/scripts
mkdir /home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/replication/scripts
python generate_sbatches.py
rm -rf /home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/replication/results
mkdir /home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/replication/results


for experiment in /home/users/rsahoo/zfs/gsb/intermediate-yens/rsahoo/poverty/replication/scripts/*.sh
do
    echo $experiment
    chmod u+x $experiment
    sbatch $experiment
#    $experiment
    sleep 1
done

echo "Done"
