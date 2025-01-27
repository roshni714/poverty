#!/bin/bash
# Hyperparameter search
python main_hparam.py main --config hparam/configs/gt_continuous_rate.yaml
sleep 1
python main_hparam.py main --config hparam/configs/gt_binary_rate.yaml
sleep 1
python main_hparam.py main --config hparam/configs/gt_binary_gap.yaml
sleep 1
python main_hparam.py main --config hparam/configs/gt_continuous_gap.yaml
sleep 1
# Learning
python main_learn.py main --config hparam/results/output_gt_continuous_rate.yaml --trainpath data/train.parquet --testpath data/test.parquet --device cuda
sleep 1
python main_learn.py main --config hparam/results/output_gt_binary_rate.yaml --trainpath data/train.parquet --testpath data/test.parquet --device cuda
sleep 1
python main_learn.py main --config hparam/results/output_gt_binary_gap.yaml --trainpath data/train.parquet --testpath data/test.parquet --device cuda
sleep 1
python main_learn.py main --config hparam/results/output_gt_continuous_gap.yaml --trainpath data/train.parquet --testpath data/test.parquet --device cuda
sleep 1
