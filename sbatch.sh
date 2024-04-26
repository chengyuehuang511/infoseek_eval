#!/bin/bash
sbatch job.sh --qos="short" --gpus-per-node="a40:1" --nodes=1 --partition=overcap --account=overcap