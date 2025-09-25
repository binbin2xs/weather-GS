#!/bin/bash

BASE_CMD="python run_weather_recon.py --sh_degree 0"

START_YEAR=2008
END_YEAR=2015


declare -a GPU_MAP
GPU_MAP=(0 1 2 3 4 5 6 7)

for i in {0..7}; do
    CURRENT_YEAR=$((START_YEAR + i))
    GPU_ID=${GPU_MAP[$i]}
    
    START_DATE="${CURRENT_YEAR}-09-09"
    END_DATE="${CURRENT_YEAR}-12-31"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID $BASE_CMD --start_time $START_DATE --end_time $END_DATE &

    echo " $i: GPU $GPU_ID  $CURRENT_YEAR ($START_DATE  $END_DATE)"
done

wait
echo "All done"
