#!/bin/bash
if [ "$1" == "" ] ; then mode="debug"; else mode=$1; fi
# set SAVE_ROOT / DATA_PATH before running, e.g.:
#   export SAVE_ROOT=/path/to/surrogate/cache
#   export DATA_PATH=/path/to/surrogate/data/<dataset>
python ./main.py --mode $mode \
				 --exp_name "Testing" \
				 --run_name "trainer_update" \
				 --run_id "" \
				 --sweep_id "" \
				 --save_root "${SAVE_ROOT}" \
				 --data_path "${DATA_PATH}" \
				 --specify-data "" \
				 --epochs 5000 \
				 --batch_size 2000 \
				 --learning_rate 0.001 \
				 --device "cuda" \


