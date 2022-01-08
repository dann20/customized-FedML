#!/bin/bash

INFERENCE="../client_simulator/transformer_inference.py"
NUM_CLIENT=4
CFG_DIR="../Transformer-related/configs/relationship/to_be_train"
CFG_DIR_AFTER="../Transformer-related/configs/relationship/al_run"

run_cfg() {
	echo "START TESTING CONFIG $1 ...."
	for ID in $(seq 0 $(($NUM_CLIENT-1))); do
		echo "TESTING CLIENT $ID"
		python -Wignore $INFERENCE --config $1 -id $ID
	done
	if [[ $? -eq 0 ]]; then
		mv -v $1 $CFG_DIR_AFTER
	fi
}

for CFG in $CFG_DIR/*.json; do
	run_cfg $CFG
done
