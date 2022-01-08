#!/bin/bash

for CFG in ../Transformer-related/configs/relationship/to_be_train/*json; do
	echo "START TESTING CONFIG $CFG ...."
	for ID in {0..3}; do
		echo "TESTING CLIENT $ID"
		python -Wignore ../client_simulator/transformer_inference.py --config $CFG -id $ID
	done
	if [[ $? -eq 0 ]]; then
		mv -v $CFG ../Transformer-related/configs/relationship/al_run/
	fi
done
