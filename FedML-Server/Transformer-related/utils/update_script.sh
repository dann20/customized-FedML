#!/bin/bash

CFG_DIR='../configs/relationship/al_run'
SCRIPT='update_results.py'

for CFG in $CFG_DIR/*.json; do
	echo "UPDATING RESULTS OF CONFIG $CFG ...."
	python $SCRIPT --config $CFG -a
done
