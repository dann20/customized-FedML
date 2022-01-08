#!/bin/bash

for CFG in ../Transformer-related/configs/relationship/to_be_train/*.json; do
	echo "STARTING TRAINING CONFIG $CFG"

	echo "Restarting EMQX broker..."
	docker stop emqx && docker start emqx && sleep 5

	echo "Starting SERVER..."
	python transformer-app.py --num-client 4 --config $CFG > ./logs/server_log.txt 2>&1 &
	export SERVER_PID=$! && echo "SERVER_PID = $SERVER_PID"
	sleep 10

	echo "Starting CLIENT 0..."
	python ../client_simulator/transformer_client_simulator.py --client_uuid '0' > ./logs/mobile_client_log_0.txt 2>&1 &
	export CLIENT0_PID=$! && echo "CLIENT0_PID = $CLIENT0_PID"
	echo "Starting CLIENT 1..."
	python ../client_simulator/transformer_client_simulator.py --client_uuid '1' > ./logs/mobile_client_log_1.txt 2>&1 &
	export CLIENT1_PID=$! && echo "CLIENT1_PID = $CLIENT1_PID"
	echo "Starting CLIENT 2..."
	python ../client_simulator/transformer_client_simulator.py --client_uuid '2' > ./logs/mobile_client_log_2.txt 2>&1 &
	export CLIENT2_PID=$! && echo "CLIENT2_PID = $CLIENT2_PID"
	echo "Starting CLIENT 3..."
	python ../client_simulator/transformer_client_simulator.py --client_uuid '3' > ./logs/mobile_client_log_3.txt 2>&1 &
	export CLIENT3_PID=$! && echo "CLIENT3_PID = $CLIENT3_PID"

	while true; do
		ps cax | grep $SERVER_PID > /dev/null
		if [[ $? -ne 0 ]]; then
			echo "FINISHED training config $CFG"
			kill $CLIENT0_PID
			kill $CLIENT1_PID
			kill $CLIENT2_PID
			kill $CLIENT3_PID
			mv -v $CFG ../Transformer-related/configs/relationship/to_be_test/
			echo "--------------------------------------------------------------------"
			break
		fi
		sleep 1
	done
done
