#!/bin/sh
cd ../client_simulator
nohup python ../client_simulator/VAE_client_simulator.py --client_uuid '0' > ./VAE_client_log_0.txt 2>&1 &
nohup python ../client_simulator/VAE_client_simulator.py --client_uuid '1' > ./VAE_client_log_1.txt 2>&1 &
nohup python ../client_simulator/VAE_client_simulator.py --client_uuid '2' > ./VAE_client_log_2.txt 2>&1 &
nohup python ../client_simulator/VAE_client_simulator.py --client_uuid '3' > ./VAE_client_log_3.txt 2>&1 &
