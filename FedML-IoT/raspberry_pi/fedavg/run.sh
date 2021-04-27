#!/bin/bash
python3 VAE_LSTM_fedavg_rpi_client.py --server_ip http://192.168.0.6:5000 --client_uuid '0' -ob scada1-16-1r-client0-test2.txt -or resmon-scada1-16-1r-client0-test2.csv
