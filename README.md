# customized-FedML
This is a fork of FedML framework with additional models

## How to use LockEdge and VAE-LSTM in this customized FedML framework

### 0. Prerequisites
#### Install and Configure EMQX broker
from repo [emqx](https://github.com/emqx/emqx)
  - If pre-built binary packages aren't available for your OS, build from source is recommended (requires erlang and some other packages)
##### Configuration:
  - In emqx.conf:
    - Set mqtt.max_packet_size = 20MB (> 2.5Mb for the largest, VAE and LSTM model config)
    - Set mqtt.retain_available = true
  - In plugins/emqx_retainer.conf:
    - Set retainer.max_payload_size = 10MB (> 2.5Mb for the largest, VAE and LSTM model config)
#### Install required python packages in anaconda on server (recommended) and Raspberry Pi's  machine env or virtual env (no conda due to unavailable tensorflow package)

### 1. Configuration
#### Clients and Server
- In FedML/fedml_iot/cfg.py: modify HOST variable to your server's IP address in your network (Assume both MQTT broker and server are running on 1 machine)
#### Clients
- In run.sh: 
  - Modify server_ip to server's IP address, 
  - Modify client_uuid from 0 for each client and 
  - Change run script for your model (VAE_LSTM_fedavg_rpi_client.py or lockedge_fedavg_rpi_client.py)
#### Server
- Modify model configuration json file (for VAE-LSTM) to your needs

### 2. Run
Instructions are written in chronological order of executions
#### On Server
- Run EMQ X broker:
  - `cd` to bin folder (if built from source)
  - `./emqx start` (built from source) or `emqx start` (binary package)
- Run Aggregator Server:
  - Activate your installed environment
  - VAE-LSTM Model: `python VAE-LSTM-app.py --config $(dir to model config file) --num-client $(number of workers)
  - LockEdge Model: `python LCHA-app.py --client-num-per-round $(number of workers) --comm-round $(number of comm rounds)`
 #### On clients:
 After server has run, execute ./run.sh
