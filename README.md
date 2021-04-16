# customized-FedML
This is a fork of FedML framework with additional models

## How to use LockEdge and VAE-LSTM in this customized FedML framework

### 0. Prerequisites
#### Install and Configure EMQX broker
from repo [emqx](https://github.com/emqx/emqx)
  - If pre-built binary packages aren't available for your OS, build from source is recommended (requires erlang and some other packages)
##### Configuration:
  - In emqx.conf:
    - Set `mqtt.max_packet_size = 20MB`
    - Set `mqtt.retain_available = true`
  - In plugins/emqx_retainer.conf:
    - Set `retainer.max_payload_size = 20MB`
#### Install required python packages
Install in anaconda on server (recommended) and Raspberry Pi's machine env or virtual env (no conda due to unavailable tensorflow package on _armv7l_ repo)

### 1. Configuration
#### Clients and Server
- In FedML/fedml_iot/cfg.py:
  - Modify HOST variable to your MQTT broker's IP address in your network
  - Modify APP_HOST variable to your server's IP address (Weight Aggregator)
- Make sure all clients and server are on the same network
#### Clients
- In run.sh: 
  - Modify server_ip to server's IP address, 
  - Modify client_uuid from 0 for each client
  - Change run script for your model (VAE_LSTM_fedavg_rpi_client.py or lockedge_fedavg_rpi_client.py)
#### Server
- Modify model configuration json file (for VAE-LSTM) to your needs

### 2. Run
Instructions are written in _chronological_ order of executions
#### On Server
- Run EMQ X broker:
  - `cd` to bin folder (if built from source)
  - `./emqx start` (built from source) or `emqx start` (binary package)
- Run Aggregator Server:
  - Activate your installed environment
  - VAE-LSTM Model:  
      `python VAE-LSTM-app.py --config $(dir to model config file) --num-client $(number of workers)`
  - LockEdge Model:  
      `python LCHA-app.py --client-num-per-round $(number of workers) --comm-round $(number of comm rounds)`
 #### On clients:
 After server has run, execute `./run.sh`

### 3. Restart 
Before re-run scripts on server and clients, restart EMQ X broker by `./emqx restart` or `emqx restart`
