# ProjectDavid — Sovereign Forge v2.0 Deployment Guide

This guide covers the "Sovereign" way to deploy intelligence. By utilizing the Distributed Mesh, you manage physical GPU hardware via a centralized Database Ledger.

### 1. Download the Target Model
Before the Mesh can serve a model, the weights must exist in the local hardware cache. Use the HuggingFace CLI inside the vLLM container to pull the weights.

```bash
# Recommendation: Use the 4-bit optimized versions for laptop GPUs
docker exec -it vllm_server huggingface-cli download unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit
````


### 2. Update the Catalog (SQL)**

````sql
/* Replace <PASSWORD> with your MYSQL_PASSWORD from .env */
docker exec -it my_mysql_cosmic_catalyst mysql -u api_user -p<PASSWORD> entities_db -e "INSERT IGNORE INTO base_models (id, name, family, parameter_count, is_multimodal, created_at) VALUES ('unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit', 'Qwen 2.5 1.5B (4-bit)', 'qwen', '1.5B', 0, UNIX_TIMESTAMP());"
````


### 3. The Mesh requires an "Inventory Check." You must register the model in the global catalog before the scheduler will allow it to be assigned to a node.


* Replace <PASSWORD> with your MYSQL_PASSWORD from .env 

````sql
docker exec -it my_mysql_cosmic_catalyst mysql -u api_user -p<PASSWORD> entities_db -e "INSERT IGNORE INTO base_models (id, name, family, parameter_count, is_multimodal, created_at) VALUES ('unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit', 'Qwen 2.5 1.5B (4-bit)', 'qwen', '1.5B', 0, UNIX_TIMESTAMP());"

````


### 3. Activate via the ProjectDavid SDK**

Trigger the Smart Scheduler. This chooses the healthiest GPU node, creates a deployment ticket, and locks the required VRAM in the cluster ledger.


   
> ⚠️ Perform this step from the host machine connected to a GPU that hosts  instance of the training-worker and  vllm 
> docker containers that you just downloaded the model to in step 2.


````python
import os
from dotenv import load_dotenv
from projectdavid import Entity

load_dotenv()

client = Entity(
    base_url=os.getenv("PROJECT_DAVID_PLATFORM_BASE_URL"),
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"),
)

# This assigns the model to the best available physical node
activate = client.models.activate_base("unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit")
print(f"🚀 Deployment Ticket Created for Node: {activate['node']}")
````
This will assign the instance of vllm hosted on this machine
as a node in the cluster, serving inference for the model.


### 4. Monitor the Edge Agent (Worker)** 

The training-worker on your local machine acts as a Hypervisor. It will detect the new ticket and physically spawn the vLLM container on your GPU.


**Expected Worker Logs:**

````bash
training_worker  | 2026-03-21 22:23:08,785 - INFO - /app/src/api/training/worker.py:336 - ✅ Node rtx4060_laptop_main joined the David Mesh.
training_worker  | 2026-03-21 22:23:08,786 - INFO - /usr/local/lib/python3.11/threading.py:982 - 💓 Heartbeat started for node: rtx4060_laptop_main
training_worker  | 2026-03-21 22:23:08,786 - INFO - /usr/local/lib/python3.11/threading.py:982 - 👀 Inference Supervisor active for node: rtx4060_laptop_main
training_worker  | 2026-03-21 22:23:08,786 - INFO - /app/src/api/training/worker.py:336 - 👷 Cluster Worker rtx4060_laptop_main listening for jobs...
training_worker  | 2026-03-21 22:23:48,795 - WARNING - /usr/local/lib/python3.11/threading.py:982 - 🚨 Deployment drift! Syncing pd_vllm_dep_1SCTdAFI5bHEtmVeMEU7UF
training_worker  | 2026-03-21 22:23:48,797 - INFO - /app/src/api/training/worker.py:192 - 🚢 Spawning vLLM: pd_vllm_dep_1SCTdAFI5bHEtmVeMEU7UF for model unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit
````
At this stage the node GPU supervisor is actively spawning an instance of the vLLM container
with the target model loaded.

**Check vLLM Progress:**

````bash
# Note: Container name is generated from the Deployment ID
docker logs -f pd_vllm_dep_1SCT...
````

**Expected Output:**


````bash
APIServer pid=1) INFO 03-21 22:25:05 [launcher.py:47] Route: /v1/completions/render, Methods: POST
(APIServer pid=1) INFO 03-21 22:25:05 [launcher.py:47] Route: /v1/messages, Methods: POST
(APIServer pid=1) INFO 03-21 22:25:05 [launcher.py:47] Route: /v1/messages/count_tokens, Methods: POST
(APIServer pid=1) INFO 03-21 22:25:05 [launcher.py:47] Route: /inference/v1/generate, Methods: POST
(APIServer pid=1) INFO 03-21 22:25:05 [launcher.py:47] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=1) INFO 03-21 22:25:05 [launcher.py:47] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
````

**Congratulations!** You have successfully registered your model with Project David.**

---


### Troubleshooting & "Big Hammer" Commands

If the Mesh hits a snag, use these verified commands to restore order.

**A. The Ledger Reset (SQL)**

If you hit an IntegrityError or the scheduler thinks your GPU is full when it isn't, clear the "Order Book" and the VRAM locks.


````bash

docker exec -it my_mysql_cosmic_catalyst mysql -u api_user -p<PASSWORD> entities_db -e "DELETE FROM inference_deployments; DELETE FROM gpu_allocations; UPDATE fine_tuned_models SET is_active = 0;"

````

**B. Nginx DNS Refresh (502 Bad Gateway)**


````bash
docker compose restart nginx
````

**D. GPU Capability Check**

Verify that the Docker engine can physically "see" the NVIDIA driver from a fresh container.

````bash
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.0.1-base-ubuntu22.04 nvidia-smi
````

**E. Cache Inspection**

See exactly which models are taking up space in your local "Sovereign Forge."

````bash
docker exec -it vllm_server huggingface-cli scan-cache
````

**F. Clear the InferenceDeployment Table**

You have broken node entries in the database.

````sql
docker exec -it my_mysql_cosmic_catalyst mysql -u api_user -<PASSWORD>  entities_db -e "DELETE FROM inference_deployments; DELETE FROM gpu_allocations;"
````

**G. Remove stale vLLM Containers**

````bash
docker ps -a | findstr "pd_vllm" | ForEach-Object { docker rm -f ($_.Split(' ')[0]) }  
````

