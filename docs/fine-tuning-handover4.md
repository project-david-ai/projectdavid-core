# ProjectDavid — Distributed Cluster & Resource Awareness Handover
**Date:** 2026-03-20  
**Status:** **v1.0 (Local Loop) Verified** — Moving to v2.0 (Distributed Cluster)  
**Repo:** `projectdavid-core`  
**Vision:** Decoupled GPU Compute Nodes with Intelligent Load Balancing  

---

## 1. Architectural Shift: The "Factory" Pattern
We are evolving from a single-node deployment into a distributed "Factory" model. In this new paradigm:
*   **The Orchestrator (API/Redis):** Acts as the "Control Tower," managing state and metadata.
*   **The Compute Nodes:** Dedicated hardware instances running `training-worker` or `vllm`.
*   **Resource Awareness:** The system no longer "blindly" enqueues jobs; it queries the cluster state to find the most capable and available GPU.

---

## 2. Database Schema Enhancements (The Resource Registry)
To support a cluster, the database must become "Hardware-Aware." We are introducing two new tables to track physical assets and real-time allocations.

### A. `compute_nodes`
Tracks the physical or virtual machines joined to the ProjectDavid cluster.
*   `id`: Unique identifier (e.g., `node_rtx4060_laptop_01`).
*   `hostname` / `ip_address`: Connectivity details for internal routing.
*   `gpu_model`: Hardware string (e.g., `NVIDIA RTX 4060`).
*   `total_vram_gb`: Hard limit of the node's memory.
*   `current_vram_usage_gb`: Live telemetry from `nvidia-smi`.
*   `status`: `active`, `offline`, or `maintenance`.
*   `last_heartbeat`: Unix timestamp of the last check-in.

### B. `gpu_allocations`
A mapping table that tracks exactly what is occupying VRAM on a specific node.
*   `node_id`: FK to `compute_nodes`.
*   `job_id`: FK to `training_jobs` (if the GPU is training).
*   `model_id`: FK to `fine_tuned_models` (if the GPU is serving via vLLM).
*   `vram_reserved_gb`: How much memory this specific instance is utilizing.

---

## 3. The Distributed Lifecycle

### Phase 1: Node Registration (Heartbeat)
Each Worker or Inference container, upon startup, will:
1.  Query local hardware via `pynvml` or `nvidia-smi`.
2.  Register itself in the `compute_nodes` table.
3.  Initiate a background **Heartbeat Thread** that updates its `current_vram_usage_gb` and `last_heartbeat` in MySQL every 15–30 seconds.

### Phase 2: Intelligent Scheduling (API)
When a user submits a training job:
1.  The `Training API` queries the `compute_nodes` table.
2.  **Filter:** Find nodes with `status == active` and `(total_vram - usage) > required_vram`.
3.  **Target:** The Redis payload is enhanced: `{"job_id": "...", "target_node": "node_01"}`.
4.  **Handoff:** Only the worker identified as `node_01` will pick up the job from Redis.

### Phase 3: vLLM Multi-LoRA Scaling
Instead of one vLLM instance per model, the system will favor **LoRA Multiplexing**:
1.  vLLM starts on a "Heavy" node with the base model backbone.
2.  The API can instruct that specific vLLM node to load *multiple* LoRA adapters dynamically.
3.  The Registry tracks which Node is currently "hosting" which fine-tuned adapter.

---

## 4. Technical Requirements & Implementation Steps

### Step 1: SQL Migration
Apply the SafedDL pattern to create `compute_nodes` and `gpu_allocations`. Update `TrainingJob` and `FineTunedModel` to include an optional `node_id` column.

### Step 2: Telemetry Integration (`worker.py`)
Integrate `pynvml` into the worker. The worker must now identify its "Node ID" (derived from hostname or environment variable) and perform the heartbeat update.

### Step 3: Scheduler Logic (`training_service.py`)
Refactor `create_training_job` to include a "Node Selector." If no nodes are available, the job stays in `queued` status until a heartbeat reports free VRAM.

### Step 4: CLI Evolution (`docker_manager.py`)
Update the CLI to support multi-node commands.
*   `platform-api --node gpu-server-01 --mode up --services training-worker`
*   The CLI will use the `node_id` to ensure environment variables and volume mounts are specific to that hardware.

---

## 5. Stress Test Guardrails
1.  **VRAM Fragmentation:** The scheduler must account for the fact that 2GB + 2GB free on two different nodes cannot run a 4GB job.
2.  **Stale Nodes:** A background daemon must mark nodes as `offline` if `last_heartbeat > 60s` to prevent jobs from being assigned to "ghost" workers.
3.  **Samba Bandwidth:** In a cluster, multiple nodes will hit the Samba hub simultaneously. We must monitor IO wait times on the Samba container.

---

## 6. Progress Tracker

| Milestone | Status | Notes |
|---|---|---|
| Single-Node Loop | ✅ | 100% Verified (SDK to vLLM) |
| Multi-Tenant Auth | ✅ | Verified via JWT |
| Infrastructure-Direct Prep | ✅ | Verified via Samba |
| **Compute Node Registry** | ❌ | **Next Priority** |
| **Worker Heartbeat** | ❌ | Implementation pending |
| **Resource-Aware Scheduler** | ❌ | Implementation pending |

---

## 7. Next Immediate Action
**Implement the `ComputeNode` and `GPUAllocation` models in `src/api/training/models/models.py`.** This will provide the foundation for the Heartbeat service.