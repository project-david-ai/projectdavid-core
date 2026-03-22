# ProjectDavid — Fine-Tuning Observability Handover
**Date:** 2026-03-20  
**Status:** **Core Orchestration Verified** — Moving to Real-time Telemetry & Metrics  
**Repo:** `projectdavid-core` (monorepo)  
**Stack:** FastAPI · MySQL · Redis · Samba · Jaeger · OpenTelemetry (OTEL)

---

## 1. Current Progress Recap
*   **Infrastructure:** Nginx Dynamic DNS fixed; Samba-direct data prep verified.
*   **Orchestration:** SDK -> API -> Redis -> Worker (Subprocess) handoff fully verified.
*   **Hardware Control:** `laptop` vs `standard` profiles integrated to manage VRAM.
*   **ML Engine:** `unsloth_train.py` implemented with real Torch/Unsloth logic (Build in progress).

---

## 2. Observability Goals
The objective is to move away from "Blind Training" where logs are only visible via `docker logs`. We are implementing a three-tier observability strategy:

1.  **Tier 1 (User Metrics):** Live Loss/Step/VRAM data inside the MySQL `training_jobs` table for SDK polling.
2.  **Tier 2 (System Tracing):** Lifecycle spans in **Jaeger** (Staging vs. Loading vs. Training).
3.  **Tier 3 (Hardware Telemetry):** GPU Temp/VRAM usage pushed to **OTEL Collector**.

---

## 3. Implementation Plan — Milestone: "The Heartbeat"

### Step 1: Metric Heartbeats (unsloth_train.py)
We inject a custom `DavidObservabilityCallback` into the Unsloth trainer. This callback intercepts training logs and prints structured JSON strings to `stdout`.

**Logic to inject:**
```python
# Custom callback to output metrics in a format the Worker can parse.
class DavidHeartbeatCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Inject hardware context
            if torch.cuda.is_available():
                logs["vram_used_gb"] = round(torch.cuda.memory_reserved() / 1e9, 2)
            # Tag the line for the Worker parser
            print(f"METRIC_HEARTBEAT:{json.dumps(logs)}", flush=True)
```

Step 2: Worker Log Parsing (worker.py)
The Worker is already reading stdout from the subprocess. We update the loop to detect the METRIC_HEARTBEAT tag, parse the JSON, and update the metrics column in MySQL.
Logic to update:
code
Python
for line in process.stdout:
    clean_line = line.strip()
    if clean_line.startswith("METRIC_HEARTBEAT:"):
        payload = json.loads(clean_line.replace("METRIC_HEARTBEAT:", ""))
        # Atomically update the metrics JSON in DB
        job.metrics = {**(job.metrics or {}), **payload}
        db.commit() 
    else:
        print(f"[{job_id}] {clean_line}", flush=True)
Step 3: Jaeger Lifecycle Tracing
We utilize the existing OTEL_EXPORTER_OTLP_ENDPOINT to wrap the Worker's activities in spans. This allows you to see a timeline in the Jaeger UI of how long the "Samba Stage" took versus the "Model Load."
Spans to implement:
worker.process_job (Root Span)
└─ worker.stage_data_samba
└─ worker.ml_subprocess_execution (Includes subprocess PID)
└─ worker.register_model
4. Hardware Safety Telemetry (Laptop Protection)
To prevent laptop thermal throttling or OS crashes during training:
pynvml Integration: The worker will monitor GPU temperature.
Emergency Stop: If GPU temperature exceeds 85°C, the worker will send a SIGTERM to the subprocess and mark the job as failed (Reason: Thermal Safety).
5. New Environment Variables
Variable	Scope	Purpose
OTEL_EXPORTER_OTLP_ENDPOINT	training-worker	Point to http://otel-collector:4317
METRIC_LOG_INTERVAL	training-worker	Frequency of DB metric updates (default: 1 step)
GPU_TEMP_LIMIT	training-worker	Safety threshold for laptop training (default: 85)
6. Progress Tracker
Milestone	Status	Notes
Core Orchestration	✅	Verified with Simulator
Real Unsloth Logic	🕒	Building (16GB WSL Fix active)
Laptop Profile	✅	Integrated in logic
Metric Heartbeats	❌	Next Priority
Jaeger Tracing	❌	Implementation pending
vLLM Hot-Reload	❌	Final production milestone
7. Action Items for the Developer
Rebuild Completion: Wait for the training-worker container to finish building with the new 16GB memory limit.
First ML Run: Execute the integration test with Qwen/Qwen2.5-1.5B-Instruct and laptop profile.
Observation: Monitor docker logs training_worker to ensure Unsloth successfully patches the kernels on your 4060.
Inject Telemetry: Once the first run succeeds, we will apply the DavidHeartbeatCallback logic to enable the live SDK dashboard.