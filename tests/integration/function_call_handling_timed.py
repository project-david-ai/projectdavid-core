import json
import os
import time
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from projectdavid import Entity

# ------------------------------------------------------------------
# 0. Setup & Profiling Tools
# ------------------------------------------------------------------
load_dotenv()

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREY = "\033[90m"
RESET = "\033[0m"


@dataclass
class TimingMetric:
    name: str
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0

    def stop(self):
        self.end = time.perf_counter()
        self.duration = self.end - self.start
        return self.duration


class Profiler:
    metrics: List[TimingMetric] = []

    @classmethod
    def start(cls, name: str) -> TimingMetric:
        m = TimingMetric(name=name, start=time.perf_counter())
        cls.metrics.append(m)
        return m

    @classmethod
    def print_report(cls):
        print(f"\n{CYAN}=== ⏱️ LIFECYCLE TIMING REPORT ==={RESET}")
        print(f"{'PHASE':<30} | {'DURATION (s)':<15}")
        print("-" * 50)
        total = 0.0
        for m in cls.metrics:
            # Highlight bottlenecks (> 2.0s) in Red
            color = RED if m.duration > 2.0 else GREEN
            print(f"{m.name:<30} | {color}{m.duration:.4f}s{RESET}")
            total += m.duration
        print("-" * 50)
        print(f"{'TOTAL DURATION':<30} | {total:.4f}s")


client = Entity(
    base_url=os.getenv("BASE_URL", "http://localhost:9000"),
    api_key=os.getenv("ENTITIES_API_KEY"),
)

USER_ID = os.getenv("ENTITIES_USER_ID")
ASSISTANT_ID = "asst_D0AVwJAFwYxZLFlFKQyb4S"
MODEL_ID = "together-ai/deepseek-ai/DeepSeek-V3"
PROVIDER_KW = "Hyperbolic"
HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY")


# ------------------------------------------------------------------
# 1. Tool Executor
# ------------------------------------------------------------------
def get_flight_times(tool_name: str, arguments) -> str:
    if tool_name != "get_flight_times":
        return json.dumps({"status": "error", "message": "unknown tool"})

    # Simulate slight processing time
    # time.sleep(0.1)

    print(
        f"\n{YELLOW}[LOCAL EXEC] Tool invoked: {tool_name} | Args: {arguments}{RESET}"
    )
    return json.dumps(
        {
            "status": "success",
            "duration": "4h 30m",
            "departure_time": "10:00 AM PST",
            "arrival_time": "06:30 PM EST",
        }
    )


# ------------------------------------------------------------------
# 2. Setup Phase
# ------------------------------------------------------------------
t_setup = Profiler.start("Setup (Thread/Run)")
print(f"{GREY}[1/4] Creating Thread & Message...{RESET}")
thread = client.threads.create_thread()
message = client.messages.create_message(
    thread_id=thread.id,
    role="user",
    content="Please fetch me the flight times between LAX and JFK.",
    assistant_id=ASSISTANT_ID,
)
print(f"{GREY}[2/4] Creating Run...{RESET}")
run = client.runs.create_run(assistant_id=ASSISTANT_ID, thread_id=thread.id)
t_setup.stop()

# ------------------------------------------------------------------
# 3. Stream 1 (LLM -> Tool Call)
# ------------------------------------------------------------------
stream = client.synchronous_inference_stream
stream.setup(
    thread_id=thread.id,
    assistant_id=ASSISTANT_ID,
    message_id=message.id,
    run_id=run.id,
    api_key=HYPERBOLIC_API_KEY,
)

print(f"\n{CYAN}[▶] STREAM 1: Initial Generation{RESET}")

t_ttft = Profiler.start("Stream 1: TTFT")  # Time to First Token
t_stream1 = Profiler.start("Stream 1: Total Generation")
first_token_received = False

# We use the iterator manually to verify precise timings
iterator = stream.stream_chunks(
    model=MODEL_ID, suppress_fc=False, timeout_per_chunk=10.0
)

for chunk in iterator:
    if not first_token_received:
        t_ttft.stop()
        first_token_received = True

    c_type = chunk.get("type", "unknown")

    # Optional: Print fewer logs to see timing clearly
    # print(f"{GREY}{c_type:<15} | {str(chunk)[:60]}...{RESET}")

t_stream1.stop()

# ------------------------------------------------------------------
# 4. Tool Execution Transition
# ------------------------------------------------------------------
t_poll = Profiler.start("Transition: Stream End -> Tool")
print(f"\n{GREY}[3/4] Polling for Tool Execution...{RESET}")

handled = client.runs.poll_and_execute_action(
    run_id=run.id,
    thread_id=thread.id,
    assistant_id=ASSISTANT_ID,
    tool_executor=get_flight_times,
    actions_client=client.actions,
    messages_client=client.messages,
    timeout=20,  # Increased timeout to catch server delay
    interval=0.1,
)
t_poll.stop()

# ------------------------------------------------------------------
# 5. Stream 2 (Tool Output -> Final Answer)
# ------------------------------------------------------------------
if handled:
    t_stream2 = Profiler.start("Stream 2: Final Response")
    print(f"\n{CYAN}[▶] STREAM 2: Final Response{RESET}")

    stream.setup(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID,
        message_id=message.id,
        run_id=run.id,
        api_key=HYPERBOLIC_API_KEY,
    )

    for chunk in stream.stream_chunks(model=MODEL_ID, timeout_per_chunk=180.0):
        pass  # Consume stream

    t_stream2.stop()
    print(f"\n{GREY}--- End of Stream ---{RESET}")
else:
    print(f"\n{RED}[!] No function call detected.{RESET}")

Profiler.print_report()
