import ast
import asyncio
import os
import re
import shutil
import tempfile
import textwrap
import time
import traceback

from dotenv import load_dotenv
from fastapi import WebSocket
from projectdavid import Entity
from sandbox.services.logging_service import LoggingUtility

load_dotenv()


class StreamingCodeExecutionHandler:
    def __init__(self):
        self.logging_utility = LoggingUtility()
        self.active_processes = {}

        # ───────────────────────── DIRS ──────────────────────────
        self.root_dir = os.getcwd()
        self.generated_files_dir = os.path.abspath(
            os.path.join(self.root_dir, "generated_files")
        )
        self.mpl_cache_dir = os.path.abspath(os.path.join(self.root_dir, "mpl_cache"))

        os.makedirs(self.generated_files_dir, exist_ok=True)
        os.makedirs(self.mpl_cache_dir, exist_ok=True)

        os.environ["MPLCONFIGDIR"] = self.mpl_cache_dir
        self.last_executed_script_path = None

        # ───────────────────────── FIREJAIL ──────────────────────
        disable_firejail = os.getenv("DISABLE_FIREJAIL", "false").lower() == "true"
        self.security_profile = {
            "firejail_args": (
                []
                if os.name == "nt" or disable_firejail
                else [
                    f"--private={self.generated_files_dir}",
                    "--noprofile",
                    "--nogroups",
                    "--nosound",
                    "--notv",
                    "--seccomp",
                    "--caps.drop=all",
                    "--net",
                    "--env=PYTHONPATH",
                ]
            )
        }

        # Standard imports provided to every script
        self.required_imports = (
            "import asyncio\n"
            "import math\n"
            "import time\n"
            "import os\n"
            "from datetime import datetime\n"
        )

    def normalize_code(self, code: str) -> str:
        """
        Cleans and normalizes code.
        REMOVED: aggressive manual indentation rebuilding (caused syntax errors).
        ADDED: Markdown stripping and smart-quote fixing.
        """
        # 1. Strip Markdown code blocks
        code = re.sub(r"^```python\s*", "", code, flags=re.MULTILINE | re.IGNORECASE)
        code = re.sub(r"^```\s*", "", code, flags=re.MULTILINE)
        code = re.sub(r"```$", "", code, flags=re.MULTILINE)

        # 2. Fix Smart Quotes and Math symbols
        replacements = {
            "\u201c": '"',
            "\u201d": '"',
            "\u2018": "'",
            "\u2019": "'",
            "\u00b2": "**2",
            "^": "**",
            "\u21d2": "->",
            "\u00d7": "*",
            "\u00f7": "/",
        }
        for k, v in replacements.items():
            code = code.replace(k, v)

        # 3. Remove non-printable characters (except newlines/tabs)
        code = re.sub(r"[^\x00-\x7F]+", "", code)

        # 4. Dedent (Fixes indentation if the LLM tabbed the whole block)
        code = textwrap.dedent(code).strip()

        # 5. Check if it's a single expression (e.g., "1+1") and wrap in print
        if "\n" not in code and not code.startswith("print"):
            # Simple heuristic: if it looks like a function call or math
            if re.match(r"^[\w\.]+\s*\(.*\)\s*$", code) or re.search(r"[+\-*/]", code):
                try:
                    ast.parse(code)  # Check if valid
                    code = f"print({code})"
                except SyntaxError:
                    pass

        return self.required_imports + "\n" + code

    async def execute_code_streaming(
        self, websocket: WebSocket, code: str, user_id: str = "test_user"
    ) -> None:
        tmp_path = None
        try:
            full_code = self.normalize_code(code)

            # Syntax Check before execution
            try:
                ast.parse(full_code)
            except SyntaxError as e:
                error_msg = traceback.format_exception_only(type(e), e)
                # Send the raw error so the LLM knows exactly which line failed
                await websocket.send_json(
                    {"error": f"Syntax Error during validation:\n{''.join(error_msg)}"}
                )
                return

            if not self._validate_code_security(full_code):
                await websocket.send_json(
                    {
                        "error": "Security violation: Forbidden import or command detected."
                    }
                )
                return

            execution_id = f"{user_id}-{int(time.time() * 1000)}"

            # Create temp file in the generated_files directory
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, dir=self.generated_files_dir
            ) as tmp:
                tmp.write(full_code)
                tmp_path = tmp.name

            self.last_executed_script_path = tmp_path

            cmd = (
                ["python3", "-u", tmp_path]
                if (os.name == "nt" or os.getenv("DISABLE_FIREJAIL") == "true")
                else [
                    "firejail",
                    *self.security_profile["firejail_args"],
                    "python3",
                    "-u",
                    tmp_path,
                ]
            )

            # Execution
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.generated_files_dir,  # Ensures files save to the right place
            )

            self.active_processes[execution_id] = proc
            await self._stream_process_output(proc, websocket, execution_id)

            # Upload files
            uploaded_files = await self._upload_generated_files(user_id=user_id)

            await websocket.send_json(
                {
                    "status": "complete",
                    "execution_id": execution_id,
                    "uploaded_files": uploaded_files,
                }
            )
        except Exception as e:
            self.logging_utility.error("Execution failed: %s", str(e))
            try:
                await websocket.send_json({"error": f"System Error: {str(e)}"})
            except Exception:
                pass
        finally:
            # Cleanup
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            self.last_executed_script_path = None
            try:
                await websocket.close()
            except Exception:
                pass

    async def _stream_process_output(
        self, proc, websocket: WebSocket, execution_id: str
    ) -> None:
        try:
            while True:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=30)
                if not line:
                    break
                # Stream logs back to client
                await websocket.send_text(line.decode(errors="replace"))

            await websocket.send_json(
                {
                    "status": "process_complete",
                    "exit_code": await proc.wait(),
                    "execution_id": execution_id,
                }
            )
        except asyncio.TimeoutError:
            proc.terminate()
            await websocket.send_json({"error": "Execution timed out (30s limit)."})
        except Exception as e:
            self.logging_utility.error("Stream error: %s", str(e))
            proc.terminate()
        finally:
            self.active_processes.pop(execution_id, None)

    def _validate_code_security(self, code: str) -> bool:
        # Regex to catch malicious intent
        blocked = [
            r"(__import__|exec|eval|input)\s*\(",
            r"(subprocess|os|sys)\.(system|popen|run|spawn|exec)",
            r"shutil\.(rmtree|move)",  # Prevent deleting system files
            r"asyncio\.(create_subprocess|run_in_executor)",
            r"import\s+(sys|ctypes)",  # Allow os, but block sys/ctypes
        ]
        return not any(re.search(p, code) for p in blocked)

    async def _upload_generated_files(self, user_id: str) -> list:
        uploaded_files = []
        try:
            # Re-initialize client to ensure fresh connection/auth if needed
            client = Entity(
                api_key=os.getenv("ADMIN_API_KEY"),
                base_url="http://fastapi_cosmic_catalyst:9000",
            )
        except Exception as e:
            self.logging_utility.error("Failed to init ProjectDavid client: %s", e)
            return []

        async def upload_single_file(filename, file_path):
            if os.path.isdir(file_path):
                return None
            try:
                # Assuming 'assistants' is the correct bucket/purpose
                upload = await asyncio.to_thread(
                    client.files.upload_file, file_path=file_path, purpose="assistants"
                )

                # Get signed URL immediately
                signed_url = await asyncio.to_thread(
                    client.files.get_signed_url,
                    file_id=upload.id,
                    use_real_filename=True,
                )

                return {
                    "filename": filename,
                    "id": upload.id,
                    "url": signed_url,
                }
            except Exception as e:
                self.logging_utility.error("Upload failed for %s: %s", filename, str(e))
                return None

        tasks = []
        if os.path.exists(self.generated_files_dir):
            for fname in os.listdir(self.generated_files_dir):
                fpath = os.path.join(self.generated_files_dir, fname)

                # Skip the running script
                if self.last_executed_script_path and os.path.abspath(
                    fpath
                ) == os.path.abspath(self.last_executed_script_path):
                    continue

                if os.path.isfile(fpath):
                    tasks.append(upload_single_file(fname, fpath))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, dict) and res:
                    uploaded_files.append(res)

        # Aggressive Cleanup of generated files after upload
        if os.path.exists(self.generated_files_dir):
            for fname in os.listdir(self.generated_files_dir):
                fpath = os.path.join(self.generated_files_dir, fname)
                try:
                    if os.path.isfile(fpath) or os.path.islink(fpath):
                        os.remove(fpath)
                    elif os.path.isdir(fpath):
                        shutil.rmtree(fpath)
                except Exception:
                    pass

        return uploaded_files
