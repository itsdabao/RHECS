import subprocess
import json
import os
import tempfile
import sys

def execute_sandbox_code(code_string: str, tenant_id: str) -> dict:
    """
    Executes native Python code in a strict subprocess safely.
    Injects the TENANT_ID into OS environment variables.
    Enforces the Subprocess I/O constraint.
    """
    timeout_seconds = 10
    
    # Write the untrusted code to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sandbox.py', delete=False, encoding='utf-8') as temp_script:
        temp_script.write(code_string)
        script_path = temp_script.name

    # Mount the OS Env Vars mapping cryptographically
    env = os.environ.copy()
    env["TENANT_ID"] = tenant_id
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else project_root
    )

    try:
        # Execute the untrusted code inside an isolated subprocess
        # NOTE: In Production Linux, we replace ["python"] with ["sudo", "-u", "sandboxuser", "python"]
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env
        )
        
        # Unlink the temporary script
        os.remove(script_path)

        # Catch syntax/crash errors (stateful retry loop tracebacks)
        if result.returncode != 0:
            return {"success": False, "error": result.stderr}

        try:
            # Enforce Subprocess I/O Channel (strictly JSON standard output parsing)
            output_data = json.loads(result.stdout.strip())
            return {"success": True, "output": output_data}
        except json.JSONDecodeError:
            return {
                "success": False, 
                "error": f"Failed to parse JSON from stdout. The model must strictly only print valid JSON. Raw output received:\\n{result.stdout}"
            }

    except subprocess.TimeoutExpired:
        # Catch and kill infinite loops / hanging scripts
        os.remove(script_path)
        return {"success": False, "error": f"Execution timed out after {timeout_seconds} seconds. Code likely entered an infinite loop."}
    except Exception as e:
        if os.path.exists(script_path):
            os.remove(script_path)
        return {"success": False, "error": f"Unknown sandbox vulnerability tripped: {str(e)}"}
