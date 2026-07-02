import os
import subprocess
import time
import requests

def run_eval(injection_flag, outfile):
    print(f"Running eval with USE_HARDCODED_INJECTION={injection_flag}")
    
    # Kill existing uvicorn processes
    subprocess.run("pkill -f uvicorn", shell=True)
    
    env = os.environ.copy()
    env["USE_HARDCODED_INJECTION"] = injection_flag
    
    # Start server
    server = subprocess.Popen(
        ["python3", "-m", "uvicorn", "app.main:app", "--port", "8000"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait for server to start
    for _ in range(30):
        try:
            r = requests.get("http://localhost:8000/health")
            if r.status_code == 200:
                break
        except:
            pass
        time.sleep(1)
        
    print("Server is up. Running harness...")
    # Run eval harness
    result = subprocess.run(
        ["python3", "scripts/eval_harness.py", "--url", "http://localhost:8000"],
        capture_output=True,
        text=True,
        env=env
    )
    
    with open(outfile, "w") as f:
        f.write(result.stdout)
        
    print(f"Finished. Saved to {outfile}")
    
    # Kill server
    server.terminate()
    server.wait()

run_eval("false", "eval_injection_off.txt")
run_eval("true", "eval_injection_on.txt")
