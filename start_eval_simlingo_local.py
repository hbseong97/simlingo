# %%
import os
import subprocess
import time
import ujson
import shutil
import threading
import queue
from tqdm.autonotebook import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# %%
def get_num_running_jobs(job_processes):
    """Count currently running jobs"""
    running_count = sum(1 for proc in job_processes.values() if proc.poll() is None)
    try:
        with open('max_num_jobs.txt', 'r', encoding='utf-8') as f:
            max_num_parallel_jobs = int(f.read())
    except:
        max_num_parallel_jobs = 5  # Default to 5 parallel jobs
    
    return running_count, max_num_parallel_jobs

# %%
def create_run_script_bench2drive(job, port, tm_port):
    """Create a shell script to run the evaluation locally"""
    cfg = job["cfg"]
    route = job["route"]
    route_id = job["route_id"]
    seed = job["seed"]
    viz_path = job["viz_path"]
    result_file = job["result_file"]
    log_file = job["log_file"]
    err_file = job["err_file"]
    job_file = job["job_file"]

    with open(job_file, 'w', encoding='utf-8') as rsh:
        rsh.write(f'''#!/bin/bash

echo "Starting job for route {route_id} with seed {seed}"
echo "Process ID: $$"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust path as needed
conda activate simlingo_upgrade_py310  # TODO: change to your conda env
conda env list

cd {cfg["repo_root"]}

# Set environment variables
export CARLA_ROOT={cfg["carla_root"]}
export PYTHONPATH=$PYTHONPATH:{cfg["carla_root"]}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:{cfg["carla_root"]}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:{cfg["repo_root"]}/Bench2Drive/leaderboard
export PYTHONPATH=$PYTHONPATH:{cfg["repo_root"]}/Bench2Drive/scenario_runner
export SCENARIO_RUNNER_ROOT={cfg["repo_root"]}/Bench2Drive/scenario_runner

export SAVE_PATH={viz_path}

# Run the evaluation
python -u {cfg["repo_root"]}/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py --routes={route} \\
--repetitions=1 \\
--track=SENSORS \\
--checkpoint={result_file} \\
--timeout=600 \\
--agent={cfg["agent_file"]} \\
--agent-config={cfg["checkpoint"]} \\
--traffic-manager-seed={seed} \\
--port={port} \\
--traffic-manager-port={tm_port}

echo "Job completed for route {route_id}"
''')
    
    # Make the script executable
    os.chmod(job_file, 0o755)

# %%
def run_job_local(job, port, tm_port):
    """Run a single job locally using subprocess"""
    try:
        # Create the run script
        if job["cfg"]["benchmark"].lower() == "bench2drive":
            create_run_script_bench2drive(job, port, tm_port)
        else:
            raise NotImplementedError(f"Benchmark {job['cfg']['benchmark']} not implemented.")
        
        # Clean up visualization path
        if os.path.exists(job["viz_path"]):
            shutil.rmtree(job["viz_path"])
        os.makedirs(job["viz_path"], exist_ok=True)
        
        # Run the job script
        with open(job["log_file"], 'w') as log_f, open(job["err_file"], 'w') as err_f:
            process = subprocess.Popen(
                ['bash', job["job_file"]],
                stdout=log_f,
                stderr=err_f,
                cwd=job["cfg"]["repo_root"]
            )
        
        return process
    
    except Exception as e:
        print(f"Error starting job {job['route_id']}: {e}")
        return None

# %%
def filter_completed(jobs, job_processes):
    """Filter out completed jobs and jobs that need resubmission"""
    filtered_jobs = []
    
    for job in jobs:
        job_id = job.get("job_id")
        
        # If job is running, keep it in list
        if job_id and job_id in job_processes:
            process = job_processes[job_id]
            if process.poll() is None:  # Still running
                filtered_jobs.append(job)
                continue
            else:
                # Process finished, remove from tracking
                del job_processes[job_id]
        
        # Check if job completed successfully
        result_file = job["result_file"]
        if os.path.exists(result_file):
            try:
                with open(result_file, "r") as f:
                    evaluation_data = ujson.load(f)
                
                progress = evaluation_data['_checkpoint']['progress']
                
                need_to_resubmit = False
                if len(progress) < 2 or progress[0] < progress[1]:
                    need_to_resubmit = True
                else:
                    for record in evaluation_data['_checkpoint']['records']:
                        if record['status'] in [
                            'Failed - Agent couldn\'t be set up',
                            'Failed',
                            'Failed - Simulation crashed',
                            'Failed - Agent crashed'
                        ]:
                            need_to_resubmit = True
                            break
                
                if need_to_resubmit and job["tries"] > 0:
                    filtered_jobs.append(job)
                    
            except Exception as e:
                print(f"Error reading result file {result_file}: {e}")
                if job["tries"] > 0:
                    filtered_jobs.append(job)
        
        # Results file doesn't exist
        elif job["tries"] > 0:
            filtered_jobs.append(job)
    
    return filtered_jobs

# %%
def kill_problematic_jobs(jobs, job_processes):
    """Kill jobs that have encountered known issues"""
    for job in jobs:
        job_id = job.get("job_id")
        
        if not job_id or job_id not in job_processes:
            continue
            
        process = job_processes[job_id]
        if process.poll() is not None:  # Already finished
            continue
            
        log_file = job["log_file"]
        if not os.path.exists(log_file):
            continue
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) == 0:
                continue
            
            # Check for known error patterns
            problematic_patterns = [
                "Watchdog exception",
                "Engine crash handling finished; re-raising signal 11 for the default handler. Good bye.",
                "[91mStopping the route, the agent has crashed:"
            ]
            
            if any(pattern in line for line in lines for pattern in problematic_patterns):
                print(f"Killing problematic job {job_id} for route {job['route_id']}")
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
                del job_processes[job_id]
                
        except Exception as e:
            print(f"Error checking log file {log_file}: {e}")

# Configuration
configs = [
    {
        "agent": "simlingo",
        "checkpoint": "/home/khemoo/simlingo/models/01-08-01/checkpoints/epoch=000-step=000500.ckpt/pytorch_model.bin",
        "benchmark": "bench2drive",
        "route_path": "/home/khemoo/simlingo/simlingo/leaderboard/data/bench2drive_split",
        "seeds": [1], # TODO: change depending on how many eval seeds you want to run
        "tries": 2,
        "out_root": "/home/khemoo/simlingo/simlingo/eval_results/Bench2Drive",
        "carla_root": "/home/khemoo/simlingo/carla0915",
        "repo_root": "/home/khemoo/simlingo/simlingo",
        "agent_file": "/home/khemoo/simlingo/simlingo/team_code/agent_simlingo.py",
        "team_code": "team_code",
        "agent_config": "not_used",
        "username": "haebin"
    }
] # TODO: change to your paths and model, you can add multiple configs here

# %%
# Build job queue
job_queue = []
for cfg_idx, cfg in enumerate(configs):
    route_path = cfg["route_path"]
    routes = [x for x in os.listdir(route_path) if x[-4:]==".xml"]
    
    if cfg["benchmark"] == "bench2drive":
        fill_zeros = 3
    else: 
        fill_zeros = 2
    
    for seed in cfg["seeds"]:
        seed = str(seed)
        
        base_dir = os.path.join(cfg["out_root"], cfg["agent"], cfg["benchmark"], seed)
        os.makedirs(os.path.join(base_dir, "run"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "res"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "out"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "err"), exist_ok=True)
        
        for route in routes:
            route_id = route.split("_")[-1][:-4].zfill(fill_zeros)
            route = os.path.join(route_path, route)
            
            viz_path = os.path.join(base_dir, "viz", route_id)
            os.makedirs(viz_path, exist_ok=True)
            
            result_file = os.path.join(base_dir, "res", f"{route_id}_res.json")
            log_file = os.path.join(base_dir, "out", f"{route_id}_out.log")
            err_file = os.path.join(base_dir, "err", f"{route_id}_err.log")
            job_file = os.path.join(base_dir, "run", f'eval_{route_id}.sh')
            
            job = {
                "cfg": cfg,
                "route": route,
                "route_id": route_id,
                "seed": seed,
                "viz_path": viz_path,
                "result_file": result_file,
                "log_file": log_file,
                "err_file": err_file,
                "job_file": job_file,
                "tries": cfg["tries"]
            }
            
            job_queue.append(job)

# %%
# Port management
carla_world_ports = set(range(10000, 20000, 50))
carla_streaming_ports = set(range(20000, 30000, 50))
carla_tm_ports = set(range(30000, 40000, 50))

# %%
# Main execution loop
print(f"{len(job_queue)=}")
print(f"{job_queue[0]=}")
if len(job_queue) > 1:
    print(f"{job_queue[1]=}")

jobs = len(job_queue)
progress = tqdm(total=jobs)
job_processes = {}  # Track running processes
job_counter = 0  # For unique job IDs

while job_queue:
    kill_problematic_jobs(job_queue, job_processes)
    job_queue = filter_completed(job_queue, job_processes)
    
    progress.update(jobs - len(job_queue) - progress.n)
    
    # Get current status
    running_jobs, max_num_parallel_jobs = get_num_running_jobs(job_processes)
    
    # Track used ports
    used_ports = set()
    for job in job_queue:
        if job.get("job_id") in job_processes:
            used_ports.update(job.get("ports", set()))
    
    print(f"{running_jobs}/{max_num_parallel_jobs} jobs are running...")
    
    if running_jobs >= max_num_parallel_jobs:
        time.sleep(5)
        continue
    
    # Submit new jobs
    for job in job_queue:
        if job["tries"] <= 0:
            continue
        
        if job.get("job_id") in job_processes:
            continue
        
        # Check if job is already running by checking log file
        if os.path.exists(job["log_file"]):
            try:
                with open(job["log_file"], "r") as f:
                    first_line = f.readline().strip()
                    if "Starting job for route" in first_line:
                        print(f"{job['log_file']} already started.")
                        continue
            except:
                pass
        
        # Assign ports
        carla_world_port_start = next(iter(carla_world_ports.difference(used_ports)))
        carla_streaming_port_start = next(iter(carla_streaming_ports.difference(used_ports)))
        carla_tm_port_start = next(iter(carla_tm_ports.difference(used_ports)))
        
        # Start the job
        process = run_job_local(job, carla_tm_port_start, carla_world_port_start)
        
        if process:
            job_counter += 1
            job_id = f"local_job_{job_counter}"
            job["job_id"] = job_id
            job["ports"] = {carla_world_port_start, carla_tm_port_start}
            job_processes[job_id] = process
            job["tries"] -= 1
            
            print(f'Started job {job_id} for route {job["route_id"]}')
            print(f'Remaining jobs: {len(job_queue)}')
            break
    
    time.sleep(10)

print("All jobs completed!")
