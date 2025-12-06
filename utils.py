import os
import subprocess
import sys
import time

def log_config(config: dict, out_dir: str, file_name: str = "config.py"):
    """Log the config to a file in out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, file_name)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("# training configuration saved at " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        for k in sorted(config.keys()):
            v = config[k]
            if isinstance(v, str):
                f.write(f'{k} = "{v}"\n')
            else:
                f.write(f"{k} = {v}\n")
    print(f"config logged to {config_path}")

def log_command(out_dir: str, file_name: str = "command.sh"):
    """Log the command that runs the script to config.out_dir/command.sh"""
    os.makedirs(out_dir, exist_ok=True)
    command_path = os.path.join(out_dir, file_name)
    with open(command_path, 'w', encoding='utf-8') as f:
        f.write("#!/bin/bash\n")
        f.write(" ".join(sys.argv) + "\n")
    os.chmod(command_path, 0o755)  # Make the script executable
    print(f"Command logged to {command_path}")

def log_code_status(out_dir: str, commit_file_name: str = "git-commit.txt", patch_file_name: str = "git-patch.txt"):
    """Log the current git commit hash to out_dir/commit_file_name and any uncommitted changes to out_dir/git_patch.txt"""
    os.makedirs(out_dir, exist_ok=True)
    commit_path = os.path.join(out_dir, commit_file_name)
    with open(commit_path, 'w', encoding='utf-8') as f:
        try:
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
            f.write(f"{commit_hash}\n\n")
            print(f"Git patch logged to {commit_path}.")
        except Exception as e:
            f.write(f"Error retrieving git information: {e}\n")
            print(f"Failed to log git information: {e}")

    patch_path = os.path.join(out_dir, patch_file_name)
    with open(patch_path, 'w', encoding='utf-8') as f:
            patch = subprocess.check_output(['git', 'diff']).decode('utf-8')
            if patch:
                f.write("Uncommitted code changes:\n")
                f.write(patch)
                print(f"Git patch logged to {patch_path}.")
            else:
                f.write("No uncommitted code changes.\n")

def log_wandb_run_id(out_dir: str, run_id: str, file_name: str = "wandb_run_id.txt"):
    """Log the wandb run ID to out_dir/wandb_run_id.txt"""
    os.makedirs(out_dir, exist_ok=True)
    run_id_path = os.path.join(out_dir, file_name)
    with open(run_id_path, 'w', encoding='utf-8') as f:
        f.write(f"{run_id}\n")
    print(f"Wandb run ID {run_id} logged to {run_id_path}")


