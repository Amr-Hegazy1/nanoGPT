
import os
import argparse
from huggingface_hub import HfApi, snapshot_download

def pull_from_hf(model_name, save_dir=None, root_save_dir=None, hf_token=None):
    # Initialize API
    api = HfApi(token=hf_token)

    # Determine final save path
    if root_save_dir:
        # Use root_save_dir/model_name as the target path
        final_path = os.path.join(root_save_dir, model_name.replace("/", "_"))
    elif save_dir:
        final_path = save_dir
    else:
        raise ValueError("You must provide either --save_dir or --root_save_dir.")

    # Ensure directory exists
    os.makedirs(final_path, exist_ok=True)

    print(f"Downloading model: {model_name} to {final_path}")

    # Download the entire repo snapshot
    local_dir = snapshot_download(
        repo_id=model_name,
        repo_type="model",  # Change to "dataset" or "space" if needed
        local_dir=final_path,  # Explicitly set target directory
        resume_download=True,
        token=hf_token
    )

    print(f"✅ Download complete. Files saved to: {local_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull a model from Hugging Face Hub.")
    parser.add_argument("--model_name", required=True, help="Full Hugging Face repo name (e.g., username/repo_name).")
    parser.add_argument("--save_dir", help="Local path to save the downloaded model.")
    parser.add_argument("--root_save_dir", help="Root directory; model will be saved under root_save_dir/model_name.")
    parser.add_argument("--hf_token", help="Hugging Face API token. If not provided, uses default from environment or CLI login.")
    args = parser.parse_args()

    pull_from_hf(model_name=args.model_name, save_dir=args.save_dir, root_save_dir=args.root_save_dir, hf_token=args.hf_token)
