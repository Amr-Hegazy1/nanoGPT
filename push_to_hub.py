
import os
import argparse
from huggingface_hub import HfApi, upload_folder

def push_to_hf(root_load_dir=None, load_dir=None, prefix="nanogpt-", private=True, hf_token=None):
    if not root_load_dir and not load_dir:
        raise ValueError("You must provide either --root_load_dir or --load_dir.")

    api = HfApi(token=hf_token)

    # Prepare list of directories to upload
    dirs_to_upload = []
    if load_dir:
        if not os.path.isdir(load_dir):
            raise ValueError(f"Provided load_dir path is not a directory: {load_dir}")
        dirs_to_upload.append(load_dir)
    elif root_load_dir:
        for subdir in os.listdir(root_load_dir):
            full_path = os.path.join(root_load_dir, subdir)
            if os.path.isdir(full_path):
                dirs_to_upload.append(full_path)


    username = api.whoami(token=hf_token)["name"]  # Get username dynamically

    # Upload each directory
    for folder_path in dirs_to_upload:
        repo_name = f"{prefix}{os.path.basename(folder_path)}"
        repo_id = f"{username}/{repo_name}"  # Correct repo_id format

        print(f"Creating and uploading: {repo_id}")

        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

        upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token
        )

        print(f"✅ Uploaded {folder_path} to {repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push directories to Hugging Face Hub without Git.")
    parser.add_argument("--root_load_dir", help="Base directory containing subdirectories.")
    parser.add_argument("--load_dir", help="Path to a single directory to upload.")
    parser.add_argument("--prefix", default="nanogpt-", help="Prefix for Hugging Face repo names.")
    parser.add_argument("--public", action="store_true", help="Make repos public instead of private.")
    parser.add_argument("--hf_token", help="Hugging Face API token. If not provided, uses default from environment or CLI login.")
    args = parser.parse_args()

    push_to_hf(root_load_dir=args.root_load_dir, load_dir=args.load_dir, prefix=args.prefix, private=not args.public, hf_token=args.hf_token)
