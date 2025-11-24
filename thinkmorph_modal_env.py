import modal

app = modal.App("thinkmorph-env")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("datasets")
    .workdir("/root/ThinkMorph")
    .add_local_dir(".", "/root/ThinkMorph")
)

@app.function(image=image, gpu="T4", timeout=600)
def env_check():
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("Torch version:", torch.__version__)

    try:
        import transformers
        print("Transformers version:", transformers.__version__)
    except Exception as e:
        print("Transformers import failed:", e)

    try:
        from datasets import load_dataset
        print("datasets import OK")
    except Exception as e:
        print("datasets import failed:", e)

    try:
        import inferencer
        print("ThinkMorph inferencer.py import OK")
    except Exception as e:
        print("ThinkMorph inferencer import failed:", e)

@app.local_entrypoint()
def main():
    env_check.remote()
