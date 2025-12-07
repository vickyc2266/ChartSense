from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="hf-internal-testing/siglip-so400m-14-980-flash-attn2-navit",
    local_dir="vit_test",
    local_dir_use_symlinks=False,
)