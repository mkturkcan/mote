from huggingface_hub import hf_hub_download
R="unsloth/gemma-4-E2B-it-GGUF"
# Q6_K = near-lossless reference; Q3_K_M and UD-IQ2_M = calibrated low-bit test points
for fn in ["gemma-4-E2B-it-Q6_K.gguf","gemma-4-E2B-it-Q3_K_M.gguf","gemma-4-E2B-it-UD-IQ2_M.gguf"]:
    print("downloading", fn, flush=True)
    print(" ->", hf_hub_download(R, fn, local_dir="models"), flush=True)
print("ALL DONE", flush=True)
