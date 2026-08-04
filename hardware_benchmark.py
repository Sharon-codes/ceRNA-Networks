import os
import sys
import time
import torch
import torch.nn as nn

# Add workspace to path
sys.path.append(os.getcwd())
from model import MambaTCR

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Hardware Benchmark running on: {device} ({gpu_name})")
    
    # Load model
    model = MambaTCR(d_model=64, nhead=8, num_layers=2).to(device)
    checkpoint_path = "./Checkpoints/best_mamba_tcr_production.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded SWA checkpoint successfully.")
    else:
        print("SWA checkpoint not found. Running with randomly initialized model.")
        
    model.eval()
    
    # Configuration
    batch_size = 512
    total_pairs = 100000
    num_full_batches = total_pairs // batch_size
    remainder = total_pairs % batch_size
    
    # Generate dummy input tensors for full batch and remainder batch
    dummy_beta_full = torch.randn(batch_size, 30, 320, device=device)
    dummy_alpha_full = torch.randn(batch_size, 30, 320, device=device)
    dummy_pephla_full = torch.randn(batch_size, 50, 320, device=device)
    
    if remainder > 0:
        dummy_beta_rem = torch.randn(remainder, 30, 320, device=device)
        dummy_alpha_rem = torch.randn(remainder, 30, 320, device=device)
        dummy_pephla_rem = torch.randn(remainder, 50, 320, device=device)
        
    # Warm up GPU/CPU
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_beta_full, dummy_alpha_full, dummy_pephla_full)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    print(f"Starting inference on {total_pairs:,} pairs with batch size {batch_size}...")
    t0 = time.time()
    
    with torch.no_grad():
        # Process full batches
        for _ in range(num_full_batches):
            _ = model(dummy_beta_full, dummy_alpha_full, dummy_pephla_full)
            
        # Process remainder
        if remainder > 0:
            _ = model(dummy_beta_rem, dummy_alpha_rem, dummy_pephla_rem)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    
    elapsed_time = t1 - t0
    throughput = total_pairs / elapsed_time
    
    # Output results
    results_str = (
        "=== HARDWARE INFERENCE BENCHMARK RESULTS ===\n"
        f"Device Used:              {gpu_name} ({device.type.upper()})\n"
        f"Total Pairs Evaluated:    {total_pairs:,}\n"
        f"Batch Size:               {batch_size}\n"
        f"Total Execution Time:     {elapsed_time:.4f} seconds\n"
        f"Inference Throughput:     {throughput:.2f} pairs/second\n"
    )
    print(results_str)
    
    # Save to file
    os.makedirs("./Evaluation", exist_ok=True)
    out_path = "./Evaluation/hardware_benchmark_results.txt"
    with open(out_path, "w") as f:
        f.write(results_str)
        
    print(f"Benchmark results logged successfully to {out_path}")

if __name__ == "__main__":
    main()
