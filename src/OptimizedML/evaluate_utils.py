# utils/evaluate_utils.py
import torch
import os
import time
import numpy as np

def print_model_size(model: torch.nn.Module, label: str, file_prefix: str):
    """Saves the model's state_dict temporarily and prints its size."""
    # Use prefix and label for unique temp filenames
    temp_file_path = f"{file_prefix}_{label}_temp_model_state.pth"
    try:
        torch.save(model.state_dict(), temp_file_path)
        size_bytes = os.path.getsize(temp_file_path)
        size_mb = size_bytes / (1024 * 1024)
        print(f"{label} Model size: {size_mb:.2f} MB")
        os.remove(temp_file_path)
        return size_mb
    except Exception as e:
        print(f"Error calculating size for {label}: {e}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path) # Clean up if error occurred after saving
        return 0

def time_model_inference(model: torch.nn.Module,
                          input_tensor: torch.Tensor,
                          num_runs: int = 50,
                          warm_up: int = 10):
    """Times model inference accurately, returning average time in milliseconds."""
    model.eval()
    model.cpu() # Ensure model is on CPU for timing comparison
    input_tensor = input_tensor.to('cpu')
    times = []

    with torch.no_grad():
        # Warm-up runs
        print(f"  Performing {warm_up} warm-up runs...")
        for _ in range(warm_up):
            _ = model(input_tensor)

        # Timed runs
        print(f"  Performing {num_runs} timed runs...")
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_tensor)
            end_time = time.time()
            times.append((end_time - start_time) * 1000) # Store time in milliseconds

    if not times:
        print("  Error: No timed runs were completed.")
        return 0.0

    avg_time_ms = np.mean(times)
    std_dev_ms = np.std(times)
    print(f"  Avg time: {avg_time_ms:.3f} ms, Std Dev: {std_dev_ms:.3f} ms")
    return avg_time_ms

def compare_models(fp32_model: torch.nn.Module,
                   int8_model: torch.nn.Module,
                   calibration_loader: torch.utils.data.DataLoader,
                   file_prefix: str):
    """Compares FP32 and INT8 models for size and inference speed."""
    print("\n--- Comparing Model Performance ---")

    # --- Size Comparison ---
    print("\nComparing Model Sizes:")
    fp32_size = print_model_size(fp32_model, "FP32_Original", file_prefix)
    int8_size = print_model_size(int8_model, "INT8_Quantized", file_prefix)

    if int8_size > 0 and fp32_size > 0:
        size_reduction = fp32_size / int8_size
        print(f"\nSize reduction factor: {size_reduction:.2f}x")
        print(f"Model size reduced from {fp32_size:.2f} MB to {int8_size:.2f} MB.")
    else:
        print("\nCould not calculate size reduction (one or both model sizes are zero or invalid).")

    # --- Speed Comparison ---
    print("\nComparing Inference Speed (CPU):")
    try:
        # Get a sample batch for timing
        calib_iter = iter(calibration_loader)
        sample_input, _ = next(calib_iter)
        sample_input_cpu = sample_input.to('cpu')
        print(f"Using sample input batch of shape: {sample_input_cpu.shape} on CPU for timing.")

        print("\nTiming Original FP32 model inference...")
        fp32_avg_time = time_model_inference(fp32_model, sample_input_cpu)
        print(f"Average FP32 inference time: {fp32_avg_time:.3f} ms per batch")

        print("\nTiming INT8 model inference...")
        int8_avg_time = time_model_inference(int8_model, sample_input_cpu)
        print(f"Average INT8 inference time: {int8_avg_time:.3f} ms per batch")

        if int8_avg_time > 0 and fp32_avg_time > 0:
            speedup_factor = fp32_avg_time / int8_avg_time
            print(f"\nInference speedup factor (INT8 vs FP32 on CPU): {speedup_factor:.2f}x")
        else:
            print("\nCould not calculate speedup factor (times were zero or invalid).")

    except StopIteration:
        print("\nError: Could not get a batch from calibration_loader for speed test.")
    except Exception as e:
        print(f"\nAn error occurred during speed comparison: {e}")