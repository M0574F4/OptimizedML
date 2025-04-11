# OLD - sys.path manipulation
# import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# utils_dir = os.path.join(parent_dir, 'utils')
# if utils_dir not in sys.path:
#     sys.path.append(utils_dir)
# from model_utils import ...
# from data_utils import ...
# from evaluate_utils import ...

# NEW - Relative imports within the package
import torch
import argparse
import os
# No sys.path manipulation needed

# Use '.' for relative import from the same directory (OptimizedML)
from .model_utils import get_model_and_transform, get_quantization_aware_model, fuse_model_modules, SUPPORTED_MODELS
from .data_utils import get_calibration_loader
from .evaluate_utils import compare_models

def main(args):
    print(f"--- Starting PTQ Static Quantization for {args.model_name} ---")
    print(f"PyTorch Version: {torch.__version__}")

    # --- 1. Load Model ---
    try:
        fp32_model, preprocess = get_model_and_transform(args.model_name)
        model_to_quantize = get_quantization_aware_model(args.model_name, fp32_model)
    except ValueError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        return

    # --- 2. Configure Quantization ---
    print("\n--- Configuring Quantization ---")
    q_backend = args.backend
    supported_backends = torch.backends.quantized.supported_engines
    if q_backend not in supported_backends:
        print(f"Warning: Requested backend '{q_backend}' not in supported list: {supported_backends}.")
        # Fallback logic or selection logic could go here
        if 'fbgemm' in supported_backends:
             q_backend = 'fbgemm'
             print(f"Falling back to 'fbgemm'.")
        elif 'qnnpack' in supported_backends:
             q_backend = 'qnnpack'
             print(f"Falling back to 'qnnpack'.")
        else:
             print("Error: No supported quantization backend found ('fbgemm' or 'qnnpack'). Cannot proceed.")
             return

    try:
        qconfig = torch.quantization.get_default_qconfig(q_backend)
        torch.backends.quantized.engine = q_backend
        model_to_quantize.qconfig = qconfig
        print(f"Quantization backend set to: {q_backend}")
        print("Quantization configuration applied to the model.")
    except Exception as e:
        print(f"Error setting up quantization backend {q_backend}: {e}")
        return

    # --- 3. Fuse Modules ---
    fuse_model_modules(model_to_quantize) # Fuse the model we intend to quantize

    # --- 4. Prepare Calibration Data ---
    calibration_loader = get_calibration_loader(
        args.data_path,
        preprocess,
        num_samples=args.num_calib_samples,
        batch_size=args.batch_size
    )
    if calibration_loader is None:
        print("Error: Failed to create calibration data loader. Exiting.")
        return

    # --- 5. Prepare Model for Quantization ---
    print("\n--- Preparing Model for Static Quantization (PTQ) ---")
    model_to_quantize.cpu().eval()
    try:
        # `prepare` inserts observers. Operates inplace.
        torch.quantization.prepare(model_to_quantize, inplace=True)
        print("Model prepared for static quantization (observers inserted).")
        prepared_model = model_to_quantize # Keep track using this name now
    except Exception as e:
        print(f"Error during torch.quantization.prepare: {e}")
        return

    # --- 6. Calibrate Model ---
    print("\n--- Calibrating the Model ---")
    print("Running calibration data through the prepared model...")
    try:
        prepared_model.cpu().eval()
        with torch.no_grad():
            for i, (images, _) in enumerate(calibration_loader):
                images_cpu = images.to('cpu')
                prepared_model(images_cpu) # Feed data to the *prepared* model
                print(f"  Calibration batch {i+1}/{len(calibration_loader)} processed.", end='\r')
        print("\nCalibration finished. Activation statistics collected.")
    except Exception as e:
        print(f"\nError during calibration loop: {e}")
        return

    # --- 7. Convert Model to INT8 ---
    print("\n--- Converting the Model to Quantized INT8 ---")
    int8_model = None
    try:
        prepared_model.cpu().eval()
        # `convert` replaces observers with quantized modules. Operates inplace.
        torch.quantization.convert(prepared_model, inplace=True)
        int8_model = prepared_model # prepared_model is now the converted INT8 model
        print("Model successfully converted to INT8 quantized format.")
    except Exception as e:
        print(f"Error during model conversion: {e}")
        return # Cannot proceed without converted model

    # --- 8. Save Quantized Model ---
    if int8_model:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            print(f"Created output directory: {args.output_dir}")

        output_model_path = os.path.join(args.output_dir, f"{args.model_name}_int8_quantized.pt")
        try:
            # Save the entire model (architecture + weights) or just state_dict
            # Saving state_dict is generally preferred for flexibility
            torch.save(int8_model.state_dict(), output_model_path)
            print(f"Quantized INT8 model state_dict saved to: {output_model_path}")
        except Exception as e:
            print(f"Error saving quantized model state_dict: {e}")

    # --- 9. Evaluate Performance (Optional) ---
    if args.evaluate and int8_model:
        compare_models(
            fp32_model=fp32_model,
            int8_model=int8_model,
            calibration_loader=calibration_loader,
            file_prefix=args.model_name # For unique temp file names in evaluation
        )
    else:
        print("\nSkipping evaluation.")

    print(f"\n--- Quantization process for {args.model_name} finished. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTQ Static Quantization Script")

    parser.add_argument('--model_name', type=str, required=True, choices=SUPPORTED_MODELS,
                        help=f"Name of the model to quantize. Supported: {SUPPORTED_MODELS}")
    parser.add_argument('--data_path', type=str, default='./data',
                        help="Path to the dataset directory (e.g., for CIFAR10 calibration)")
    parser.add_argument('--output_dir', type=str, default='./quantized_models',
                        help="Directory to save the quantized model state_dict")
    parser.add_argument('--backend', type=str, default='fbgemm', choices=['fbgemm', 'qnnpack'],
                        help="Quantization backend to use ('fbgemm' for x86, 'qnnpack' for ARM)")
    parser.add_argument('--num_calib_samples', type=int, default=500,
                        help="Number of samples to use for calibration")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for calibration data loader")
    parser.add_argument('--evaluate', action='store_true',
                        help="Evaluate model size and speed after quantization")

    args = parser.parse_args()
    main(args)