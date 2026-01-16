import argparse
import sys
import json
from . import __version__

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="dredge", 
        description="DREDGE x Dolly - GPU-CPU Lifter · Save · Files · Print"
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("serve", help="Start the DREDGE x Dolly web server")
    server_parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Host to bind to (default: 0.0.0.0)"
    )
    server_parser.add_argument(
        "--port", 
        type=int, 
        default=3001, 
        help="Port to listen on (default: 3001)"
    )
    server_parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    
    # Quasimoto fit-wave command
    fit_wave_parser = subparsers.add_parser("fit-wave", help="Fit Quasimoto wave to signal data")
    fit_wave_parser.add_argument("input_file", help="Input signal file (JSON array)")
    fit_wave_parser.add_argument("--output", "-o", help="Output model path (JSON)")
    fit_wave_parser.add_argument("--epochs", type=int, default=2000, help="Training epochs (default: 2000)")
    fit_wave_parser.add_argument("--n-waves", type=int, default=16, help="Number of waves in ensemble (default: 16)")
    fit_wave_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use")
    
    # Process signal command
    process_signal_parser = subparsers.add_parser("process-signal", help="Process signal with trained wave")
    process_signal_parser.add_argument("signal_file", help="Signal file to process (JSON array)")
    process_signal_parser.add_argument("--model", "-m", help="Trained model path (JSON)")
    process_signal_parser.add_argument("--output", "-o", help="Output file for predictions")
    
    # Analyze anomaly command
    analyze_anomaly_parser = subparsers.add_parser("analyze-anomaly", help="Detect anomalies in signal")
    analyze_anomaly_parser.add_argument("signal_file", help="Signal file to analyze (JSON array)")
    analyze_anomaly_parser.add_argument("--threshold", type=float, default=2.0, help="Anomaly threshold (std devs)")
    analyze_anomaly_parser.add_argument("--n-waves", type=int, default=16, help="Number of waves")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run Quasimoto benchmarks")
    benchmark_parser.add_argument("--model", default="quasimoto", choices=["quasimoto", "all"], help="Model to benchmark")
    benchmark_parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    
    args = parser.parse_args(argv)
    
    if args.version:
        print(__version__)
        return 0
    
    if args.command == "serve":
        from .server import run_server
        run_server(host=args.host, port=args.port, debug=args.debug)
        return 0
    
    elif args.command == "fit-wave":
        return _fit_wave_command(args)
    
    elif args.command == "process-signal":
        return _process_signal_command(args)
    
    elif args.command == "analyze-anomaly":
        return _analyze_anomaly_command(args)
    
    elif args.command == "benchmark":
        return _benchmark_command(args)
    
    parser.print_help()
    return 0


def _fit_wave_command(args):
    """Execute fit-wave command."""
    try:
        from .quasimoto import create_processor
    except ImportError:
        print("Error: PyTorch is required for wave fitting. Install with: pip install torch")
        return 1
    
    # Load signal data
    try:
        with open(args.input_file, 'r') as f:
            signal_data = json.load(f)
    except Exception as e:
        print(f"Error loading signal file: {e}")
        return 1
    
    print(f"Fitting Quasimoto wave to {len(signal_data)} data points...")
    print(f"Ensemble size: {args.n_waves} waves")
    print(f"Training for {args.epochs} epochs on {args.device}")
    
    processor = create_processor(n_waves=args.n_waves, device=args.device)
    result = processor.fit(signal_data, epochs=args.epochs)
    
    print(f"\nTraining complete!")
    print(f"Final loss: {result['final_loss']:.8f}")
    
    # Save model parameters if output specified
    if args.output:
        output_data = {
            "parameters": processor.get_wave_parameters(),
            "training_result": result,
            "n_waves": args.n_waves
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Model saved to {args.output}")
    
    return 0


def _process_signal_command(args):
    """Execute process-signal command."""
    try:
        from .quasimoto import create_processor
    except ImportError:
        print("Error: PyTorch is required. Install with: pip install torch")
        return 1
    
    # Load signal
    try:
        with open(args.signal_file, 'r') as f:
            signal_data = json.load(f)
    except Exception as e:
        print(f"Error loading signal file: {e}")
        return 1
    
    print(f"Processing signal with {len(signal_data)} data points...")
    
    # For now, fit the signal (in production, would load pre-trained model)
    processor = create_processor(n_waves=16)
    processor.fit(signal_data, epochs=1000)
    
    # Generate predictions
    x_values = list(range(len(signal_data)))
    predictions = processor.predict(x_values)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Predictions saved to {args.output}")
    else:
        print(f"First 10 predictions: {predictions[:10]}")
    
    return 0


def _analyze_anomaly_command(args):
    """Execute analyze-anomaly command."""
    try:
        from .quasimoto import create_processor
    except ImportError:
        print("Error: PyTorch is required. Install with: pip install torch")
        return 1
    
    # Load signal
    try:
        with open(args.signal_file, 'r') as f:
            signal_data = json.load(f)
    except Exception as e:
        print(f"Error loading signal file: {e}")
        return 1
    
    print(f"Analyzing signal for anomalies ({len(signal_data)} points)...")
    print(f"Threshold: {args.threshold} standard deviations")
    
    processor = create_processor(n_waves=args.n_waves)
    processor.fit(signal_data, epochs=1500)
    
    anomalies = processor.detect_anomalies(signal_data, threshold=args.threshold)
    
    print(f"\nFound {len(anomalies)} anomalies:")
    if anomalies:
        print(f"Anomaly indices: {anomalies[:20]}" + ("..." if len(anomalies) > 20 else ""))
        print(f"Percentage of signal: {100 * len(anomalies) / len(signal_data):.2f}%")
    
    return 0


def _benchmark_command(args):
    """Execute benchmark command."""
    try:
        from .quasimoto import create_processor
        import torch
    except ImportError:
        print("Error: PyTorch is required. Install with: pip install torch")
        return 1
    
    print(f"Running Quasimoto benchmark with {args.epochs} epochs...")
    
    # Generate test signal (chirp with glitch)
    x = torch.linspace(-10, 10, 1000)
    signal = torch.sin(0.5 * x**2) * torch.exp(-0.1 * x**2)
    glitch_start = 500
    glitch_end = 550
    signal[glitch_start:glitch_end] += 0.5 * torch.sin(20 * x[glitch_start:glitch_end])
    
    signal_data = signal.tolist()
    
    processor = create_processor(n_waves=16)
    result = processor.fit(signal_data, epochs=args.epochs)
    
    print(f"\nBenchmark Results:")
    print(f"Model: Quasimoto Ensemble (16 waves)")
    print(f"Epochs: {result['epochs']}")
    print(f"Final Loss: {result['final_loss']:.8f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
