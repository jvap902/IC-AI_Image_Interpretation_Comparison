import subprocess
import sys
import os
import time
from src import *

def run_main_with_subprocess(args):
    """
    Executes the main.py script as an external command, passing along 
    any command-line arguments provided to this script.
    """
    print("--- Starting Execution of main.py via Subprocess ---")
    
    # Base command: [python interpreter, 'main.py']
    command = [sys.executable, 'main.py']
    
    # Append all arguments passed to run_pipeline.py to the command list
    if args:
        print(f"--- Passing arguments to main.py: {args} ---")
        command.extend(args) # Use .extend() to add the list of arguments
    
    try:
        # Run the command
        # check=True raises an exception if main.py exits with an error
        start_time = time.time()
        process = subprocess.run(
            command, 
            check=True,  
            capture_output=True, 
            text=True    
        )
        end_time = time.time()

        print("\n--- main.py Output (stdout) ---")
        print(process.stdout)
        
        if process.stderr:
            # Note: Errors here mean main.py printed to stderr, not necessarily failed (unless check=True caught it)
            print("\n--- main.py Errors (stderr) ---")
            print(process.stderr)
        
        print(f"--- Execution of main.py finished successfully in {end_time - start_time:.2f} seconds ---")

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: main.py failed with exit code {e.returncode}")
        print("\n--- main.py Errors (stderr) ---")
        print(e.stderr)
    except FileNotFoundError:
        print("ERROR: Could not find 'main.py'. Ensure it is in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred during subprocess execution: {e}")


if __name__ == "__main__":
    # Capture all command-line arguments passed to this script, 
    # excluding the script name itself (sys.argv[0] is 'run_pipeline.py').

    sizes = [500, 300, 200]
    models = [('resnet50.a1_in1k', 'efficientnet_b0.ra_in1k')]

    for size in sizes:
        for model in models:
            m1, m2 = model
            arguments_to_pass = ['--size', str(size), "--model1", m1, "--model2", m2, "--dataset", "cifar100"]
            run_main_with_subprocess(arguments_to_pass)