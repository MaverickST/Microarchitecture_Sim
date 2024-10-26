import subprocess
import shutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import re
import threading

# Base paths for workloads and other directories
WORKLOADS_BASE_PATH = os.path.expanduser("~/SimTools/workloads")
GEM5_PATH = os.path.expanduser("~/gem5/build/ARM")
SCRIPT_PATH = os.path.expanduser("~/SimTools/scripts/CortexA76_scripts_gem5")
MCPAT_SCRIPT = os.path.expanduser("~/SimTools/scripts/McPAT/gem5toMcPAT_cortexA76.py")
MCPAT_CONFIG = os.path.expanduser("~/SimTools/scripts/McPAT/ARM_A76_2.1GHz.xml")
MCPAT_PATH = os.path.expanduser("~/SimTools/mcpat/mcpat")

# Define hardcoded options for each workload
MOREOPTIONS_MAP = {
    "h264_enc": [
        [""],
        ["--l1i_size=32kB", "--l1d_size=128kB"],
        ["--l1i_size=64kB", "--l1d_size=256kB"],
        ["--l1i_size=16kB", "--l1d_size=512kB"]
    ],
    "h264_dec": [
        [""],
        ["--l1i_size=64kB", "--l2_size=512kB"],
        ["--l1i_size=128kB", "--l2_size=1MB"]
    ],
    "jpg2k_enc": [
        [""],
        ["--l1i_size=16kB", "--l1d_size=256kB"],
        ["--l1i_size=32kB", "--l1d_size=512kB"]
    ],
    "jpg2k_dec": [
        [""],
        ["--l1i_size=32kB", "--l2_size=128kB"],
        ["--l1i_size=64kB", "--l2_size=256kB"]
    ],
    "mp3_enc": [
        [""],
        ["--l1i_size=64kB", "--l1d_size=128kB"],
        ["--l1i_size=128kB", "--l2_size=512kB"]
    ],
    "mp3_dec": [
        [""],
        ["--l1i_size=32kB", "--l2_size=256kB"],
        ["--l1i_size=64kB", "--l2_size=1MB"]
    ],
}

def read_energy_txt(energy_txt_path):
    if not os.path.exists(energy_txt_path):
        raise FileNotFoundError(f"{energy_txt_path} not found.")
    
    # Ensure the file is not empty
    if os.path.getsize(energy_txt_path) == 0:
        raise ValueError(f"{energy_txt_path} is empty.")
    
    total_leakage = None
    runtime_dynamic = None

    # Parse the text file for relevant energy values
    with open(energy_txt_path, "r") as file:
        for line in file:
            if "Total Leakage" in line:  # Adjust based on the actual format of mcpat.txt
                total_leakage = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
            if "Runtime Dynamic" in line:  # Adjust based on the actual format of mcpat.txt
                runtime_dynamic = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])

    if total_leakage is None or runtime_dynamic is None:
        raise ValueError(f"Failed to extract required values from {energy_txt_path}")
    
    return total_leakage, runtime_dynamic

def read_stats_txt(stats_txt_path):
    cpi = None
    
    with open(stats_txt_path, "r") as f:
        for line in f:
            if "system.cpu.cpi" in line:
                cpi = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                break
    
    if cpi is None:
        raise ValueError(f"Failed to extract system.cpu.cpi from {stats_txt_path}")
    
    return cpi

def calculate_energy_and_edp(total_leakage, runtime_dynamic, cpi):
    energy = (total_leakage + runtime_dynamic) * cpi
    edp = energy * cpi
    return energy, edp

def append_to_energy_txt(energy_txt_path, energy, edp, cpi):
    with open(energy_txt_path, "a") as file:
        file.write(f"\nCalculated Energy: {energy}\n")
        file.write(f"EDP: {edp}\n")
        file.write(f"CPI: {cpi}\n")

# Create a lock to prevent concurrent access to McPAT
mcpat_lock = threading.Lock()

# Function to run a single simulation (thread) with output as a folder
def run_gem5(cmd, output, cmd_options, moreoptions):
    gem5_cmd = os.path.join(GEM5_PATH, "gem5.fast")
    script = os.path.join(SCRIPT_PATH, "CortexA76.py")

    # Ensure the output folder exists
    os.makedirs(output, exist_ok=True)
    
    # Build the command list for subprocess.run
    outdir = "--outdir=" + output
    cmd_arg = "--cmd=" + cmd
    options_arg = "--options=" + cmd_options
    
    # Combine all arguments into a list
    gem5_command = [gem5_cmd, outdir, script, cmd_arg, options_arg] + moreoptions

    # Execute the gem5 command and redirect output to a log file inside the output folder
    logfile = os.path.join(output, "simulation.log")
    # with open(logfile, "w") as outfile:
    #     subprocess.run(gem5_command, stdout=outfile, stderr=outfile)  # Redirect stderr to the same file

    # After gem5 simulation, run McPAT conversion
    stats_file = os.path.join(output, "stats.txt")
    config_file = os.path.join(output, "config.json")
    
    if os.path.exists(stats_file) and os.path.exists(config_file):

        # Lock the McPAT conversion to prevent concurrent access
        mcpat_lock.acquire()
        try:
            # Execute the McPAT conversion
            mcpat_command = ["python3", MCPAT_SCRIPT, stats_file, config_file, MCPAT_CONFIG]
            subprocess.run(mcpat_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Move config.xml to the output folder
            destination_file = os.path.join(output, "config.xml")
            try:
                shutil.move(os.path.join(".", "config.xml"), destination_file)
            except Exception as e:
                with open(logfile, "a") as outfile:
                    outfile.write(f"Error moving config.xml: {e}\n")  # Log any errors in moving

            # Execute McPAT and redirect output to a log file inside the output folder
            mcpat_command = [MCPAT_PATH, "-infile", os.path.join(output, "config.xml")]
            mcpat_logfile = os.path.join(output, "mcpat.txt")
            with open(mcpat_logfile, "a") as mcpat_outfile:
                subprocess.run(mcpat_command, stdout=mcpat_outfile, stderr=mcpat_outfile)

        finally:
            mcpat_lock.release()
        
        # Now process the results and append calculations to mcpat.txt
        try:
            # Read mcpat.txt file
            energy_txt_file = os.path.join(output, "mcpat.txt")
            total_leakage, runtime_dynamic = read_energy_txt(energy_txt_file)

            # Read stats.txt file
            cpi = read_stats_txt(stats_file)

            # Calculate energy and EDP
            energy, edp = calculate_energy_and_edp(total_leakage, runtime_dynamic, cpi)
            print(f"For:{output}\n  ->Energy: {energy}, EDP: {edp}, CPI: {cpi}, Total Leakage: {total_leakage}, Runtime Dynamic: {runtime_dynamic}")

            # Append calculated values to mcpat.txt
            append_to_energy_txt(energy_txt_file, energy, edp, cpi)
        except Exception as e:
            # Log this in the same log file instead of printing to the console
            with open(logfile, "a") as outfile:
                outfile.write(f"Error processing results for {output}: {e}\n")

    else:
        # Log this in the same log file instead of printing to the console
        with open(logfile, "a") as outfile:
            outfile.write(f"Skipping McPAT for {output}: Missing stats.txt or config.json\n")

# Function to manage thread pools with a specific limit (concurrent threads)
def run_simulations_in_batches(simulations, threads_per_batch):
    with ThreadPoolExecutor(max_workers=threads_per_batch) as executor:
        futures = []

        # Submit each simulation to the pool
        for sim in simulations:
            cmd = sim["cmd"]
            output = sim["output"]
            cmd_options = sim["cmd_options"]
            moreoptions = sim["moreoptions"]
            
            futures.append(executor.submit(run_gem5, cmd, output, cmd_options, moreoptions))
        
        # Wait for each future (simulation) to complete
        for future in as_completed(futures):
            try:
                future.result()  # This will raise any exception if the thread fails
            except Exception as e:
                # Log the exception in a file instead of printing
                with open("error.log", "a") as error_log:
                    error_log.write(f"Simulation failed: {e}\n")

# Main function with help and usage examples
def main():
    # Define the available commands and their respective options using WORKLOADS_BASE_PATH
    cmd_map = {
        "h264_enc": os.path.join(WORKLOADS_BASE_PATH, "h264_enc/h264_enc"),
        "h264_dec": os.path.join(WORKLOADS_BASE_PATH, "h264_dec/h264_dec"),
        "jpg2k_enc": os.path.join(WORKLOADS_BASE_PATH, "jpg2k_enc/jpg2k_enc"),
        "jpg2k_dec": os.path.join(WORKLOADS_BASE_PATH, "jpg2k_dec/jpg2k_dec"),
        "mp3_enc": os.path.join(WORKLOADS_BASE_PATH, "mp3_enc/mp3_enc"),
        "mp3_dec": os.path.join(WORKLOADS_BASE_PATH, "mp3_dec/mp3_dec")
    }

    # Define the corresponding options for each command
    cmd_options_map = {
        "h264_enc": os.path.join(WORKLOADS_BASE_PATH, "h264_enc/h264enc_configfile.cfg") + " -org " + os.path.join(WORKLOADS_BASE_PATH, "h264_enc/h264enc_testfile.yuv"),
        "h264_dec": os.path.join(WORKLOADS_BASE_PATH, "h264_dec/h264dec_testfile.264") + " " + os.path.join(WORKLOADS_BASE_PATH, "h264_dec/h264dec_outfile.yuv"),
        "jpg2k_enc": "-i " + os.path.join(WORKLOADS_BASE_PATH, "jpg2k_enc/jpg2kenc_testfile.bmp") + " -o " + os.path.join(WORKLOADS_BASE_PATH, "jpg2k_enc/jpg2kenc_outfile.j2k"),
        "jpg2k_dec": "-i " + os.path.join(WORKLOADS_BASE_PATH, "jpg2k_dec/jpg2kdec_testfile.j2k") + " -o " + os.path.join(WORKLOADS_BASE_PATH, "jpg2k_dec/jpg2kdec_outfile.bmp"),
        "mp3_enc": os.path.join(WORKLOADS_BASE_PATH, "mp3_enc/mp3enc_testfile.wav") + " " + os.path.join(WORKLOADS_BASE_PATH, "mp3_enc/mp3enc_outfile.mp3"),
        "mp3_dec": "-w " + os.path.join(WORKLOADS_BASE_PATH, "mp3_dec/mp3dec_outfile.wav") + " " + os.path.join(WORKLOADS_BASE_PATH, "mp3_dec/mp3dec_testfile.mp3")
    }

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Run GEM5 simulations in parallel for different workloads.",
        epilog="Example usage:\n  python3 profiling.py -t 2 -w h264_enc h264_dec"
    )
    parser.add_argument('-t', '--threads', type=int, default=3, help="Number of concurrent simulations to use (default: 3)")
    parser.add_argument('-w', '--workloads', nargs='+', required=True, help="List of workloads to simulate (e.g., h264_enc h264_dec)")

    args = parser.parse_args()

    # Create a list of simulations to run based on specified workloads
    simulations = []
    workload_counts = {key: 0 for key in MOREOPTIONS_MAP.keys()}  # Count for each workload

    for cmd_key in args.workloads:
        if cmd_key in cmd_map:
            cmd = cmd_map[cmd_key]
            cmd_options = cmd_options_map[cmd_key]

            # Determine the index for options from MOREOPTIONS_MAP
            option_index = workload_counts[cmd_key] % len(MOREOPTIONS_MAP[cmd_key])
            moreoptions = MOREOPTIONS_MAP[cmd_key][option_index]

            # Define unique output folder for each simulation
            outputfolder = f"{cmd_key}_simout__{workload_counts[cmd_key]}"
            workload_counts[cmd_key] += 1  # Increment count for this workload
            
            # Append simulation configuration
            simulations.append({
                "cmd": cmd,
                "cmd_options": cmd_options,
                "moreoptions": moreoptions,
                "output": outputfolder
            })

    # Run the simulations with the specified number of threads
    run_simulations_in_batches(simulations, threads_per_batch=args.threads)

if __name__ == "__main__":
    main()
