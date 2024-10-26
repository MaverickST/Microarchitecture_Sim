import os
import re
import matplotlib.pyplot as plt

# Function to extract the first percentage values from the given file
def extract_values(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Define the keys to look for individually and as categories
    keys = {
        'No_OpClass': 'system.cpu.statIssuedInstType_0::No_OpClass',
        'IntAlu': 'system.cpu.statIssuedInstType_0::IntAlu',
        'MemRead': 'system.cpu.statIssuedInstType_0::MemRead',
        'MemWrite': 'system.cpu.statIssuedInstType_0::MemWrite'
    }
    
    # Dictionary to store values
    values = {key: 0 for key in keys.keys()}
    values['Simd'] = 0  # Initialize Simd as 0
    total_percentage = 0

    # Process each line and extract the first percentage value
    for line in data:
        # Extract percentage for specific keys
        for key, identifier in keys.items():
            if identifier in line:
                match = re.search(r'(\d+\.\d+%)\s+(\d+\.\d+%)', line)
                if match:
                    extracted_value = float(match.group(1).replace('%', ''))
                    values[key] = extracted_value
                    total_percentage += extracted_value
                    print(f"Extracted {extracted_value}% for {key}")  # Debugging log
        
        # Sum all "Simd" related instructions
        if 'system.cpu.statIssuedInstType_0::Simd' in line:
            match = re.search(r'(\d+\.\d+%)\s+(\d+\.\d+%)', line)
            if match:
                simd_value = float(match.group(1).replace('%', ''))
                values['Simd'] += simd_value
                total_percentage += simd_value
                print(f"Added {simd_value}% to Simd")  # Debugging log

    # Calculate 'Others' category as the remaining percentage
    values['Others'] = max(0, 100 - total_percentage)
    print(f"Calculated 'Others': {values['Others']}%")  # Debugging log

    return values

# Function to process all output folders
def process_output_folders(base_dir):
    results = {}

    # Iterate over all directories in the base folder
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            stats_file = os.path.join(folder_path, 'stats.txt')

            # Check if the stats.txt file exists
            if os.path.exists(stats_file):
                print(f"Processing {folder}")  # Debugging log
                values = extract_values(stats_file)
                
                # Simplify folder name for labeling
                simple_name = folder.split('_simout')[0]
                results[simple_name] = values

    return results

# Function to generate a general plot with workloads on x-axis and annotations
def plot_results(results):
    categories = ['No_OpClass', 'IntAlu', 'MemRead', 'MemWrite', 'Simd', 'Others']
    workloads = list(results.keys())
    
    # Set positions for each workload on the x-axis
    x = range(len(workloads))
    bar_width = 0.15

    plt.figure(figsize=(12, 8))

    # Plot bars for each instruction type across all workloads
    for i, category in enumerate(categories):
        # Collect data for each workload for the given category
        percentages = [results[workload].get(category, 0) for workload in workloads]
        
        # Calculate positions for bars of each category
        bar_positions = [pos + i * bar_width for pos in x]
        
        # Plot each category with an offset to group by workload
        bars = plt.bar(bar_positions, percentages, width=bar_width, label=category)

        # Add annotations on top of each bar
        for bar, percentage in zip(bars, percentages):
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # Center of the bar
                bar.get_height(),  # Height of the bar
                f'{percentage:.2f}%',  # Text to display
                ha='center', va='bottom', fontsize=8  # Center align
            )

    # Customize plot
    plt.xlabel('Workload')
    plt.ylabel('Percentage (%)')
    plt.title('Instruction Type Distribution Across Workloads')
    plt.xticks([pos + (bar_width * (len(categories) / 2)) for pos in x], workloads, rotation=45)
    plt.legend(title="Instruction Types")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig('instruction_distribution_by_workload.png', bbox_inches='tight')
    plt.show()

# Example usage
base_dir = os.path.expanduser("~/SimTools/code_py")
results = process_output_folders(base_dir)
plot_results(results)
