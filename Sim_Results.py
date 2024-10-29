import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pandas.plotting import table

# Function to extract values from stats.txt
def extract_stats_values(stats_file):
    with open(stats_file, 'r') as file:
        data = file.read()
    sim_seconds = float(re.search(r'simSeconds\s+([\d.]+)', data).group(1))
    cpi = float(re.search(r'system\.cpu\.cpi\s+([\d.]+)', data).group(1))
    return sim_seconds, cpi

# Function to extract values from mcpat.txt
def extract_mcpat_values(mcpat_file):
    with open(mcpat_file, 'r') as file:
        data = file.read()
    total_leakage = float(re.search(r'Total Leakage\s+=\s+([\d.]+)', data).group(1))
    runtime_dynamic = float(re.search(r'Runtime Dynamic\s+=\s+([\d.]+)', data).group(1))
    return total_leakage, runtime_dynamic

# Function to calculate metrics
def calculate_metrics(sim_seconds, cpi, total_leakage, runtime_dynamic):
    energy = (total_leakage + runtime_dynamic) * cpi
    edp = energy * cpi
    ipc = 1 / cpi
    performance = 1 / sim_seconds
    return energy, edp, ipc, performance

# Function to process simulation output folders and generate metrics
def process_simulations(base_dir):
    results = []
    cnt_simulations = 0 # Counter for the number of different simulations: e.g., 0 and 1 for h264_dec_0 and h264_dec_1
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        stats_file = os.path.join(folder_path, 'stats.txt')
        mcpat_file = os.path.join(folder_path, 'mcpat.txt')

        if os.path.exists(stats_file) and os.path.exists(mcpat_file):
            sim_seconds, cpi = extract_stats_values(stats_file)
            total_leakage, runtime_dynamic = extract_mcpat_values(mcpat_file)
            energy, edp, ipc, performance = calculate_metrics(sim_seconds, cpi, total_leakage, runtime_dynamic)

            # Clean workload name to differentiate output folder simulations
            match = re.match(r'([a-zA-Z0-9_]+)_simout__?(\d+)', folder)
            if match:
                # Format to show base name and number as desired
                workload_name = f"{match.group(1)}_{match.group(2)}"  # e.g., "h264_dec_1"
            else:
                workload_name = folder  # Fallback if no match
            
            if (cnt_simulations >= int(match.group(2))):
                cnt_simulations = int(match.group(2)) + 1

            results.append((workload_name, round(energy, 4), round(edp, 4), round(ipc, 4), round(performance, 4)))

    # Convert results to DataFrame for easier handling
    df = pd.DataFrame(results, columns=['Workload', 'Energy (J)', 'EDP', 'IPC', 'Performance'])
    
    # Compute averages for energy and performance grouped by simulation output (0 and 1)
    avg_results = []
    for sim_output in range(cnt_simulations):  # Assuming two simulation outputs: 0 and 1
        avg_energy = df[df['Workload'].str.endswith(f'_{sim_output}')]['Energy (J)'].mean()
        avg_performance = df[df['Workload'].str.endswith(f'_{sim_output}')]['Performance'].mean()
        avg_ipc = df[df['Workload'].str.endswith(f'_{sim_output}')]['IPC'].mean()
        avg_edp = df[df['Workload'].str.endswith(f'_{sim_output}')]['EDP'].mean()
        avg_results.append((f'Average_{sim_output}', round(avg_energy, 4), round(avg_edp, 4), round(avg_ipc, 4), round(avg_performance, 4)))
    
    # Append average results to the DataFrame
    avg_df = pd.DataFrame(avg_results, columns=['Workload', 'Energy (J)', 'EDP', 'IPC', 'Performance'])
    df = pd.concat([df, avg_df], ignore_index=True)

    # Print the average results from general df
    print("Average results:")
    print(df[df['Workload'].str.startswith('Average')])

    return df

# Function to generate summary table image with four decimal precision
def generate_summary_table(df):
    df = df.sort_values(by='Performance', ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.5))
    ax.axis('tight')
    ax.axis('off')
    tbl = table(ax, df, loc='center', cellLoc='center', colWidths=[0.15]*len(df.columns))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)
    plt.savefig("simulation_summary_table.png")
    plt.close()
    Image.open("simulation_summary_table.png").show()

# Function to add value labels to bars
def add_value_labels(ax):
    for p in ax.patches:
        ax.annotate(f'{p.get_width():.4f}', (p.get_width(), p.get_y() + p.get_height() / 2),
                    ha='left', va='center', xytext=(5, 0), textcoords='offset points')

# Function to plot top N simulations for Performance and IPC
def plot_performance_ipc(df, top_n=5):
    top_performance = df.nlargest(top_n, 'Performance')
    top_ipc = df.nlargest(top_n, 'IPC')

    fig, ax = plt.subplots(figsize=(12, 8))

    # Set colors for clarity
    colors = ['skyblue', 'salmon']
    
    # Set bar width and positions
    bar_width = 0.35
    positions_performance = range(top_n)
    positions_ipc = [pos + bar_width for pos in positions_performance]

    # Plot bars for performance and IPC
    bars_performance = ax.barh(positions_performance, top_performance['Performance'], height=bar_width, color=colors[0], label='Performance')
    bars_ipc = ax.barh(positions_ipc, top_ipc['IPC'], height=bar_width, color=colors[1], label='IPC')

    # Adding labels
    add_value_labels(ax)

    # Set y-ticks and labels
    ax.set_yticks([pos + bar_width / 2 for pos in positions_performance])
    ax.set_yticklabels(top_performance['Workload'])

    # Plot customizations
    plt.xlabel('Metric Value')
    plt.title(f'Top {top_n} Simulations by Performance and IPC')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("top_simulations_performance_ipc.png")
    plt.show()

# Function to plot top N simulations for Energy and EDP
def plot_energy_edp(df, top_n=5):
    top_energy = df.nsmallest(top_n, 'Energy (J)')
    top_edp = df.nsmallest(top_n, 'EDP')

    fig, ax = plt.subplots(figsize=(12, 8))

    # Set colors for clarity
    colors = ['lightgreen', 'gold']
    
    # Set bar width and positions
    bar_width = 0.35
    positions_energy = range(top_n)
    positions_edp = [pos + bar_width for pos in positions_energy]

    # Plot bars for energy and EDP
    bars_energy = ax.barh(positions_energy, top_energy['Energy (J)'], height=bar_width, color=colors[0], label='Energy (J)')
    bars_edp = ax.barh(positions_edp, top_edp['EDP'], height=bar_width, color=colors[1], label='EDP')

    # Adding labels
    add_value_labels(ax)

    # Set y-ticks and labels
    ax.set_yticks([pos + bar_width / 2 for pos in positions_energy])
    ax.set_yticklabels(top_energy['Workload'])

    # Plot customizations
    plt.xlabel('Metric Value')
    plt.title(f'Top {top_n} Simulations by Energy and EDP')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("top_simulations_energy_edp.png")
    plt.show()

# Function to plot Energy vs. Performance with enhancements
def plot_energy_vs_performance(df):
    plt.figure(figsize=(12, 8))
    
    # Remove rows with NaN or infinite values for the plot
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Energy (J)', 'Performance'])
    
    # Create scatter plot
    scatter = plt.scatter(df_clean['Energy (J)'], df_clean['Performance'], color='blue', alpha=0.6)

    # Adding annotations for each point
    for i in range(len(df_clean)):
        plt.annotate(df_clean['Workload'].iloc[i], (df_clean['Energy (J)'].iloc[i], df_clean['Performance'].iloc[i]),
                     textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8)

    # Calculate and plot average values if they exist
    avg_row = df_clean[df_clean['Workload'].str.startswith('Average')]
    if not avg_row.empty:
        avg_energy = avg_row['Energy (J)'].values[0]
        avg_performance = avg_row['Performance'].values[0]
        plt.scatter(avg_energy, avg_performance, color='red', marker='X', s=100, label='Average')

    # Add a trend line if there are sufficient points
    if len(df_clean) >= 2:  # Ensure there are at least 2 points for fitting
        z = np.polyfit(df_clean['Energy (J)'], df_clean['Performance'], 1)
        p = np.poly1d(z)
        plt.plot(df_clean['Energy (J)'], p(df_clean['Energy (J)']), color='orange', linestyle='--', label='Trend Line')

    # Customizations
    plt.title('Energy vs. Performance')
    plt.xlabel('Energy (J)')
    plt.ylabel('Performance (1/s)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("energy_vs_performance.png")
    plt.show()

# Function to plot Performance vs. EDP with enhancements
def plot_performance_vs_edp(df):
    plt.figure(figsize=(12, 8))
    
    # Remove rows with NaN or infinite values for the plot
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['EDP', 'Performance'])

    # Create scatter plot
    scatter = plt.scatter(df_clean['EDP'], df_clean['Performance'], color='orange', alpha=0.6)

    # Adding annotations for each point
    for i in range(len(df_clean)):
        plt.annotate(df_clean['Workload'].iloc[i], (df_clean['EDP'].iloc[i], df_clean['Performance'].iloc[i]),
                     textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8)

    # Calculate and plot average values if they exist
    avg_row = df_clean[df_clean['Workload'].str.startswith('Average')]
    if not avg_row.empty:
        avg_performance = avg_row['Performance'].values[0]
        avg_edp = avg_row['EDP'].values[0]
        print(f'Average Performance: {avg_performance:.4f}  Average EDP: {avg_edp:.4f}')
        plt.scatter(avg_edp, avg_performance, color='purple', marker='X', s=100, label='Average', edgecolor='black')

    # Add a trend line if there are sufficient points
    if len(df_clean) >= 2:  # Ensure there are at least 2 points for fitting
        z = np.polyfit(df_clean['EDP'], df_clean['Performance'], 1)
        p = np.poly1d(z)
        plt.plot(df_clean['EDP'], p(df_clean['EDP']), color='red', linestyle='--', label='Trend Line')

    # Customizations
    plt.title('Performance vs. EDP for Simulations', fontsize=14)
    plt.xlabel('EDP (J·s)', fontsize=12)
    plt.ylabel('Performance (1/s)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig("performance_vs_edp.png")
    plt.show()

# Main execution
if __name__ == "__main__":
    base_dir = os.path.expanduser("~/SimTools/code_py")  # Replace with your actual path
    df = process_simulations(base_dir)
    generate_summary_table(df)
    # plot_performance_ipc(df, top_n=5)
    # plot_energy_edp(df, top_n=5)
    plot_energy_vs_performance(df)
    plot_performance_vs_edp(df)
