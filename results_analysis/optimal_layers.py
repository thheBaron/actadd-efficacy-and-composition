import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    """
    Reads the CSV file and returns a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None



def plot_layer_slices(df, prompt_type):
    """
    Creates a 2D grid divided into colored vertical slices representing Optimal Layers.
    Only layers present in the filtered data will get a slice.
    Concepts are plotted inside their respective slice based on their Final Score.
    """
    # 1. Filter by prompt type and drop any rows missing an Optimal Layer
    filtered_df = df[df['Prompt Type'] == prompt_type].dropna(subset=['Optimal Layer']).copy()
    
    if filtered_df.empty:
        print(f"No data found for Prompt Type {prompt_type}")
        return

    # 2. Find only the layers that actually exist in this subset, and sort them
    unique_layers = sorted(filtered_df['Optimal Layer'].unique())
    num_slices = len(unique_layers)
    
    # Map each actual layer to an X-coordinate (0, 1, 2, 3...)
    layer_to_x = {layer: i for i, layer in enumerate(unique_layers)}
    
    # 3. Set up the wide figure
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Get a good colormap with enough distinct colors for the slices (Pastel tones look great as backgrounds)
    cmap = plt.get_cmap('Pastel2') 

    # 4. Draw the colored background slices
    for i, layer in enumerate(unique_layers):
        # Calculate left and right edges of this specific slice (width of 1)
        left_edge = i - 0.5
        right_edge = i + 0.5
        
        # Pick a color from the colormap
        slice_color = cmap(i % cmap.N) 
        
        # Draw the colored rectangle spanning the whole vertical height
        ax.axvspan(left_edge, right_edge, color=slice_color, alpha=0.4)
        
        # Draw a vertical dividing line between slices
        if i > 0:
            ax.axvline(left_edge, color='gray', linestyle='--', alpha=0.5)

    # 5. Plot the concepts inside their slices
    # We add a tiny bit of random horizontal "jitter" so dots don't overlap perfectly
    np.random.seed(42) # Keep jitter consistent 
    
    for idx, row in filtered_df.iterrows():
        base_x = layer_to_x[row['Optimal Layer']]
        jitter = np.random.uniform(-0.25, 0.25) # Shift slightly left or right
        x_pos = base_x + jitter
        y_pos = row['Final Score']
        concept = row['Concept']
        
        # Draw the dot
        ax.scatter(x_pos, y_pos, color='steelblue', edgecolor='black', s=50, zorder=3)
        
        # Draw the concept label next to the dot
        ax.annotate(
            concept, 
            (x_pos, y_pos),
            xytext=(5, 0), 
            textcoords='offset points', 
            fontsize=9,
            va='center'
        )

    # 6. Format the X-axis to show the actual Optimal Layer numbers
    ax.set_xticks(range(num_slices))
    # Convert layer numbers to integers for clean text
    ax.set_xticklabels([f"Layer {int(layer)}" for layer in unique_layers], fontsize=12, fontweight='bold')

    # Lock the X limits so the graph ends cleanly at the edges of the first and last slices
    ax.set_xlim(-0.5, num_slices - 0.5)

    # 7. Labels and Titles
    ax.set_title(f'Concept Distributions by Optimal Layer (Prompt Type {prompt_type})', fontsize=16, fontweight='bold')
    ax.set_ylabel('Final Score', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()



def plot_concept_counts_per_layer(df, prompt_type):
    # 0. Update global plot aesthetics
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9
    })

    min_layer = int(df['Optimal Layer'].dropna().min())
    max_layer = int(df['Optimal Layer'].dropna().max())
    all_layers = range(min_layer, max_layer + 1)

    filtered_df = df[df['Prompt Type'] == prompt_type].dropna(subset=['Optimal Layer']).copy()
    
    if filtered_df.empty:
        print(f"No data found for Prompt Type {prompt_type}")
        return

    layer_counts = filtered_df.groupby('Optimal Layer').size()
    layer_counts = layer_counts.reindex(all_layers, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    
    # Plot the bars
    ax.bar(layer_counts.index, layer_counts.values, color='teal', edgecolor='black', zorder=3, width=0.8)
    
    # Only put a tick mark and label every 5 layers to stop overlapping
    tick_positions = np.arange(0, max_layer + 1, 5)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(pos) for pos in tick_positions])
    
    # Labels
    ax.set_xlabel('Optimal Layer')
    ax.set_ylabel('Number of Concepts')
    
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    plt.tight_layout()
    plt.savefig(f"concept_count_per_layer_{prompt_type}.pdf", format='pdf', bbox_inches='tight')
    print(f"Plot saved successfully")
    plt.close(fig)

file_name = '..\\results\concept_steerability_results\\bert_scores.csv' 

# Load the data
results_df = load_data(file_name)
prompt_type=2

plot_concept_counts_per_layer(results_df, prompt_type)
plot_layer_slices(results_df, prompt_type)