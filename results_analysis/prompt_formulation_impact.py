import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import scipy.stats as stats

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


def plot_type1_vs_type2(df):

    df_type1 = df[df['Prompt Type'] == 1][['Concept', 'Final Score']]
    df_type2 = df[df['Prompt Type'] == 2][['Concept', 'Final Score']]
    merged_df = pd.merge(df_type1, df_type2, on='Concept', suffixes=('_type1', '_type2'))
    
  
    fig, ax = plt.subplots(figsize=(16, 9)) 
    
    min_val = min(merged_df['Final Score_type1'].min(), merged_df['Final Score_type2'].min())
    max_val = max(merged_df['Final Score_type1'].max(), merged_df['Final Score_type2'].max())
    
    padding = (max_val - min_val) * 0.05
    min_val -= padding
    max_val += padding
    total_width = max_val - min_val

    ax.scatter(
        merged_df['Final Score_type1'], merged_df['Final Score_type2'], 
        color='steelblue', edgecolor='black', alpha=0.8, s=50                 
    )
    ax.set_ylim(min_val, max_val)
    
    data_range = max_val - min_val
    mid_point = (max_val + min_val) / 2
    aspect_ratio = 16 / 9  
    x_range_wide = data_range * aspect_ratio
    
    ax.set_xlim(mid_point - (x_range_wide / 2), mid_point + (x_range_wide / 2))
    
    ax.set_aspect('equal', adjustable='box')
    
    step_size = data_range / 12  
    ax.xaxis.set_major_locator(MultipleLocator(step_size))
    ax.yaxis.set_major_locator(MultipleLocator(step_size))

    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='x', rotation=45)

    ax.plot(
        [mid_point - (x_range_wide / 2), mid_point + (x_range_wide / 2)], 
        [mid_point - (x_range_wide / 2), mid_point + (x_range_wide / 2)], 
        color='red', linestyle='--', alpha=0.5, label='Equal Score (y=x)'
    )

    ax.set_title('Prompt Type 1 vs. Prompt Type 2', fontsize=16, fontweight='bold')
    ax.set_xlabel('Score on Prompt Type 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score on Prompt Type 2', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')
    

    annotations = []
    for idx, row in merged_df.iterrows():
        ann = ax.annotate(
            row['Concept'], 
            (row['Final Score_type1'], row['Final Score_type2']),
            xytext=(6, 6), textcoords='offset points', 
            fontsize=11, 
            alpha=0.9, visible=False,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7) 
        )
        annotations.append(ann)


    zoom_threshold = total_width * 0.70

    def on_xlims_change(event_ax):
        x_min_current, x_max_current = event_ax.get_xlim()
        current_view_width = x_max_current - x_min_current
        should_be_visible = current_view_width < zoom_threshold
        for ann in annotations:
            ann.set_visible(should_be_visible)
            
    ax.callbacks.connect('xlim_changed', on_xlims_change)
    ax.callbacks.connect('ylim_changed', on_xlims_change)


    def zoom_factory(ax, base_scale=1.2):
        def zoom_fun(event):
            if event.xdata is None or event.ydata is None: 
                return

            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            if event.button == 'up':
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                scale_factor = base_scale
            else:
                scale_factor = 1

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([event.xdata - new_width * (1 - relx), event.xdata + new_width * (relx)])
            ax.set_ylim([event.ydata - new_height * (1 - rely), event.ydata + new_height * (rely)])
            
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('scroll_event', zoom_fun)

    zoom_factory(ax)

    plt.tight_layout()
    plt.show()


def print_regression_stats(df):
    """
    Calculates and prints the linear regression slope, Pearson r, 
    and p-value comparing Prompt Type 1 and Prompt Type 2 scores.
    """
    # Separate and merge the data
    df_type1 = df[df['Prompt Type'] == 1][['Concept', 'Final Score']]
    df_type2 = df[df['Prompt Type'] == 2][['Concept', 'Final Score']]
    
    # Check if we have enough data to compare
    if df_type1.empty or df_type2.empty:
        print("Error: Missing data for one or both prompt types.")
        return

    merged_df = pd.merge(df_type1, df_type2, on='Concept', suffixes=('_type1', '_type2'))
    
    x_data = merged_df['Final Score_type1']
    y_data = merged_df['Final Score_type2']

    # Calculate statistics
    slope, _ = np.polyfit(x_data, y_data, 1) 
    
    # Pearsonr returns (statistic, p-value)
    pearson_r, p_value = stats.pearsonr(x_data, y_data)
    
    print("\n" + "="*30)
    print("   REGRESSION STATISTICS")
    print("="*30)
    print(f"Slope:     {slope:.4f}")
    print(f"Pearson r: {pearson_r:.4f}")
    print(f"p-value:   {p_value:.4e}") 
    print("="*30 + "\n")
    
    return slope, pearson_r, p_value


file_name = '..\\results\concept_steerability_results\\bert_scores.csv'
results_df = load_data(file_name)

print_regression_stats(results_df)

plot_type1_vs_type2(results_df)


