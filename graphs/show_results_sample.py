import pandas as pd
import sys

def print_concept_stats(csv_filepath, concepts_list, prompt_type):
    """
    Reads a CSV file, filters by a list of concepts and a specific prompt type,
    drops the Coefficient and sentence example columns, and prints the stats.
    """
    try:
        # Load the data
        df = pd.read_csv(csv_filepath)
        
        # Filter rows by both the Concepts list AND the Prompt Type
        filtered_df = df[
            (df['Concept'].isin(concepts_list)) & 
            (df['Prompt Type'] == prompt_type)
        ]
        
        # Drop the columns we want to hide
        columns_to_drop = ['Coefficient', 'Random Completion']
        filtered_df = filtered_df.drop(columns=columns_to_drop, errors='ignore')
            
        # Check if we actually found any matches after filtering
        if filtered_df.empty:
            print(f"No matching records found for the given concepts and Prompt Type {prompt_type}.")
            return

        # Print the data as a clean Markdown table
        print(filtered_df.to_markdown(index=False))
        
    except FileNotFoundError:
        print(f"Error: Could not find the file at '{csv_filepath}'. Please check the path.")
    except Exception as e:
        print(f"An error occurred in print_concept_stats: {e}")

def calculate_final_score_std(csv_filepath, concepts_list, prompt_type):
    """
    Calculates the standard deviation of the 'Final Score' for a given list of concepts
    and a specific prompt type.
    """
    try:
        # Load the data
        df = pd.read_csv(csv_filepath)
        
        # Filter rows by both the Concepts list AND the Prompt Type
        filtered_df = df[
            (df['Concept'].isin(concepts_list)) & 
            (df['Prompt Type'] == prompt_type)
        ]
        
        if filtered_df.empty:
            print(f"No matching records found to calculate STD for Prompt Type {prompt_type}.")
            return None
            
        # Calculate the standard deviation of the Final Score
        std_value = filtered_df['Final Score'].std()
        
        print(f"\nStandard Deviation of Final Score for Prompt Type {prompt_type}: {std_value:.6f}")
        return std_value
        
    except Exception as e:
        print(f"An error occurred in calculate_final_score_std: {e}")
        return None

if __name__ == "__main__":
    # The specific CSV file path
    file_path = "..\\results\part2_ultimate_final(improved_score)\\bert_scores.csv" 
    
    # The specific list of concepts
    # target_concepts = [
    #     "Capitalism", 
    #     "Democracy", 
    #     "Dictatorship", 
    #     "Man vs. Nature", 
    #     "Socialism", 
    #     "The Black Death", 
    #     "Tidiness", 
    #     "Violence", 
    #     "War", 
    #     "War and Peace", 
    #     "Military Conflict"
    # ]

    # target_concepts = [
    #     "Capitalism", 
    #     "Democracy", 
    #     "Dictatorship", 
    #     "Socialism", 
    #     "Violence", 
    #     "War", 
    # ]

    # target_concepts = ["The Civil War", "The Cold War", "The Gulf War", "The Vietnam War", "World War 1", "World War 2"]

    # target_concepts = ["Television", "Science Fiction", "Aliens", "Film Festival", "Zombies", "Television Station", "Music Genre"]

    # target_concepts = ["Washington", "Democracy", "Prime Minister", "Economist", "Gandhi", "Socialism", "Congressman"]

    # target_concepts = [
    # "Washington",
    # "Democracy",
    # "Prime Minister",
    # "Economist",
    # "Gandhi",
    # "Socialism",
    # "Congressman",
    # "Capitalism",
    # "Dictatorship",
    # "Violence",
    # "War",
    # ]

    target_concepts = ["Ambassador", "Governor", "Mayor", "Pope", "President", "Prime Minister"]

    # target_concepts = ["Buildings", "Square", "Castle", "Glacier", "Lighthouse", "Mountain", "Museum", "Restaurant", "Volcano"]

    
    # Grab the Prompt Type from the command line argument (e.g., python script.py 1)
    # If no argument is provided, it defaults to Prompt Type 1
    try:
        target_prompt_type = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    except ValueError:
        print("Please provide a valid number for the Prompt Type.")
        sys.exit(1)
    
    # Run the functions
    print(f"--- Stats for Prompt Type {target_prompt_type} ---")
    print_concept_stats(file_path, target_concepts, target_prompt_type)
    
    calculate_final_score_std(file_path, target_concepts, target_prompt_type)