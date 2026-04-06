import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

SEED = 16

def iterative_kmeans_clustering(csv_path, prompt_type, cutoff, min_in_cluster):
    # Load the data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return "Error: CSV file not found.", []

    # Filter the dataset to ignore concepts that don't match the prompt_type
    filtered_df = df[df["Prompt Type"] == prompt_type].copy()
    
    if filtered_df.empty:
        return "No data found for the specified prompt_type.", []

    # Extract the relevant columns 
    concepts = filtered_df["Concept"].tolist()
    layers = filtered_df["Optimal Layer"].to_numpy()

    # Embed each concept and normelize them
    print(f"Embedding {len(concepts)} concepts...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(concepts, normalize_embeddings=True) 

    # State tracking
    working_indices = list(range(len(concepts)))
    good_clusters = [] 

    print("Starting iterative clustering process...")
    
    # Iterate k
    k = 1
    while k <= len(working_indices):
        found_good_cluster_at_current_k = True
        
        while found_good_cluster_at_current_k:
            found_good_cluster_at_current_k = False
            current_n = len(working_indices)
            
            if current_n == 0 or k > current_n:
                break
                
            current_embeddings = embeddings[working_indices]
            current_layers = layers[working_indices]
            
            kmeans = KMeans(n_clusters=k, n_init='auto', random_state=SEED)
            labels = kmeans.fit_predict(current_embeddings)
            
            indices_to_remove_this_round = []
            
            # Evaluate each cluster
            for cluster_id in range(k):
                cluster_mask = (labels == cluster_id)
                cluster_size = np.sum(cluster_mask)
                
                if cluster_size >= min_in_cluster:
                    cluster_layers_subset = current_layers[cluster_mask]
                    
                    # Calculate the Standard Deviation of the Optimal Layers
                    cluster_std = np.std(cluster_layers_subset) 
                    
                    if cluster_std <= cutoff:
                        actual_indices = np.array(working_indices)[cluster_mask].tolist()
                        
                        # Save both the indices and the standard deviation
                        good_clusters.append((actual_indices, cluster_std))
                        indices_to_remove_this_round.extend(actual_indices)
                        found_good_cluster_at_current_k = True
            
            # Remove good clusters from the working set
            if found_good_cluster_at_current_k:
                working_indices = [idx for idx in working_indices if idx not in indices_to_remove_this_round]
                print(f"  [k={k}] Found good cluster(s). Removed {len(indices_to_remove_this_round)} concepts. {len(working_indices)} remaining.")

        k += 1
        
        if len(working_indices) < min_in_cluster:
            break

    # Format the outputs
    final_good_clusters = []
    for cluster_indices, cluster_std in good_clusters:
        cluster_items = [{"concept": concepts[i], "layer": layers[i]} for i in cluster_indices]
        final_good_clusters.append({
            "items": cluster_items,
            "std": cluster_std
        })
        
    leftover_items = [{"concept": concepts[i], "layer": layers[i]} for i in working_indices]
    
    return final_good_clusters, leftover_items


if __name__ == "__main__":
    # Define parameters
    CSV_FILE = "..\\results\\part2_ultimate_final(improved_score)\\bert_scores.csv"
    PROMPT_TYPE_TO_KEEP = 2
    STD_CUTOFF = 1
    MINIMUM_IN_CLUSTER = 3
    
    print(f"running on seed {SEED}")
    # Run the algorithm
    good_clusters_out, left_over_out = iterative_kmeans_clustering(
        csv_path=CSV_FILE,
        prompt_type=PROMPT_TYPE_TO_KEEP,
        cutoff=STD_CUTOFF,
        min_in_cluster=MINIMUM_IN_CLUSTER
    )
    
    # Print Results
    if isinstance(good_clusters_out, str):
        print(good_clusters_out) 
    else:
        print("\n=== RESULTS ===")
        print(f"Total Good Clusters Found: {len(good_clusters_out)}")
        for i, cluster_data in enumerate(good_clusters_out):
            items = cluster_data["items"]
            std_dev = cluster_data["std"]
            
            print(f"\nCluster {i+1} (Size: {len(items)}, Std Dev: {std_dev:.4f}):")
            for item in items:
                print(f"  - {item['concept']} (Layer: {item['layer']})")
                
        print(f"\nTotal Concepts Left Over: {len(left_over_out)}")