# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
import warnings

# Suppress some common KMeans memory leak warnings on Windows
warnings.filterwarnings("ignore", category=UserWarning)

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

def get_bert_embeddings(concepts):
    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    model.eval() 
    
    embeddings = []
    print(f"Embedding {len(concepts)} concepts...")
    
    with torch.no_grad():
        for concept in concepts:
            inputs = tokenizer(concept, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            concept_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(concept_embedding)
            
    return np.array(embeddings) # Return as a clean numpy array

def kmeans_elbow_and_cluster(df, prompt_type, a_param):
    """
    Encodes concepts, appends a weighted score dimension, plots the K-Means 
    elbow curve, and asks the user for K to print the final clusters.
    """
    # 1. Filter and clean data
    filtered_df = df[df['Prompt Type'] == prompt_type].copy()
    if filtered_df.empty:
        print(f"No data found for Prompt Type {prompt_type}")
        return
        
    filtered_df = filtered_df.dropna(subset=['Concept', 'Final Score'])
    concepts = filtered_df['Concept'].tolist()
    scores = filtered_df['Final Score'].values
    
    # 2. Get BERT embeddings (usually 768 dimensions)
    embeddings = get_bert_embeddings(concepts)
    
    # 3. Calculate and append the new dimension
    num_dimensions = embeddings.shape[1]
    print(f"BERT embedding dimensions: {num_dimensions}")
    
    # Formula: sqrt(a * num_dimensions) * score
    weight_factor = np.sqrt(a_param * num_dimensions)
    weighted_scores = scores * weight_factor
    
    # Append the weighted score as a new column to the embeddings array
    # Now our data is 769 dimensions
    combined_features = np.column_stack((embeddings, weighted_scores))
    print(f"Combined feature matrix shape: {combined_features.shape}")
    
    # 4. Run K-Means for K = 1 to 30 to find the loss (inertia)
    # Note: If you have less than 30 concepts, we cap max_k to the number of concepts
    max_k = len(concepts)
    k_values = range(1, max_k + 1)
    inertias = []
    
    print(f"Running K-Means for K=1 to {max_k}...")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(combined_features)
        inertias.append(kmeans.inertia_)
        
    # 5. Plot the Elbow Curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, marker='o', linestyle='-', color='steelblue')
    plt.title(f'K-Means Elbow Method (Prompt Type {prompt_type}, a={a_param})', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    plt.ylabel('Loss (Inertia)', fontsize=12, fontweight='bold')
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # In VS Code, this will display the plot inline.
    plt.show() 
    
    # 6. Ask the user for their chosen K
    # Look at the top center of VS Code (or your terminal) for the input box!
    while True:
        try:
            chosen_k_str = input(f"\nLook at the graph above. Enter your chosen K (between 1 and {max_k}): ")
            chosen_k = int(chosen_k_str)
            if 1 <= chosen_k <= max_k:
                break
            else:
                print(f"Please enter a number between 1 and {max_k}.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")
            
    # 7. Rerun K-Means with the chosen K
    print(f"\nRunning final K-Means with K={chosen_k}...")
    final_kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    labels = final_kmeans.fit_predict(combined_features)
    
    # 8. Group and print the clusters
    clusters = {i: [] for i in range(chosen_k)}
    for concept, label in zip(concepts, labels):
        clusters[label].append(concept)
        
    print("\n" + "="*50)
    print(f"FINAL CLUSTERS (K={chosen_k})")
    print("="*50)
    for i in range(chosen_k):
        # Join the list into a clean comma-separated string
        cluster_items = ", ".join(clusters[i])
        print(f"Cluster {i + 1}: [{cluster_items}]\n")
def kmeans_semantic_score_variance(df, prompt_type, k):
    """
    Runs K-Means clustering solely on BERT embeddings for a given prompt type.
    Prints the variance of the Final Scores within each cluster, 
    along with the concepts that belong to it.
    """
    # 1. Filter and clean data
    filtered_df = df[df['Prompt Type'] == prompt_type].copy()
    if filtered_df.empty:
        print(f"No data found for Prompt Type {prompt_type}")
        return
        
    filtered_df = filtered_df.dropna(subset=['Concept', 'Final Score'])
    concepts = filtered_df['Concept'].tolist()
    scores = filtered_df['Final Score'].values
    
    # 2. Get standard BERT embeddings (pure semantic meaning, no scores attached)
    embeddings = get_bert_embeddings(concepts)
    
    # 3. Run K-Means on the pure embeddings
    print(f"\nRunning K-Means with K={k} on pure BERT embeddings...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # 4. Group concepts and scores by their new cluster labels
    clusters = {i: {'concepts': [], 'scores': []} for i in range(k)}
    
    for concept, score, label in zip(concepts, scores, labels):
        clusters[label]['concepts'].append(concept)
        clusters[label]['scores'].append(score)
        
    # 5. Calculate variance and print the results
    print("\n" + "="*60)
    print(f"SEMANTIC CLUSTER SCORE VARIANCES (Prompt Type {prompt_type}, K={k})")
    print("="*60)
    
    for i in range(k):
        cluster_concepts = clusters[i]['concepts']
        cluster_scores = clusters[i]['scores']
        
        # Calculate variance. (If only 1 concept is in a cluster, variance is 0)
        if len(cluster_scores) > 1:
            variance = np.var(cluster_scores) 
        else:
            variance = 0.0
            
        # Join the list into a clean comma-separated string
        concept_list_str = ", ".join(cluster_concepts)
        
        # Print the metrics clearly
        print(f"Cluster {i + 1} | Size: {len(cluster_concepts)} | Score Variance: {variance:.6f}")
        print(f"Concepts: [{concept_list_str}]\n")



def kmeans_semantic_score_std(df, prompt_type, k):
    """
    Runs K-Means clustering solely on BERT embeddings for a given prompt type.
    Prints the standard deviation of the Final Scores within each cluster, 
    along with the concepts that belong to it.
    """
    # 1. Filter and clean data
    filtered_df = df[df['Prompt Type'] == prompt_type].copy()
    if filtered_df.empty:
        print(f"No data found for Prompt Type {prompt_type}")
        return
        
    filtered_df = filtered_df.dropna(subset=['Concept', 'Final Score'])
    concepts = filtered_df['Concept'].tolist()
    scores = filtered_df['Final Score'].values
    
    # 2. Get standard BERT embeddings (pure semantic meaning, no scores attached)
    embeddings = get_bert_embeddings(concepts)
    
    # 3. Run K-Means on the pure embeddings
    print(f"\nRunning K-Means with K={k} on pure BERT embeddings...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # 4. Group concepts and scores by their new cluster labels
    clusters = {i: {'concepts': [], 'scores': []} for i in range(k)}
    
    for concept, score, label in zip(concepts, scores, labels):
        clusters[label]['concepts'].append(concept)
        clusters[label]['scores'].append(score)
        
    # 5. Calculate standard deviation and print the results
    print("\n" + "="*65)
    print(f"SEMANTIC CLUSTER SCORE STD DEVIATIONS (Prompt Type {prompt_type}, K={k})")
    print("="*65)
    
    for i in range(k):
        cluster_concepts = clusters[i]['concepts']
        cluster_scores = clusters[i]['scores']
        
        # Calculate standard deviation. (If only 1 concept is in a cluster, std is 0)
        if len(cluster_scores) > 1:
            std_dev = np.std(cluster_scores) 
        else:
            std_dev = 0.0
            
        # Join the list into a clean comma-separated string
        concept_list_str = ", ".join(cluster_concepts)
        
        # Print the metrics clearly
        print(f"Cluster {i + 1} | Size: {len(cluster_concepts)} | Score Std Dev: {std_dev:.6f}")
        print(f"Concepts: [{concept_list_str}]\n")
# ==========================================
# Example usage:
# ==========================================
df = load_data('..\\results\part2_ultimate_final(improved_score)\\bert_scores.csv' )
# if df is not None:
#     # We pass a_param=1.0 as an example. Adjust it to give the score more or less weight!
#     kmeans_elbow_and_cluster(df, prompt_type=1, a_param=1)

# if df is not None:
#     kmeans_semantic_score_variance(df, prompt_type=1, k=30)


if df is not None:
    kmeans_semantic_score_std(df, prompt_type=1, k=1)