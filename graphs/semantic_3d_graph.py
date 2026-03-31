# %%
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import plotly.express as px

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

def get_bert_embeddings(concepts):
    """
    Takes a list of concept strings and returns their BERT embeddings.
    """
    print("Loading BERT model (this might take a few seconds)...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    model.eval() # Set to evaluation mode for faster, deterministic inference
    
    embeddings = []
    print(f"Embedding {len(concepts)} concepts...")
    
    with torch.no_grad():
        for concept in concepts:
            # Tokenize the concept
            inputs = tokenizer(concept, return_tensors="pt", padding=True, truncation=True)
            # Pass through BERT
            outputs = model(**inputs)
            # Take the mean of the last hidden state to represent the whole concept word/phrase
            concept_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(concept_embedding)
            
    return embeddings

def plot_3d_semantic_scores(df, prompt_type):
    """
    Filters data, embeds concepts using BERT, reduces to 3D via PCA, 
    and plots them interactively using Plotly.
    """
    # 1. Filter the dataframe
    filtered_df = df[df['Prompt Type'] == prompt_type].copy()
    
    if filtered_df.empty:
        print(f"No data found for Prompt Type {prompt_type}")
        return
        
    # Drop rows where Final Score or Concept might be missing just to be safe
    filtered_df = filtered_df.dropna(subset=['Concept', 'Final Score'])
    concepts = filtered_df['Concept'].tolist()
    
    # 2. Get high-dimensional BERT embeddings
    embeddings = get_bert_embeddings(concepts)
    
    # 3. Run PCA to crunch it down to 3 dimensions
    print("Running PCA to reduce dimensions to 3...")
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    
    # Add the new 3D coordinates back into our dataframe
    filtered_df['PCA_X'] = embeddings_3d[:, 0]
    filtered_df['PCA_Y'] = embeddings_3d[:, 1]
    filtered_df['PCA_Z'] = embeddings_3d[:, 2]
    
    # 4. Create the 3D Plotly Graph
    print("Generating 3D interactive plot...")
    
    fig = px.scatter_3d(
        filtered_df, 
        x='PCA_X', 
        y='PCA_Y', 
        z='PCA_Z',
        color='Final Score',
        color_continuous_scale='Bluered', # The obvious blue-to-red scale!
        text='Concept',                   # Puts the label directly on the point
        title=f'3D Semantic Space vs. Final Score (Prompt Type {prompt_type})'
    )
    
    # Make the dots a bit larger and position the text so it's readable
    fig.update_traces(
        marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')),
        textposition='top center',
        textfont=dict(size=10, color='black')
    )
    
    # Tweak the layout for a cleaner look
    fig.update_layout(
        scene=dict(
            xaxis_title='PCA Dimension 1',
            yaxis_title='PCA Dimension 2',
            zaxis_title='PCA Dimension 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40) # Removes wasted white space around the 3D box
    )
    
    # Render the graph
    fig.show()

# ==========================================
# Example usage:
# ==========================================
df = load_data('..\\results\part2_ultimate_final(improved_score)\\bert_scores.csv')
if df is not None:
    plot_3d_semantic_scores(df, prompt_type=1)