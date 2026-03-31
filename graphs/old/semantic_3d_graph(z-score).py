# %%
import pandas as pd
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
            
    return embeddings

def plot_semantic_landscape(df, prompt_type):
    """
    Embeds concepts using BERT, reduces to 2D via PCA.
    Plots a 3D graph where X/Y are semantic meaning, Z is Final Score, 
    and Color is Optimal Layer.
    """
    # 1. Filter the dataframe and drop NaNs in critical columns
    filtered_df = df[df['Prompt Type'] == prompt_type].copy()
    
    if filtered_df.empty:
        print(f"No data found for Prompt Type {prompt_type}")
        return
        
    filtered_df = filtered_df.dropna(subset=['Concept', 'Final Score', 'Optimal Layer'])
    concepts = filtered_df['Concept'].tolist()
    
    # 2. Get high-dimensional BERT embeddings
    embeddings = get_bert_embeddings(concepts)
    
    # 3. Run PCA to crunch it down to exactly 2 dimensions
    print("Running PCA to reduce dimensions to 2...")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Add the 2D coordinates back into our dataframe
    filtered_df['PCA_X'] = embeddings_2d[:, 0]
    filtered_df['PCA_Y'] = embeddings_2d[:, 1]
    
    # 4. Create the 3D Plotly Landscape Graph
    print("Generating 3D interactive landscape...")
    
    fig = px.scatter_3d(
        filtered_df, 
        x='PCA_X',               # Semantic Dimension 1
        y='PCA_Y',               # Semantic Dimension 2
        z='Final Score',         # Height is now the score!
        color='Optimal Layer',   # Color is now the layer
        color_continuous_scale='Bluered', # Blue = early layers, Red = deep layers
        text='Concept',          
        title=f'Semantic Landscape: Score & Optimal Layer (Prompt Type {prompt_type})'
    )
    
    # Styling to make it readable
    fig.update_traces(
        marker=dict(size=7, line=dict(width=1, color='DarkSlateGrey')),
        textposition='top center',
        textfont=dict(size=10, color='black')
    )
    
    # Clean up the axis titles to reflect what they represent
    fig.update_layout(
        scene=dict(
            xaxis_title='Semantic Axis X',
            yaxis_title='Semantic Axis Y',
            zaxis_title='Final Score'
        ),
        margin=dict(l=0, r=0, b=0, t=40) 
    )
    
    # Render the graph
    fig.show()

# ==========================================
# Example usage:
# ==========================================
df = load_data('results\part2_final(100)\\bert_scores.csv')
if df is not None:
    plot_semantic_landscape(df, prompt_type=1)