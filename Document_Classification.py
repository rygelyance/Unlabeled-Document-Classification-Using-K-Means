# Imports
from sentence_transformers import SentenceTransformer # For word embeddings model
from sklearn.metrics import confusion_matrix # For accuracy reading
from sklearn.metrics import pairwise_distances_argmin # For implementing K-Means
from sklearn.utils import check_random_state # For MaxMin centroid initialization
from scipy.optimize import linear_sum_assignment # For accuracy reading 
import pandas as pd
import numpy as np

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create the dataframe and list of documents from the original CSV (dataset obtained from Kaggle, https://www.kaggle.com/datasets/sunilthite/text-document-classification-dataset)
df = pd.read_csv("df_file.csv")
docs = df["Text"].tolist()

# Using a pretrained Sentence Transformer Model to gather word embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs, show_progress_bar=True)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Initializes the cluster centroids for K-Means using the Farthest-Point Heuristic (MaxMin)
def maxmin_init(X, n_clusters, random_state=None):
    rng = check_random_state(random_state)
    n_samples, _ = X.shape
    
    # Choose the first centroid randomly
    first_idx = rng.randint(n_samples)
    centers = [X[first_idx]]

    # Select the rest using furthest point heuristic (the other two centroids should be as far as possible from the first randomly selected one)
    for _ in range(1, n_clusters):
        distances = np.min(np.linalg.norm(X[:, None] - np.array(centers), axis=2), axis=1)
        next_idx = np.argmax(distances)
        centers.append(X[next_idx])

    return np.array(centers)

# The K-Means algorithm, run using the centroids initialized using MaxMin
def kmeans_maxmin(X, n_clusters, n_init_custom, max_iter, random_state=None):
    best_inertia = np.inf
    best_labels = None
    best_centers = None

    rng = check_random_state(random_state)

    for run in range(n_init_custom):
        centers = maxmin_init(X, n_clusters, random_state=rng)
        
        for _ in range(max_iter):
            # Assign labels based on closest center
            labels = pairwise_distances_argmin(X, centers)
            
            # Compute new centers
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
            
            # Check for convergence
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        
        # Compute inertia (within-cluster sum of squared distances between each point and the centroid of the cluster it belongs to)
        # Can also be thought of of how "compact" the cluster are, the more compact, the more distinct/better the clusters are and the better the fit (generally, there are caveats to this)
        inertia = np.sum((X - centers[labels]) ** 2)
        
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centers = centers

    return best_labels, best_centers, best_inertia

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Using the k-means with repeated maxmin random starts for the best accuracy
labels, centers, inertia = kmeans_maxmin(
    embeddings, 
    n_clusters=5, # In this case, I know the data is supposed to have five clusters, if I did not know this, I'd use either the Elbow or Silhouette Methods to find the optimal K
    n_init_custom=1, # Try 100 random starts with MaxMin
    max_iter=300, 
    random_state=None
)

# Attach cluster labels
df["cluster"] = labels

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Measuring the accuracy of the predicted labels against the true labels from this dataset (5 categories, Politics, Sport, Technology, Entertainment, and Business)
# True labels were NOT used in the process of classifying these documents, this is to demonstrate an unsupervised learning problem!

true_labels = df["Label"].to_numpy()
cluster_labels = df["cluster"].to_numpy()

# Build confusion matrix
cm = confusion_matrix(true_labels, cluster_labels)

# Hungarian algorithm to find best label mapping
row_ind, col_ind = linear_sum_assignment(-cm)

# Map cluster labels to true labels
mapping = dict(zip(col_ind, row_ind))
mapped_clusters = np.array([mapping[c] for c in cluster_labels])

# Compute accuracy
accuracy = (mapped_clusters == true_labels).mean()
print(f"Accuracy: {accuracy:.4f}")