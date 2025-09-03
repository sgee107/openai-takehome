"""
Semantic quality metrics for embeddings.

Moved from app/experiments/metrics/semantic.py and adapted for the new structure.
"""
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


async def calculate_semantic_metrics(
    embeddings: np.ndarray,
    labels: List[str],
    strategy_name: str
) -> Dict[str, Any]:
    """
    Calculate semantic quality metrics for embeddings.
    
    Args:
        embeddings: Numpy array of embeddings (n_samples, n_features)
        labels: List of category labels for each embedding
        strategy_name: Name of the strategy being evaluated
    
    Returns:
        Dictionary of semantic metrics
    """
    
    metrics = {
        "strategy": strategy_name,
        "num_embeddings": len(embeddings)
    }
    
    # Skip if not enough embeddings
    if len(embeddings) < 2:
        return metrics
    
    # 1. Calculate intra-category similarity
    category_similarities = {}
    unique_categories = list(set(labels))
    
    for category in unique_categories:
        # Get indices for this category
        category_indices = [i for i, l in enumerate(labels) if l == category]
        
        if len(category_indices) > 1:
            # Get embeddings for this category
            category_embeddings = embeddings[category_indices]
            
            # Calculate pairwise cosine similarity
            similarity_matrix = cosine_similarity(category_embeddings)
            
            # Get upper triangle (excluding diagonal)
            upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
            similarities = similarity_matrix[upper_triangle_indices]
            
            # Calculate statistics
            category_similarities[category] = {
                "mean": float(np.mean(similarities)),
                "std": float(np.std(similarities)),
                "min": float(np.min(similarities)),
                "max": float(np.max(similarities)),
                "count": len(category_indices)
            }
    
    metrics["category_similarities"] = category_similarities
    
    # Calculate overall intra-category similarity
    all_intra_similarities = []
    for cat_data in category_similarities.values():
        all_intra_similarities.extend([cat_data["mean"]] * cat_data["count"])
    
    if all_intra_similarities:
        metrics["avg_intra_category_similarity"] = float(np.mean(all_intra_similarities))
    
    # 2. Calculate inter-category distinction
    if len(unique_categories) > 1:
        inter_similarities = []
        
        for i, cat1 in enumerate(unique_categories):
            for cat2 in unique_categories[i+1:]:
                # Get embeddings for each category
                cat1_indices = [idx for idx, l in enumerate(labels) if l == cat1]
                cat2_indices = [idx for idx, l in enumerate(labels) if l == cat2]
                
                if cat1_indices and cat2_indices:
                    cat1_embeddings = embeddings[cat1_indices]
                    cat2_embeddings = embeddings[cat2_indices]
                    
                    # Calculate similarity between categories
                    cross_similarity = cosine_similarity(cat1_embeddings, cat2_embeddings)
                    inter_similarities.append(float(np.mean(cross_similarity)))
        
        if inter_similarities:
            metrics["avg_inter_category_similarity"] = float(np.mean(inter_similarities))
            
            # Category separation score (higher is better)
            if "avg_intra_category_similarity" in metrics:
                metrics["category_separation_score"] = (
                    metrics["avg_intra_category_similarity"] - 
                    metrics["avg_inter_category_similarity"]
                )
    
    # 3. Clustering quality (if enough categories)
    n_clusters = min(len(unique_categories), 5)  # Use up to 5 clusters
    
    if n_clusters > 1 and len(embeddings) > n_clusters:
        try:
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Calculate silhouette score
            silhouette = silhouette_score(embeddings, cluster_labels)
            metrics["silhouette_score"] = float(silhouette)
            
            # Calculate inertia (within-cluster sum of squares)
            metrics["kmeans_inertia"] = float(kmeans.inertia_)
            
        except Exception as e:
            print(f"  ⚠️  Clustering failed: {e}")
    
    # 4. Nearest neighbor consistency
    # For a sample of products, check if nearest neighbors are from same category
    sample_size = min(20, len(embeddings))
    sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
    
    neighbor_consistency_scores = []
    for idx in sample_indices:
        # Calculate distances to all other embeddings
        similarities = cosine_similarity([embeddings[idx]], embeddings)[0]
        
        # Get top 5 neighbors (excluding self)
        similarities[idx] = -1  # Exclude self
        top_5_indices = np.argsort(similarities)[-5:]
        
        # Check how many are from same category
        same_category_count = sum(
            1 for neighbor_idx in top_5_indices 
            if labels[neighbor_idx] == labels[idx]
        )
        
        neighbor_consistency_scores.append(same_category_count / 5.0)
    
    metrics["avg_neighbor_consistency"] = float(np.mean(neighbor_consistency_scores))
    
    # 5. Embedding statistics
    metrics["embedding_stats"] = {
        "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
        "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
        "mean_value": float(np.mean(embeddings)),
        "std_value": float(np.std(embeddings))
    }
    
    return metrics
