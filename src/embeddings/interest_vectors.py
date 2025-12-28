"""Binary interest vector construction from categorical labels."""

from typing import Dict, List, Optional

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


class InterestVectorizer:
    """Convert categorical labels to binary interest vectors.
    
    Transforms category labels (e.g., "Sports / Fitness") into
    binary indicator vectors for clustering based on shared interests.
    
    Attributes:
        categories: Unique category names
        n_categories: Number of distinct categories
        binarizer: MultiLabelBinarizer for encoding
    """
    
    def __init__(self, categories: Optional[List[str]] = None):
        """Initialize the vectorizer.
        
        Args:
            categories: Predefined list of categories (optional)
        """
        self.categories = categories
        self.binarizer = MultiLabelBinarizer(classes=categories)
        self._is_fitted = False
        
    def fit(self, labels: List[str]) -> "InterestVectorizer":
        """Fit the vectorizer on category labels.
        
        Args:
            labels: List of category labels for each item
            
        Returns:
            Self for method chaining
        """
        # Wrap each label in a list for MultiLabelBinarizer
        labels_wrapped = [[label] for label in labels]
        self.binarizer.fit(labels_wrapped)
        self.categories = list(self.binarizer.classes_)
        self._is_fitted = True
        return self
    
    def transform(self, labels: List[str]) -> np.ndarray:
        """Transform labels into binary vectors.
        
        Args:
            labels: Category labels to transform
            
        Returns:
            Binary array of shape (n_samples, n_categories)
        """
        if not self._is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        labels_wrapped = [[label] for label in labels]
        return self.binarizer.transform(labels_wrapped)
    
    def fit_transform(self, labels: List[str]) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            labels: Category labels
            
        Returns:
            Binary interest vectors
        """
        return self.fit(labels).transform(labels)
    
    def inverse_transform(self, vectors: np.ndarray) -> List[List[str]]:
        """Convert binary vectors back to category labels.
        
        Args:
            vectors: Binary interest vectors
            
        Returns:
            List of category lists for each sample
        """
        return self.binarizer.inverse_transform(vectors)
    
    def get_category_names(self, indices: List[int]) -> List[str]:
        """Get category names from indices.
        
        Args:
            indices: Category indices
            
        Returns:
            Category names
        """
        return [self.categories[i] for i in indices]
    
    def get_category_indices(self, vector: np.ndarray) -> List[int]:
        """Get active category indices from a binary vector.
        
        Args:
            vector: Single binary interest vector
            
        Returns:
            List of active category indices
        """
        return np.where(vector > 0)[0].tolist()
    
    @property
    def n_categories(self) -> int:
        """Number of unique categories."""
        return len(self.categories) if self.categories else 0
    
    def compute_jaccard_similarity(
        self,
        vectors: np.ndarray
    ) -> np.ndarray:
        """Compute pairwise Jaccard similarity matrix.
        
        Jaccard similarity = |A ∩ B| / |A ∪ B|
        
        Args:
            vectors: Binary interest vectors (n_samples, n_categories)
            
        Returns:
            Similarity matrix (n_samples, n_samples)
        """
        intersection = vectors @ vectors.T  # Dot product = intersection count
        
        # Union = |A| + |B| - |A ∩ B|
        row_sums = vectors.sum(axis=1, keepdims=True)
        union = row_sums + row_sums.T - intersection
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            similarity = intersection / union
            similarity[~np.isfinite(similarity)] = 0.0
        
        return similarity
    
    def get_interest_summary(self, vectors: np.ndarray) -> Dict[str, int]:
        """Get category frequency summary.
        
        Args:
            vectors: Binary interest vectors
            
        Returns:
            Dictionary mapping category names to counts
        """
        category_counts = vectors.sum(axis=0).astype(int)
        return {
            cat: int(count)
            for cat, count in zip(self.categories, category_counts)
        }
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"InterestVectorizer(n_categories={self.n_categories}, {status})"