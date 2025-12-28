"""MPNet-based semantic embedding generation."""

from typing import List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class MPNetEmbedder:
    """Generate semantic embeddings using MPNet model.
    
    Uses sentence-transformers/all-mpnet-base-v2 for high-quality
    sentence embeddings (384 dimensions).
    
    Attributes:
        model_name: Name of the sentence-transformers model
        device: Computing device ('cuda', 'cpu', or 'mps')
        batch_size: Batch size for encoding
        show_progress: Whether to display progress bar
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        show_progress: bool = True
    ):
        """Initialize the MPNet embedder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Computing device (auto-detected if None)
            batch_size: Batch size for encoding
            show_progress: Show tqdm progress bar
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress = show_progress
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        
    def encode(
        self,
        texts: Union[List[str], np.ndarray],
        normalize: bool = True,
        convert_to_numpy: bool = True
    ) -> np.ndarray:
        """Encode texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            normalize: Whether to L2-normalize embeddings
            convert_to_numpy: Return numpy array (vs torch tensor)
            
        Returns:
            Array of embeddings with shape (n_texts, 384)
        """
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
            
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=convert_to_numpy
        )
        
        return embeddings
    
    def encode_batches(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Encode texts in batches with manual control.
        
        Useful for very large datasets where you want explicit
        batch processing control.
        
        Args:
            texts: List of texts to encode
            batch_size: Override default batch size
            
        Returns:
            Array of embeddings
        """
        batch_size = batch_size or self.batch_size
        n_batches = (len(texts) + batch_size - 1) // batch_size
        
        embeddings_list = []
        
        iterator = range(n_batches)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Encoding batches")
        
        for i in iterator:
            start = i * batch_size
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            embeddings_list.append(batch_embeddings)
        
        return np.vstack(embeddings_list)
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self.model.get_sentence_embedding_dimension()
    
    def __repr__(self) -> str:
        return (
            f"MPNetEmbedder(model={self.model_name}, "
            f"device={self.device}, dim={self.embedding_dim})"
        )