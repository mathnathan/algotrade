# src/services/embedding_service.py
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import logging
from datetime import datetime, timedelta

from src.config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Clean embedding service that prepares headline embeddings and sentiment vectors
    for consumption by the cross-attention model.
    
    Philosophy: Separation of Concerns
    This service focuses purely on converting text to numerical representations.
    All complex temporal modeling, attention mechanisms, and cross-modal interactions
    are handled by the model itself. This gives the model maximum flexibility to learn 
    optimal integration strategies.
    
    The service produces two parallel sequences:
    1. Headline embedding vectors (384-dimensional semantic representations)
    2. Sentiment vectors (3-dimensional emotional signals)
    
    Both sequences are temporally aligned and bucketed, ready for the model's
    learned positional encodings and self-attention mechanisms.
    """
    
    def __init__(self):
        self.embedding_model = self._load_embedding_model()
        self.sentiment_analyzer = self._load_sentiment_analyzer()
        
        # Dimensions for our clean two-stream approach
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()  # 384
        self.sentiment_dim = 3  # [positive, negative, intensity]
        
        logger.info(f"Embedding service initialized - "
                   f"Embedding dim: {self.embedding_dim}, Sentiment dim: {self.sentiment_dim}")
    
    def _load_embedding_model(self) -> SentenceTransformer:
        """
        Load the sentence transformer model for semantic embeddings.
        
        We use a pre-trained model that captures general semantic relationships.
        The model will learn domain-specific financial relationships through
        the cross-attention mechanism during training.
        """
        try:
            model = SentenceTransformer(
                settings.embedding_model,
                cache_folder=str(settings.huggingface_cache_dir),
                device='cpu'
            )
            # Optimize for financial headline length
            model.max_seq_length = 256
            logger.info(f"Loaded embedding model: {settings.embedding_model}")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _load_sentiment_analyzer(self):
        """
        Load sentiment analysis model for emotional signal extraction.
        
        We extract three-dimensional sentiment signals that capture both
        the direction and intensity of emotional content. This gives the
        model rich signals about market psychology.
        """
        try:
            analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1,
                return_all_scores=True
            )
            logger.info("Loaded sentiment analyzer")
            return analyzer
        except Exception as e:
            logger.warning(f"Failed to load sentiment analyzer: {e}")
            return None
    
    def extract_headline_embedding(self, headline: str) -> np.ndarray:
        """
        Extract semantic embedding for a single headline.
        
        This captures the semantic content of the headline in a 384-dimensional
        vector space where similar meanings cluster together. The embedding
        preserves the rich semantic relationships that the model can leverage
        through attention mechanisms.
        """
        if not headline or not headline.strip():
            return np.zeros(self.embedding_dim)
        
        try:
            embedding = self.embedding_model.encode(
                headline.strip(),
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for stable training
            )
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding for headline: {e}")
            return np.zeros(self.embedding_dim)
    
    def extract_sentiment_vector(self, headline: str) -> np.ndarray:
        """
        Extract three-dimensional sentiment vector from headline.
        
        The sentiment vector captures:
        [0] Positive sentiment strength (0-1)
        [1] Negative sentiment strength (0-1)  
        [2] Sentiment intensity (0-1, where 1 = highly emotional, 0 = neutral)
        
        This representation allows the model to distinguish between mildly positive
        and strongly positive news, which is crucial for financial prediction where
        emotional intensity often drives market reactions.
        """
        if not self.sentiment_analyzer or not headline or not headline.strip():
            return np.array([0.33, 0.33, 0.34])  # Neutral baseline
        
        try:
            # Get sentiment scores for all classes
            results = self.sentiment_analyzer(headline.strip()[:512])
            
            # Parse results (format: [{'label': 'POSITIVE', 'score': 0.8}, ...])
            sentiment_scores = {result['label'].lower(): result['score'] for result in results[0]}
            
            # Extract individual sentiment strengths
            positive_score = sentiment_scores.get('positive', 0.0)
            negative_score = sentiment_scores.get('negative', 0.0)
            neutral_score = sentiment_scores.get('neutral', 0.0)
            
            # Normalize to ensure probabilities sum to 1
            total = positive_score + negative_score + neutral_score
            if total > 0:
                positive_score /= total
                negative_score /= total
                neutral_score /= total
            
            # Calculate intensity (how far from neutral)
            intensity = 1.0 - neutral_score
            
            return np.array([positive_score, negative_score, intensity])
            
        except Exception as e:
            logger.error(f"Failed to analyze sentiment: {e}")
            return np.array([0.33, 0.33, 0.34])
    
    def combine_embeddings_with_sentiment_weighting(
        self, 
        embeddings: List[np.ndarray], 
        sentiment_vectors: List[np.ndarray]
    ) -> np.ndarray:
        """
        Combine multiple embeddings using sentiment-aware weighting.
        
        When multiple headlines exist in the same temporal bucket, we need to
        create a single representative embedding. We use sentiment intensity
        to weight the combination, simulating how markets pay more attention
        to emotionally charged news.
        
        Mathematical Foundation:
        weight_i = sentiment_intensity_i + baseline_weight
        combined_embedding = Σ(weight_i * embedding_i) / Σ(weight_i)
        
        This ensures that highly emotional news gets more influence while
        preventing any single headline from completely dominating the signal.
        """
        if not embeddings:
            return np.zeros(self.embedding_dim)
        
        if len(embeddings) == 1:
            return embeddings[0]
        
        # Calculate weights based on sentiment intensity
        weights = []
        for sentiment_vector in sentiment_vectors:
            # Use intensity (third dimension) as primary weight
            intensity = sentiment_vector[2]
            # Add baseline weight to prevent zero weights
            weight = intensity + 0.1
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted combination of embeddings
        combined_embedding = np.average(embeddings, axis=0, weights=weights)
        
        # Renormalize the combined embedding
        norm = np.linalg.norm(combined_embedding)
        if norm > 0:
            combined_embedding = combined_embedding / norm
        
        return combined_embedding
    
    def combine_sentiment_vectors_with_weighting(
        self, 
        sentiment_vectors: List[np.ndarray]
    ) -> np.ndarray:
        """
        Combine multiple sentiment vectors using the same weighting mechanism.
        
        This maintains consistency with how we combine embeddings, ensuring that
        the sentiment signals align with the semantic signals when multiple
        headlines are bucketed together.
        """
        if not sentiment_vectors:
            return np.array([0.33, 0.33, 0.34])
        
        if len(sentiment_vectors) == 1:
            return sentiment_vectors[0]
        
        # Use the same weighting logic as embeddings
        weights = []
        for sentiment_vector in sentiment_vectors:
            intensity = sentiment_vector[2]
            weight = intensity + 0.1
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted combination
        combined_sentiment = np.average(sentiment_vectors, axis=0, weights=weights)
        
        return combined_sentiment
    
    def process_news_for_temporal_bucket(
        self, 
        headlines: List[str],
        bucket_identifier: str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a list of headlines that belong to the same temporal bucket.
        
        This is the core function that converts raw headlines into the two
        parallel streams that the model expects:
        
        1. A single combined embedding vector representing all headlines
        2. A single combined sentiment vector with the same weighting
        
        Returns:
        - combined_embedding: Single 384-dim vector for this time bucket
        - combined_sentiment: Single 3-dim vector for this time bucket
        """
        if not headlines:
            return np.zeros(self.embedding_dim), np.array([0.33, 0.33, 0.34])
        
        # Extract individual embeddings and sentiment vectors
        embeddings = []
        sentiment_vectors = []
        
        for headline in headlines:
            if headline and headline.strip():
                embedding = self.extract_headline_embedding(headline)
                sentiment = self.extract_sentiment_vector(headline)
                
                embeddings.append(embedding)
                sentiment_vectors.append(sentiment)
        
        if not embeddings:
            return np.zeros(self.embedding_dim), np.array([0.33, 0.33, 0.34])
        
        # Combine using sentiment-aware weighting
        combined_embedding = self.combine_embeddings_with_sentiment_weighting(
            embeddings, sentiment_vectors
        )
        combined_sentiment = self.combine_sentiment_vectors_with_weighting(
            sentiment_vectors
        )
        
        if bucket_identifier:
            logger.debug(f"Processed {len(headlines)} headlines for bucket {bucket_identifier}")
        
        return combined_embedding, combined_sentiment
    
    def create_temporal_sequences(
        self, 
        news_df: pd.DataFrame,
        sequence_length: int,
        bucket_hours: int = 24
    ) -> Dict[str, np.ndarray]:
        """
        Create temporally-ordered sequences of embeddings and sentiment vectors.
        
        This function transforms the raw news dataframe into the structured
        sequences that the model expects. It handles temporal bucketing,
        ensures proper chronological ordering, and produces fixed-length
        sequences ready for the model's learned positional encodings.
        
        Parameters:
        - news_df: Raw news data with 'published_at' and 'headline' columns
        - sequence_length: Number of time buckets to include in the sequence
        - bucket_hours: Hours per temporal bucket (default 24 = daily buckets)
        
        Returns:
        - embedding_sequence: Array of shape (sequence_length, 384)
        - sentiment_sequence: Array of shape (sequence_length, 3)
        - bucket_dates: List of dates corresponding to each sequence position
        """
        if news_df.empty:
            return {
                'embedding_sequence': np.zeros((sequence_length, self.embedding_dim)),
                'sentiment_sequence': np.zeros((sequence_length, self.sentiment_dim)),
                'bucket_dates': [],
                'valid_buckets': 0
            }
        
        # Sort news by publication time (oldest first for chronological order)
        sorted_news = news_df.sort_values('published_at').copy()
        
        # Create temporal buckets
        end_date = sorted_news['published_at'].max()
        start_date = end_date - timedelta(hours=bucket_hours * sequence_length)
        
        # Initialize sequences
        embedding_sequence = np.zeros((sequence_length, self.embedding_dim))
        sentiment_sequence = np.zeros((sequence_length, self.sentiment_dim))
        bucket_dates = []
        valid_buckets = 0
        
        # Process each temporal bucket
        for i in range(sequence_length):
            # Calculate bucket time range (going backwards from end_date)
            bucket_end = end_date - timedelta(hours=bucket_hours * i)
            bucket_start = bucket_end - timedelta(hours=bucket_hours)
            
            # Find headlines in this bucket
            bucket_news = sorted_news[
                (sorted_news['published_at'] >= bucket_start) & 
                (sorted_news['published_at'] < bucket_end)
            ]
            
            bucket_headlines = bucket_news['headline'].fillna('').tolist()
            bucket_headlines = [h for h in bucket_headlines if h.strip()]
            
            # Process headlines for this bucket
            if bucket_headlines:
                embedding, sentiment = self.process_news_for_temporal_bucket(
                    bucket_headlines,
                    bucket_identifier=f"bucket_{sequence_length-1-i}_{bucket_start.date()}"
                )
                # Store in chronological order (oldest first)
                embedding_sequence[sequence_length-1-i] = embedding
                sentiment_sequence[sequence_length-1-i] = sentiment
                valid_buckets += 1
            else:
                # Use zero vectors for empty buckets
                embedding_sequence[sequence_length-1-i] = np.zeros(self.embedding_dim)
                sentiment_sequence[sequence_length-1-i] = np.array([0.33, 0.33, 0.34])
            
            bucket_dates.append(bucket_start.date())
        
        # Reverse bucket_dates to match chronological order
        bucket_dates.reverse()
        
        logger.info(f"Created sequences with {valid_buckets}/{sequence_length} valid buckets")
        
        return {
            'embedding_sequence': embedding_sequence,
            'sentiment_sequence': sentiment_sequence,
            'bucket_dates': bucket_dates,
            'valid_buckets': valid_buckets
        }
    
    def prepare_model_inputs(
        self, 
        news_df: pd.DataFrame,
        target_date: pd.Timestamp,
        sequence_length: int = None
    ) -> Dict[str, np.ndarray]:
        """
        Main interface for preparing news data for model consumption.
        
        This function takes raw news data and produces clean, temporally-ordered
        sequences ready for the model's three-stream architecture.
        
        The output format matches exactly what the model expects:
        - Embedding sequences for the headline self-attention stream
        - Sentiment sequences for the sentiment self-attention stream
        - Both sequences are temporally aligned and ready for learned positional encodings
        """
        if sequence_length is None:
            sequence_length = settings.model.news_sequence_length
        
        # Filter news up to the target date
        relevant_news = news_df[news_df['published_at'] <= target_date].copy()
        
        # Create the temporal sequences
        sequences = self.create_temporal_sequences(
            relevant_news,
            sequence_length=sequence_length,
            bucket_hours=24  # Daily buckets
        )
        
        logger.info(f"Prepared model inputs for {target_date.date()} "
                   f"with {sequences['valid_buckets']} valid news buckets")
        
        return {
            'headline_embeddings': sequences['embedding_sequence'],
            'sentiment_vectors': sequences['sentiment_sequence'],
            'bucket_dates': sequences['bucket_dates'],
            'metadata': {
                'valid_buckets': sequences['valid_buckets'],
                'total_buckets': sequence_length,
                'target_date': target_date,
                'coverage_ratio': sequences['valid_buckets'] / sequence_length
            }
        }