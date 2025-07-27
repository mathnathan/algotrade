from datetime import datetime

import torch
import torch.nn as nn


class UnifiedTemporalEmbedding(nn.Module):
    """
    A temporal embedding system designed for multimodal financial data.
    """

    def __init__(self, d_model, max_sequence_length=2000):
        super().__init__()
        self.d_model = d_model

        # Hierarchical temporal components that work for both modalities
        self.month_embedding = nn.Embedding(12, d_model // 8)
        self.weekday_embedding = nn.Embedding(5, d_model // 8)
        self.hour_embedding = nn.Embedding(24, d_model // 8)  # 24 hours for full day coverage
        self.minute_embedding = nn.Embedding(60, d_model // 8)

        # Relative position embeddings for sequence modeling
        # This helps the model understand "how far apart in time are these two events?"
        self.relative_position_embedding = nn.Embedding(max_sequence_length * 2, d_model // 4)

        # Trading session context (crucial for understanding market context)
        self.session_embedding = nn.Embedding(
            4, d_model // 8
        )  # premarket, open, regular, close, afterhours

        # Final projection to combine all temporal components
        self.temporal_projection = nn.Linear(d_model // 2 + d_model // 4, d_model)

        # Learnable parameters for modality-specific temporal scaling
        # This allows news and price data to have different temporal sensitivities
        self.modality_temporal_scale = nn.Parameter(torch.ones(2))  # [news_scale, price_scale]

    def _get_trading_session(self, hour, minute):
        """
        Classify the exact trading session for precise market context.

        Different sessions have different volatility patterns, participant types,
        and information processing characteristics. The model needs to know
        "when in the trading day am I operating?"
        """
        total_minutes = hour * 60 + minute

        # Define session boundaries in minutes from midnight
        premarket_start = 4 * 60  # 4:00 AM
        market_open = 9 * 60 + 30  # 9:30 AM
        market_close = 16 * 60  # 4:00 PM
        afterhours_end = 20 * 60  # 8:00 PM

        if total_minutes < premarket_start:
            return 0  # Outside trading hours
        elif total_minutes < market_open:
            return 1  # Premarket
        elif total_minutes < market_close:
            return 2  # Regular hours
        elif total_minutes < afterhours_end:
            return 3  # After hours
        else:
            return 0  # Outside trading hours

    def encode_timestamp(self, timestamp, modality_type="price"):
        """
        Convert a timestamp into a rich temporal embedding.

        Args:
            timestamp: datetime object or tensor of datetime values
            modality_type: 'price' or 'news' to apply appropriate scaling

        This is like creating a temporal fingerprint that captures not just
        when something happened, but what kind of market environment it
        happened in.
        """
        if isinstance(timestamp, datetime):
            month = timestamp.month - 1  # 0-indexed
            weekday = timestamp.weekday()  # Monday=0
            hour = timestamp.hour
            minute = timestamp.minute
            session = self._get_trading_session(hour, minute)
        else:
            # Handle batch processing of timestamps
            month = timestamp[:, 0] - 1
            weekday = timestamp[:, 1]
            hour = timestamp[:, 2]
            minute = timestamp[:, 3]
            session = torch.tensor(
                [
                    self._get_trading_session(h.item(), m.item())
                    for h, m in zip(hour, minute, strict=False)
                ]
            ).to(timestamp.device)

        # Get embeddings for each temporal component
        month_emb = self.month_embedding(month)
        weekday_emb = self.weekday_embedding(weekday)
        hour_emb = self.hour_embedding(hour)
        minute_emb = self.minute_embedding(minute)
        session_emb = self.session_embedding(session)

        # Combine all temporal components
        temporal_features = torch.cat(
            [month_emb, weekday_emb, hour_emb, minute_emb, session_emb], dim=-1
        )

        return temporal_features

    def compute_relative_positions(self, timestamps_a, timestamps_b):
        """
        Compute relative temporal distances between two sets of timestamps.

        This is crucial for cross-attention between news and price data.
        The model needs to understand "this news article came out 3 minutes
        before this price movement" to learn cause-and-effect relationships.
        """

        # Convert timestamps to minutes since a reference point
        def timestamp_to_minutes(ts):
            if isinstance(ts, datetime):
                return ts.hour * 60 + ts.minute
            else:
                return ts[:, 2] * 60 + ts[:, 3]  # hour * 60 + minute

        minutes_a = timestamp_to_minutes(timestamps_a)
        minutes_b = timestamp_to_minutes(timestamps_b)

        # Compute relative differences (clamped to avoid extreme values)
        relative_diff = torch.clamp(
            minutes_b.unsqueeze(-1) - minutes_a.unsqueeze(-2),
            min=-500,
            max=500,  # +/- ~8 hours maximum relative distance
        )

        # Map to embedding indices (add offset to handle negative values)
        relative_indices = relative_diff + 500

        return self.relative_position_embedding(relative_indices)

    def forward(self, price_timestamps, news_timestamps=None):
        """
        Generate temporal embeddings for multimodal data.

        Returns both absolute temporal embeddings and relative position
        embeddings that enable precise cross-modal temporal attention.
        """
        # Encode absolute temporal positions for price data
        price_temporal_emb = self.encode_timestamp(price_timestamps, "price")
        price_temporal_emb = self.temporal_projection(price_temporal_emb)

        # Apply modality-specific scaling
        price_temporal_emb = price_temporal_emb * self.modality_temporal_scale[1]

        result = {"price_temporal_embeddings": price_temporal_emb}

        # If news timestamps are provided, handle cross-modal temporal relationships
        if news_timestamps is not None:
            news_temporal_emb = self.encode_timestamp(news_timestamps, "news")
            news_temporal_emb = self.temporal_projection(news_temporal_emb)
            news_temporal_emb = news_temporal_emb * self.modality_temporal_scale[0]

            # Compute relative positions for cross-attention
            relative_pos_emb = self.compute_relative_positions(news_timestamps, price_timestamps)

            result.update(
                {
                    "news_temporal_embeddings": news_temporal_emb,
                    "cross_modal_relative_positions": relative_pos_emb,
                }
            )

        return result
