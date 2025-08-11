# src/services/alpaca_service.py
import asyncio
import logging
import random
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient

# Import `Adjustment` from the alpaca library
from alpaca.data.requests import Adjustment, DataFeed, NewsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlpacaService:
    """
    Comprehensive Alpaca API service for algorithmic trading.

    This service encapsulates all interactions with Alpaca's Trading API, Market Data API,
    and News API. In institutional trading, this would be called the "execution management
    system" (EMS) interface.
    """

    def __init__(self):
        # Initialize all Alpaca clients
        # Dynamically select API keys based on environment (paper/live)
        env = "paper" if settings.paper_trading else "live"
        client_params = {
            "api_key": getattr(settings.alpaca, f"{env}_api_key_id"),
            "secret_key": getattr(settings.alpaca, f"{env}_api_secret_key"),
        }

        # TradingClient handles order management and account info
        self.trading_client = TradingClient(**client_params, paper=settings.paper_trading)

        # StockHistoricalDataClient handles price data
        self.stock_data_client = StockHistoricalDataClient(**client_params)

        # NewsClient handles news data - no API keys needed for basic access
        self.news_client = NewsClient(**client_params)

    async def get_account_info(self) -> dict:
        """
        Retrieve account information including buying power and positions.

        This is our "risk dashboard" - we need to know our current exposure,
        available capital, and any restrictions (like PDT rules).
        """
        try:
            account = self.trading_client.get_account()
            return account.model_dump() if hasattr(account, "model_dump") else dict(account)
        except Exception as e:
            raise Exception(f"Failed to get account info: {e}") from e

    async def fetch_historical_prices(
        self,
        symbol_or_symbols: str | list[str],
        days_back: int = 252,  # ~1 trading year
        timeframe: TimeFrame = TimeFrame.Day,
    ) -> pd.DataFrame:
        """
        Fetch historical price data for model training and backtesting.

        Financial Concept: OHLCV Data
        - Open: First trade price of the day
        - High: Highest trade price of the day
        - Low: Lowest trade price of the day
        - Close: Last trade price of the day
        - Volume: Number of shares traded

        For SPY (an ETF), volume indicates institutional interest and liquidity.
        High volume often precedes significant price movements, and this is especially
        true in extended hours when only sophisticated traders are active
        """
        # We must use a 15-minute buffer because real-time data is not available on the free tier
        end_date = datetime.now() - timedelta(minutes=15)  # Latest data with 15 min buffer
        start_date = end_date - timedelta(days=days_back * 1.5)  # Buffer for weekends/holidays

        symbol_or_symbols = (
            [symbol_or_symbols] if isinstance(symbol_or_symbols, str) else symbol_or_symbols
        )
        request = StockBarsRequest(
            symbol_or_symbols=symbol_or_symbols,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            feed=DataFeed.SIP,  # SIP stands for Securities Information Processor, which provides consolidated data. Better than 'iex' for SPY
            asof=None,  # Use 'asof' to get the latest available data
            adjustment=Adjustment.RAW,  # Use 'raw' to get unadjusted prices
        )

        columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "returns",
            "volatility",
            "volume_sma",
            "session",
        ]
        try:
            bars = self.stock_data_client.get_stock_bars(request)
            df = bars.df.reset_index()

            # Add technical features that might be useful for the model
            df["returns"] = df["close"].pct_change()
            df["volatility"] = df["returns"].rolling(window=20).std()
            df["volume_sma"] = df["volume"].rolling(window=20).mean()

            # Add a helpful column to identify extended hours
            df["session"] = self._classify_trading_session(df)

            result_df = (
                df[
                    [
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "returns",
                        "volatility",
                        "volume_sma",
                        "session",
                    ]
                ]
                .dropna()
                .reset_index(drop=True)
            )
            return result_df if not result_df.empty else pd.DataFrame(columns=columns)

        except Exception:
            # Return an empty DataFrame with the expected columns on error
            return pd.DataFrame(columns=columns)

    def _classify_trading_session(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify each bar as 'premarket', 'regular', or 'afterhours'.

        This is like tagging each piece of data with when it happened in the
        trading day. Very useful for understanding behavioral patterns:
        - Premarket: Often driven by overnight news, earnings, global events
        - Regular: Full institutional and retail participation
        - Afterhours: Often continuation of regular hours trends, earnings reactions
        """
        if df.empty:
            return pd.Series([], dtype="object")

        df_copy = df.copy()
        df_copy["timestamp_et"] = pd.to_datetime(df_copy["timestamp"]).dt.tz_convert("US/Eastern")
        df_copy["time_only"] = df_copy["timestamp_et"].dt.time

        def classify_session(time_val):
            if time_val < pd.Timestamp("09:30:00").time():
                return "premarket"
            elif time_val <= pd.Timestamp("16:00:00").time():
                return "regular"
            else:
                return "afterhours"

        result = df_copy["time_only"].apply(classify_session)
        return pd.Series(result, dtype="object")

    async def fetch_news_data(
        self,
        comma_separated_symbols: str,
        start_date: datetime,
        end_date: datetime,
        max_articles: int = 10000,  # Increased default for full ranges
    ) -> pd.DataFrame:
        """
        Fetch news data for complete date range using.

        """

        request = NewsRequest(
            symbols=comma_separated_symbols,
            start=start_date,
            end=end_date,
            limit=max_articles,  # API will paginate if more articles exist
        )

        try:
            # Single API call - pagination handled automatically
            news = self.news_client.get_news(request)
            news_df = news.df.reset_index()

            logger.debug(f"📰 API returned {len(news_df)} articles")

            if news_df.empty:
                return pd.DataFrame(
                    columns=["created_at", "headline", "summary", "symbols", "source", "url"]
                )

            # Apply your business logic filters
            news_df = self._filter_news_by_relevance(news_df)
            news_df = news_df.sort_values("created_at", ascending=False)

            # Ensure consistent column structure
            expected_cols = ["created_at", "headline", "summary", "symbols", "source", "url"]
            result_df = news_df.reindex(columns=expected_cols).reset_index(drop=True)

            return pd.DataFrame(result_df)

        except Exception as e:
            raise Exception(f"Failed to fetch news data for {comma_separated_symbols}: {e}") from e

    def _filter_news_by_relevance(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter news articles based on relevance to SPY trading strategy.

        This implements our news selection strategy - we want macro-economic news
        and broad market sentiment, not individual stock stories (unless they're
        major S&P 500 constituents that could move the entire index).
        """
        if news_df.empty:
            return news_df

        # Filter by headline length (too short = low information content)
        news_df = news_df[news_df["headline"].str.len() >= settings.news.min_headline_length].copy()

        # Create relevance score based on keywords
        macro_pattern = "|".join(settings.news.news_keywords_macro or [])
        financial_pattern = "|".join(settings.news.news_keywords_financial or [])

        news_df["is_macro"] = news_df["headline"].str.contains(macro_pattern, case=False, na=False)
        news_df["is_financial"] = news_df["headline"].str.contains(
            financial_pattern, case=False, na=False
        )

        # Keep articles that match our criteria or are from major financial sources
        major_sources = ["Reuters", "Bloomberg", "MarketWatch", "CNBC", "Wall Street Journal"]
        news_df["is_major_source"] = news_df["source"].isin(major_sources)

        # Filter: macro news OR financial news OR major source
        relevant_news = news_df[
            news_df["is_macro"] | news_df["is_financial"] | news_df["is_major_source"]
        ].copy()

        return relevant_news.drop(["is_macro", "is_financial", "is_major_source"], axis=1)

    async def submit_market_order(
        self, symbol: str, side: OrderSide, quantity: float, client_order_id: str | None = None
    ) -> dict:
        """
        Submit a market order to Alpaca.

        Financial Concept: Market Orders vs. Limit Orders
        - Market Order: Executes immediately at current market price
        - Limit Order: Executes only at specified price or better

        For our 12 PM strategy, market orders are appropriate because:
        1. SPY has excellent liquidity (tight bid-ask spreads)
        2. We're not trying to time the market precisely
        3. Execution certainty is more important than price precision
        4. Our holding period is only a few hours
        """
        market_order = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )

        try:
            order = self.trading_client.submit_order(market_order)
            return {
                "order_id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": float(order.qty),
                "status": order.status.value,
                "submitted_at": order.submitted_at,
                "filled_qty": float(order.filled_qty or 0),
                "filled_price": float(order.filled_avg_price or 0),
            }
        except Exception as e:
            raise Exception(f"Failed to submit order: {e}") from e

    async def get_current_position(self, symbol: str) -> dict | None:
        """Get current position for a symbol."""
        try:
            positions = self.trading_client.get_all_positions()
            for position in positions:
                if position.symbol == symbol:
                    return {
                        "symbol": position.symbol,
                        "quantity": float(position.qty),
                        "side": position.side.value,
                        "market_value": float(position.market_value),
                        "cost_basis": float(position.cost_basis),
                        "unrealized_pnl": float(position.unrealized_pnl),
                        "unrealized_pnl_pct": float(position.unrealized_plpc),
                    }
            return None
        except Exception as e:
            raise Exception(f"Failed to get position for {symbol}: {e}") from e

    async def close_position(self, symbol: str) -> dict:
        """Close all positions for a symbol."""
        try:
            response = self.trading_client.close_position(symbol)
            return {"status": "closed", "order_id": response.id}
        except Exception as e:
            raise Exception(f"Failed to close position for {symbol}: {e}") from e

    async def fetch_bulk_historical_prices(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: TimeFrame = TimeFrame.Day,
        delay_range: tuple[float, float] = (0.1, 0.3),
    ) -> AsyncGenerator[tuple[str, pd.DataFrame], None]:
        """
        Strategic chunking for large historical price datasets.

        We chunk by logical units (bar count or timeframe-appropriate
        periods) rather than arbitrary calendar periods.
        """

        for symbol in symbols:
            logger.info(f"🔄 Processing {symbol} from {start_date.date()} to {end_date.date()}")

            # Calculate appropriate chunk size based on timeframe
            if timeframe == TimeFrame.Minute:
                chunk_days = 30  # ~11,700 minute bars per month
            elif timeframe == TimeFrame.Hour:
                chunk_days = 180  # ~6 months of hourly data
            elif timeframe == TimeFrame.Day:
                chunk_days = 365  # 1 year of daily data
            else:
                chunk_days = 30  # Conservative default

            current_date = start_date
            chunk_count = 0

            while current_date < end_date:
                chunk_end = min(current_date + timedelta(days=chunk_days), end_date)
                chunk_count += 1

                try:
                    chunk_df = await self.fetch_historical_prices_range(
                        symbol_or_symbols=symbol,
                        start_date=current_date,
                        end_date=chunk_end,
                        timeframe=timeframe,
                    )

                    if not chunk_df.empty:
                        logger.info(
                            f"✅ Fetched {len(chunk_df)} {timeframe} bars for {symbol} chunk {chunk_count}"
                        )
                        yield symbol, chunk_df

                    # Smart delay based on data volume
                    delay = random.uniform(*delay_range)
                    await asyncio.sleep(delay)

                except Exception as e:
                    logger.error(f"❌ Failed to fetch {symbol} chunk {chunk_count}: {e}")
                    continue

                current_date = chunk_end

    async def fetch_historical_prices_range(
        self,
        symbol_or_symbols: str | list[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: TimeFrame = TimeFrame.Day,
    ) -> pd.DataFrame:
        """
        Fetch price data for exact date range - no more days_back confusion!

        This method properly uses the provided start and end dates instead of
        always fetching data relative to "now".
        """

        symbol_or_symbols = (
            [symbol_or_symbols] if isinstance(symbol_or_symbols, str) else symbol_or_symbols
        )

        request = StockBarsRequest(
            symbol_or_symbols=symbol_or_symbols,
            timeframe=timeframe,
            start=start_date,  # 🔧 FIXED: Use actual start date
            end=end_date,  # 🔧 FIXED: Use actual end date
            feed=DataFeed.SIP,
            asof=None,
            adjustment=Adjustment.RAW,
        )

        try:
            bars = self.stock_data_client.get_stock_bars(request)
            df = bars.df.reset_index()

            if df.empty:
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

            # Add your technical indicators and features
            df["returns"] = df["close"].pct_change()
            df["volatility"] = df["returns"].rolling(window=20).std()
            df["volume_sma"] = df["volume"].rolling(window=20).mean()
            df["session"] = self._classify_trading_session(df)

            return df.dropna().reset_index(drop=True)

        except Exception as e:
            raise Exception(f"Failed to fetch price data for {symbol_or_symbols}: {e}") from e

    async def fetch_bulk_news_data(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        max_articles_per_symbol: int = 10000,  # Let API handle limits
        delay_between_symbols: float = 1.0,
    ) -> AsyncGenerator[tuple[str, pd.DataFrame], None]:
        """
        Fetch historical news data using API pagination.
        """
        total_symbols = len(symbols)

        for symbol_idx, symbol in enumerate(symbols):
            logger.info(f"📰 Processing news for {symbol} ({symbol_idx + 1}/{total_symbols})")
            logger.info(f"📅 Date range: {start_date.date()} to {end_date.date()}")

            try:
                # Single call - let API handle all pagination
                news_df = await self.fetch_news_data(
                    comma_separated_symbols=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    max_articles=max_articles_per_symbol,
                )

                if not news_df.empty:
                    logger.info(f"✅ Retrieved {len(news_df)} news articles for {symbol}")
                    yield symbol, news_df
                else:
                    logger.info(f"ℹ️ No news found for {symbol} in date range")

                # Rate limiting between symbols
                if symbol_idx < total_symbols - 1:  # Don't delay after last symbol
                    await asyncio.sleep(delay_between_symbols)

            except Exception as e:
                logger.error(f"❌ Failed to fetch news for {symbol}: {e}")
                continue
