# src/scripts/populate_historical_data.py

# --- Instructions and Examples ---
# Populate SPY and QQQ data for the last year
# python -m src.scripts.populate_historical_data \
#    --symbols SPY QQQ \
#    --start-date 2024-01-01 \
#    --data-types prices news

# Resume a previous job that was interrupted
# python -m src.scripts.populate_historical_data \
#    --symbols SPY QQQ AAPL MSFT AMZN \
#    --start-date 2023-01-01 \
#    --end-date 2024-12-31

# Just collect price data for a larger set of symbols
# python -m src.scripts.populate_historical_data \
#    --symbols SPY QQQ IWM TLT GLD \
#    --start-date 2022-01-01 \
#    --data-types prices \
#    --log-level DEBUG

import argparse
import asyncio
import logging
from datetime import datetime

from alpaca.data.timeframe import TimeFrame

from src.database.connection import db_manager
from src.services.alpaca_service import AlpacaService
from src.services.data_insertion_service import DataInsertionService
from src.services.progress_manager import DataPopulationProgressManager, ProcessingStatus

logger = logging.getLogger(__name__)


class HistoricalDataPopulator:
    """
    Coordinates the complete historical data population process.

    ETL Pipeline for Financial Data
    This class implements a complete Extract-Transform-Load (ETL) pipeline

    Extract: Pull data from Alpaca APIs with rate limiting and error handling
    Transform: Validate and clean data using your DataValidator
    Load: Insert into PostgreSQL with proper conflict resolution
    """

    def __init__(self):
        self.alpaca_service = AlpacaService()
        self.progress_manager = DataPopulationProgressManager()
        self.insertion_service = DataInsertionService()
        self.start_time = datetime.now()

    async def populate_data(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime | None = None,
        data_types: list[str] | None = None,
        resume: bool = True,
    ) -> None:
        """
        Main coordination method for historical data population.

        Args:
            symbols: List of stock symbols to process
            start_date: Beginning of historical data range
            end_date: End of range (defaults to today)
            data_types: Types of data to collect ("prices", "news", or both)
            resume: Whether to resume from previous progress
        """
        if data_types is None:
            data_types = ["prices", "news"]
        if end_date is None:
            end_date = datetime.now()

        logger.info("🚀 Starting historical data population")
        logger.info(f"📊 Symbols: {symbols}")
        logger.info(f"📅 Date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"📈 Data types: {data_types}")

        # Initialize progress tracking for all symbols
        for symbol in symbols:
            self.progress_manager.initialize_symbol(symbol, start_date, end_date)

        # Determine which symbols need processing
        if resume:
            pending_symbols = self.progress_manager.get_pending_symbols()
            logger.info(f"🔄 Resuming: {len(pending_symbols)} symbols still need processing")
        else:
            pending_symbols = symbols
            logger.info(f"🆕 Fresh start: processing all {len(symbols)} symbols")

        # Process each symbol
        for symbol_idx, symbol in enumerate(pending_symbols):
            logger.info(f"\n📈 Processing {symbol} ({symbol_idx + 1}/{len(pending_symbols)})")

            if "prices" in data_types:
                await self._process_symbol_prices(symbol, start_date, end_date)

            if "news" in data_types:
                await self._process_symbol_news(symbol, start_date, end_date)

            # Print progress summary periodically
            if (symbol_idx + 1) % 10 == 0:
                summary = self.progress_manager.get_progress_summary()
                logger.info(
                    f"📊 Progress: {summary['prices_completion_rate']:.1f}% prices, "
                    f"{summary['news_completion_rate']:.1f}% news completed"
                )

        # Final summary
        await self._print_final_summary()

    async def _process_symbol_prices(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> None:
        """Process price data for a single symbol."""
        try:
            self.progress_manager.update_prices_progress(symbol, ProcessingStatus.IN_PROGRESS)

            total_records = 0
            total_errors = 0

            # Use the bulk method to get chunked data
            async for _chunk_symbol, price_df in self.alpaca_service.fetch_bulk_historical_prices(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                timeframe=TimeFrame.Minute,
            ):
                # Insert this chunk
                (
                    records_inserted,
                    validation_errors,
                ) = await self.insertion_service.insert_price_data(symbol, price_df)
                total_records += records_inserted
                total_errors += validation_errors

                logger.debug(f"💰 {symbol}: +{records_inserted} price records")

            # Update progress
            if total_errors == 0:
                self.progress_manager.update_prices_progress(
                    symbol, ProcessingStatus.COMPLETED, total_records
                )
                logger.info(f"✅ {symbol}: {total_records} price records completed")
            else:
                self.progress_manager.update_prices_progress(
                    symbol,
                    ProcessingStatus.FAILED,
                    total_records,
                    f"{total_errors} validation errors",
                )
                logger.warning(f"⚠️  {symbol}: {total_records} records saved, {total_errors} errors")

        except Exception as e:
            error_msg = f"Price processing failed: {e}"
            self.progress_manager.update_prices_progress(
                symbol, ProcessingStatus.FAILED, 0, error_msg
            )
            logger.error(f"❌ {symbol}: {error_msg}")

    async def _process_symbol_news(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> None:
        """Process news data for a single symbol."""
        try:
            self.progress_manager.update_news_progress(symbol, ProcessingStatus.IN_PROGRESS)

            total_records = 0
            total_errors = 0

            # Use the bulk method to get chunked news data
            async for _chunk_symbol, news_df in self.alpaca_service.fetch_bulk_news_data(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
            ):
                # Insert this chunk
                records_inserted, validation_errors = await self.insertion_service.insert_news_data(
                    symbol, news_df
                )
                total_records += records_inserted
                total_errors += validation_errors

                logger.debug(f"📰 {symbol}: +{records_inserted} news records")

            # Update progress
            if total_errors == 0:
                self.progress_manager.update_news_progress(
                    symbol, ProcessingStatus.COMPLETED, total_records
                )
                logger.info(f"✅ {symbol}: {total_records} news records completed")
            else:
                self.progress_manager.update_news_progress(
                    symbol,
                    ProcessingStatus.FAILED,
                    total_records,
                    f"{total_errors} validation errors",
                )
                logger.warning(f"⚠️  {symbol}: {total_records} records saved, {total_errors} errors")

        except Exception as e:
            error_msg = f"News processing failed: {e}"
            self.progress_manager.update_news_progress(
                symbol, ProcessingStatus.FAILED, 0, error_msg
            )
            logger.error(f"❌ {symbol}: {error_msg}")

    async def _print_final_summary(self) -> None:
        """Print comprehensive summary of the data population process."""
        summary = self.progress_manager.get_progress_summary()
        total_time = datetime.now() - self.start_time

        logger.info("\n" + "=" * 60)
        logger.info("📊 HISTORICAL DATA POPULATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total Runtime: {total_time}")
        logger.info(f"📈 Symbols Processed: {summary['total_symbols']}")
        logger.info(
            f"💰 Price Data: {summary['prices_completion_rate']:.1f}% complete "
            f"({summary['total_price_records']} records)"
        )
        logger.info(
            f"📰 News Data: {summary['news_completion_rate']:.1f}% complete "
            f"({summary['total_news_records']} records)"
        )

        # Calculate performance metrics
        records_per_second = (
            summary["total_price_records"] + summary["total_news_records"]
        ) / total_time.total_seconds()
        logger.info(f"🚀 Processing Rate: {records_per_second:.1f} records/second")

        logger.info("=" * 60)


async def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Populate historical market data")
    parser.add_argument("--symbols", nargs="+", required=True, help="Stock symbols to process")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD), defaults to today")
    parser.add_argument(
        "--data-types",
        nargs="+",
        choices=["prices", "news"],
        default=["prices", "news"],
        help="Types of data to collect",
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Start fresh (ignore existing progress)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else None

    # Initialize database
    await db_manager.initialize()

    try:
        # Create and run the populator
        populator = HistoricalDataPopulator()
        await populator.populate_data(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            data_types=args.data_types,
            resume=not args.no_resume,
        )

    except KeyboardInterrupt:
        logger.info("🛑 Population interrupted by user")
    except Exception as e:
        logger.error(f"💥 Population failed: {e}")
        raise
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
