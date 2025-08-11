# src/services/progress_manager.py

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Status tracking for data processing operations."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SymbolProgress:
    """Track progress for a single symbol's data collection."""

    symbol: str
    start_date: datetime
    end_date: datetime
    prices_status: ProcessingStatus = ProcessingStatus.PENDING
    news_status: ProcessingStatus = ProcessingStatus.PENDING
    prices_records_collected: int = 0
    news_records_collected: int = 0
    last_processed_date: datetime | None = None
    error_count: int = 0
    last_error: str | None = None
    processing_time_seconds: float = 0.0


class DataPopulationProgressManager:
    """
    Manages progress tracking for large-scale data population operations.

    Financial Concept: Operational Resilience
    Professional trading operations require systems that can recover gracefully
    from failures. In data collection, this means:

    1. Persistent State: Track exactly what's been processed
    2. Resumption Logic: Continue from the last successful point
    3. Error Tracking: Monitor which operations are consistently failing
    4. Performance Metrics: Understand how long operations take

    Think of this like a mission control system for a space launch - it tracks
    every system, monitors for problems, and ensures the mission can continue
    even if individual components fail.
    """

    def __init__(self, progress_file: Path = Path("data_population_progress.json")):
        self.progress_file = progress_file
        self.progress: dict[str, SymbolProgress] = {}
        self.session_start_time = datetime.now()
        self.load_progress()

    def load_progress(self) -> None:
        """Load existing progress from disk."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    data = json.load(f)

                # Convert dict back to SymbolProgress objects
                for symbol, progress_data in data.get("symbols", {}).items():
                    # Convert datetime strings back to datetime objects
                    if progress_data.get("start_date"):
                        progress_data["start_date"] = datetime.fromisoformat(
                            progress_data["start_date"]
                        )
                    if progress_data.get("end_date"):
                        progress_data["end_date"] = datetime.fromisoformat(
                            progress_data["end_date"]
                        )
                    if progress_data.get("last_processed_date"):
                        progress_data["last_processed_date"] = datetime.fromisoformat(
                            progress_data["last_processed_date"]
                        )

                    # Convert status strings back to enums
                    progress_data["prices_status"] = ProcessingStatus(
                        progress_data["prices_status"]
                    )
                    progress_data["news_status"] = ProcessingStatus(progress_data["news_status"])

                    self.progress[symbol] = SymbolProgress(**progress_data)

                logger.info(f"📋 Loaded progress for {len(self.progress)} symbols")

            except Exception as e:
                logger.warning(f"⚠️  Could not load progress file: {e}. Starting fresh.")
                self.progress = {}
        else:
            logger.info("📋 No existing progress file found. Starting fresh.")

    def save_progress(self) -> None:
        """Save current progress to disk."""
        try:
            # Convert to JSON-serializable format
            save_data: dict[str, any] = {
                "session_start_time": self.session_start_time.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "symbols": {},
            }

            for symbol, progress in self.progress.items():
                progress_dict = asdict(progress)
                # Convert datetime objects to ISO strings
                for field in ["start_date", "end_date", "last_processed_date"]:
                    if progress_dict[field]:
                        progress_dict[field] = progress_dict[field].isoformat()
                # Convert enums to strings
                progress_dict["prices_status"] = progress_dict["prices_status"].value
                progress_dict["news_status"] = progress_dict["news_status"].value

                save_data["symbols"][symbol] = progress_dict

            # Atomic write (write to temp file, then rename)
            temp_file = self.progress_file.with_suffix(".json.tmp")
            with open(temp_file, "w") as f:
                json.dump(save_data, f, indent=2)

            temp_file.rename(self.progress_file)
            logger.debug("💾 Progress saved successfully")

        except Exception as e:
            logger.error(f"❌ Failed to save progress: {e}")

    def initialize_symbol(self, symbol: str, start_date: datetime, end_date: datetime) -> None:
        """Initialize tracking for a new symbol."""
        if symbol not in self.progress:
            self.progress[symbol] = SymbolProgress(
                symbol=symbol, start_date=start_date, end_date=end_date
            )
            logger.debug(f"📊 Initialized tracking for {symbol}")

    def update_prices_progress(
        self,
        symbol: str,
        status: ProcessingStatus,
        records_added: int = 0,
        error: str | None = None,
    ) -> None:
        """Update progress for price data collection."""
        if symbol in self.progress:
            progress = self.progress[symbol]
            progress.prices_status = status
            progress.prices_records_collected += records_added
            progress.last_processed_date = datetime.now()

            if error:
                progress.error_count += 1
                progress.last_error = error

            self.save_progress()

    def update_news_progress(
        self,
        symbol: str,
        status: ProcessingStatus,
        records_added: int = 0,
        error: str | None = None,
    ) -> None:
        """Update progress for news data collection."""
        if symbol in self.progress:
            progress = self.progress[symbol]
            progress.news_status = status
            progress.news_records_collected += records_added
            progress.last_processed_date = datetime.now()

            if error:
                progress.error_count += 1
                progress.last_error = error

            self.save_progress()

    def get_pending_symbols(self, data_type: str = "both") -> list[str]:
        """Get list of symbols that still need processing."""
        pending = []

        for symbol, progress in self.progress.items():
            if data_type in ["prices", "both"] and progress.prices_status in [
                ProcessingStatus.PENDING,
                ProcessingStatus.FAILED,
            ]:
                pending.append(symbol)
                continue

            if data_type in ["news", "both"] and progress.news_status in [
                ProcessingStatus.PENDING,
                ProcessingStatus.FAILED,
            ]:
                pending.append(symbol)

        return list(set(pending))

    def get_progress_summary(self) -> dict:
        """Get summary of overall progress."""
        total_symbols = len(self.progress)
        if total_symbols == 0:
            return {"total_symbols": 0, "message": "No symbols initialized"}

        prices_completed = sum(
            1 for p in self.progress.values() if p.prices_status == ProcessingStatus.COMPLETED
        )
        news_completed = sum(
            1 for p in self.progress.values() if p.news_status == ProcessingStatus.COMPLETED
        )

        total_price_records = sum(p.prices_records_collected for p in self.progress.values())
        total_news_records = sum(p.news_records_collected for p in self.progress.values())

        return {
            "total_symbols": total_symbols,
            "prices_completed": prices_completed,
            "news_completed": news_completed,
            "prices_completion_rate": prices_completed / total_symbols * 100,
            "news_completion_rate": news_completed / total_symbols * 100,
            "total_price_records": total_price_records,
            "total_news_records": total_news_records,
            "session_runtime": (datetime.now() - self.session_start_time).total_seconds(),
        }
