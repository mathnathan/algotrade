from datetime import datetime, time, timedelta

import pytz


class MarketHours:
    """Handle market hours and trading windows for US equity markets."""

    EST = pytz.timezone("US/Eastern")

    # Market hours in ET
    PREMARKET_START = time(4, 0)  # 4:00 AM ET
    MARKET_OPEN = time(9, 30)  # 9:30 AM ET
    MARKET_CLOSE = time(16, 0)  # 4:00 PM ET
    AFTERHOURS_END = time(20, 0)  # 8:00 PM ET

    # Your strategy windows
    PREDICTION_TIME = time(12, 0)  # 12:00 PM ET - when to make prediction
    ENTRY_TIME = time(12, 0)  # 12:00 PM ET - when to enter position
    EXIT_TIME = time(15, 55)  # 3:55 PM ET - when to exit (5 min before close)

    @classmethod
    def now_et(cls) -> datetime:
        """Get current time in Eastern Time."""
        return datetime.now(cls.EST)

    @classmethod
    def is_market_open(cls, dt: datetime | None = None) -> bool:
        """Check if market is currently open (regular hours)."""
        if dt is None:
            dt = cls.now_et()

        # Convert to ET if needed
        if dt.tzinfo != cls.EST:
            dt = dt.astimezone(cls.EST)

        # Check if weekday (Monday=0, Sunday=6)
        if dt.weekday() >= 5:  # Saturday or Sunday
            return False

        current_time = dt.time()
        return cls.MARKET_OPEN <= current_time <= cls.MARKET_CLOSE

    @classmethod
    def is_extended_hours(cls, dt: datetime | None = None) -> bool:
        """Check if extended hours trading is available."""
        if dt is None:
            dt = cls.now_et()

        if dt.tzinfo != cls.EST:
            dt = dt.astimezone(cls.EST)

        if dt.weekday() >= 5:  # Weekend
            return False

        current_time = dt.time()
        return cls.PREMARKET_START <= current_time <= cls.AFTERHOURS_END

    @classmethod
    def next_prediction_time(cls) -> datetime:
        """Get the next 12:00 PM ET on a trading day."""
        now = cls.now_et()

        # If it's before 12 PM today and it's a weekday, use today
        if now.weekday() < 5 and now.time() < cls.PREDICTION_TIME:
            return now.replace(
                hour=cls.PREDICTION_TIME.hour,
                minute=cls.PREDICTION_TIME.minute,
                second=0,
                microsecond=0,
            )

        # Otherwise, find next weekday at 12 PM
        days_ahead = 1
        while True:
            next_day = now + timedelta(days=days_ahead)
            if next_day.weekday() < 5:  # Monday-Friday
                return next_day.replace(
                    hour=cls.PREDICTION_TIME.hour,
                    minute=cls.PREDICTION_TIME.minute,
                    second=0,
                    microsecond=0,
                )
            days_ahead += 1

    @classmethod
    def get_trading_window(cls, dt: datetime) -> tuple[datetime, datetime]:
        """Get entry and exit times for a given trading day."""
        if dt.tzinfo != cls.EST:
            dt = dt.astimezone(cls.EST)

        entry_time = dt.replace(
            hour=cls.ENTRY_TIME.hour, minute=cls.ENTRY_TIME.minute, second=0, microsecond=0
        )

        exit_time = dt.replace(
            hour=cls.EXIT_TIME.hour, minute=cls.EXIT_TIME.minute, second=0, microsecond=0
        )

        return entry_time, exit_time
