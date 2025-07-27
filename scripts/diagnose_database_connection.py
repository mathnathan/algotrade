# scripts/diagnose_database_connection.py
"""
Diagnostic script to troubleshoot database connection and driver issues.

This helps us understand exactly what SQLAlchemy is seeing and why it's
choosing the wrong driver.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_environment():
    """Check environment variables and their values."""
    print("🔍 Environment Diagnostic")
    print("=" * 50)

    database_url = os.getenv("DATABASE_URL")
    print(f"DATABASE_URL from environment: {database_url}")

    if not database_url:
        print("❌ DATABASE_URL not set in environment")
        print("💡 Tip: Create a .env file or set the environment variable")
        return False

    return True


def check_drivers():
    """Check which database drivers are available."""
    print("\n🚗 Driver Availability Check")
    print("=" * 50)

    # Check asyncpg (what we want)
    try:
        import asyncpg

        print(f"✅ asyncpg available: version {asyncpg.__version__}")
        asyncpg_available = True
    except ImportError as e:
        print(f"❌ asyncpg not available: {e}")
        asyncpg_available = False

    # Check psycopg2 (what's being used incorrectly)
    try:
        import psycopg2

        print(f"⚠️  psycopg2 available: version {psycopg2.__version__}")
        print("   (This is the synchronous driver that's causing conflicts)")
        psycopg2_available = True
    except ImportError:
        print("✅ psycopg2 not available (this is fine for async operations)")
        psycopg2_available = False

    return asyncpg_available, psycopg2_available


def check_sqlalchemy_url_parsing():
    """Test how SQLAlchemy parses different connection string formats."""
    print("\n🔗 Connection String Parsing Test")
    print("=" * 50)

    try:
        from sqlalchemy import make_url

        test_urls = [
            "postgresql://trading_user:secure_trading_password@localhost:5433/trading_db",
            "postgresql+asyncpg://trading_user:secure_trading_password@localhost:5433/trading_db",
            "postgresql+psycopg2://trading_user:secure_trading_password@localhost:5433/trading_db",
        ]

        for url_string in test_urls:
            try:
                parsed_url = make_url(url_string)
                print(f"URL: {url_string}")
                print(f"  Drivername: {parsed_url.drivername}")
                print(f"  Driver: {parsed_url.get_driver_name()}")
                print()
            except Exception as e:
                print(f"❌ Failed to parse {url_string}: {e}")

    except ImportError as e:
        print(f"❌ Cannot import SQLAlchemy: {e}")


def check_settings_loading():
    """Check how our settings are loading the database URL."""
    print("\n⚙️  Settings Loading Test")
    print("=" * 50)

    try:
        from src.config.settings import settings

        print(f"Settings database_url: {settings.database_url}")

        # Test the URL transformation logic
        original_url = settings.database_url
        if "postgresql://" in original_url and "+asyncpg" not in original_url:
            transformed_url = original_url.replace("postgresql://", "postgresql+asyncpg://")
            print(f"Original URL: {original_url}")
            print(f"Transformed URL: {transformed_url}")
        else:
            print("URL already contains asyncpg driver specification")

    except Exception as e:
        print(f"❌ Failed to load settings: {e}")


def main():
    """Run all diagnostic checks."""
    print("🩺 Database Connection Diagnostic Tool")
    print("=" * 60)

    env_ok = check_environment()
    asyncpg_available, psycopg2_available = check_drivers()
    check_sqlalchemy_url_parsing()
    check_settings_loading()

    print("\n📋 Summary and Recommendations")
    print("=" * 50)

    if not env_ok:
        print("❌ Environment setup issues detected")
        return False

    if not asyncpg_available:
        print("❌ asyncpg driver not available - install it first")
        return False

    if psycopg2_available:
        print("⚠️  Both sync and async drivers present - this can cause conflicts")
        print("💡 Consider removing psycopg2-binary if you only need async operations")

    print("✅ Basic setup looks good - checking URL transformation logic...")
    return True


if __name__ == "__main__":
    main()
