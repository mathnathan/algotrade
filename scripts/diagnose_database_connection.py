# scripts/enhanced_database_diagnostics.py
"""
Database diagnostic script with detailed error reporting.

This script helps you understand exactly what's happening with your database
connections and provides actionable insights for troubleshooting.
"""

import asyncio
import sys
from pathlib import Path

from sqlalchemy import text

from src.config.logging import setup_logging
from src.database.connection import db_manager

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = setup_logging()


async def comprehensive_database_test():
    """
    Run comprehensive database diagnostics with detailed reporting.

    Think of this as a full trading system pre-flight checklist - we test
    every component that your algorithmic trading system depends on.
    """
    print("🔍 Starting Comprehensive Database Diagnostics")
    print("=" * 60)

    test_results = {
        "basic_connection": False,
        "session_management": False,
        "transaction_handling": False,
        "concurrent_sessions": False,
        "error_recovery": False,
    }

    # Test 1: Basic Connection
    print("\n1️⃣ Testing Basic Connection...")
    try:
        connection_ok = await db_manager.test_connection()
        test_results["basic_connection"] = connection_ok
        print(
            f"   {'✅' if connection_ok else '❌'} Basic connection: {'PASS' if connection_ok else 'FAIL'}"
        )

        if connection_ok:
            pool_status = await db_manager.get_pool_status()
            print(f"   📊 Pool status: {pool_status}")

    except Exception as e:
        print(f"   ❌ Basic connection failed: {type(e).__name__}: {e}")

    # Test 2: Session Management
    print("\n2️⃣ Testing Session Management...")
    try:
        async with db_manager.get_session() as session:
            result = await session.execute(text("SELECT 'Session test successful' as message"))
            message = result.scalar()
            test_results["session_management"] = message == "Session test successful"
            print("   ✅ Session management: PASS")
            print(f"   📝 Session details: ID={id(session)}, Active={session.is_active}")

    except Exception as e:
        print(f"   ❌ Session management failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    # Test 3: Transaction Handling
    print("\n3️⃣ Testing Transaction Handling...")
    try:
        async with db_manager.get_session() as session:
            # Test that we can handle transactions properly
            await session.execute(text("CREATE TEMP TABLE test_tx AS SELECT 1 as test_col"))
            result = await session.execute(text("SELECT test_col FROM test_tx"))
            test_value = result.scalar()

            test_results["transaction_handling"] = test_value == 1
            print("   ✅ Transaction handling: PASS")

    except Exception as e:
        print(f"   ❌ Transaction handling failed: {type(e).__name__}: {e}")

    # Test 4: Concurrent Sessions
    print("\n4️⃣ Testing Concurrent Sessions...")
    try:

        async def concurrent_query(session_id: int):
            async with db_manager.get_session() as session:
                await session.execute(text(f"SELECT {session_id} as session_id"))
                return session_id

        # Run 5 concurrent sessions
        tasks = [concurrent_query(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        test_results["concurrent_sessions"] = len(results) == 5
        print(f"   ✅ Concurrent sessions: PASS ({len(results)}/5 sessions completed)")

    except Exception as e:
        print(f"   ❌ Concurrent sessions failed: {type(e).__name__}: {e}")

    # Test 5: Error Recovery
    print("\n5️⃣ Testing Error Recovery...")
    try:
        # Test that errors don't leave sessions in bad state
        try:
            async with db_manager.get_session() as session:
                # Intentionally cause an error
                await session.execute(text("SELECT * FROM nonexistent_table"))
        except Exception as e:
            logger.debug(f"Expected error during error recovery test: {type(e).__name__}: {e}")

        # Now test that we can still use the database normally
        async with db_manager.get_session() as session:
            result = await session.execute(text("SELECT 'Recovery successful' as message"))
            message = result.scalar()
            test_results["error_recovery"] = message == "Recovery successful"
            print("   ✅ Error recovery: PASS")

    except Exception as e:
        print(f"   ❌ Error recovery failed: {type(e).__name__}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)

    passed_tests = sum(test_results.values())
    total_tests = len(test_results)

    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")

    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All database tests passed! Your system is ready for trading operations.")
    else:
        print("⚠️  Some tests failed. Review the errors above before proceeding.")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = asyncio.run(comprehensive_database_test())
    sys.exit(0 if success else 1)
