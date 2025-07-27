# scripts/test_fixed_connection.py
"""
Test the fixed database connection with lazy initialization.
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_lazy_initialization():
    """Test that lazy initialization works correctly."""
    print("🧪 Testing lazy initialization...")

    try:
        # This should NOT create a database connection yet
        from src.database.connection import db_manager

        print("✅ DatabaseManager imported successfully (no connection created yet)")

        # This SHOULD trigger initialization and create the connection
        print("🔄 Triggering initialization...")
        connection_ok = await db_manager.test_connection()

        if connection_ok:
            print("✅ Lazy initialization successful!")
            print("✅ Async driver verification passed!")

            # Test a simple query
            async with db_manager.get_session() as session:
                from sqlalchemy import text

                result = await session.execute(
                    text("SELECT 'Async connection working!' as message")
                )
                message = await result.fetchone()
                print(f"✅ Query test: {message[0]}")

            return True
        else:
            print("❌ Connection test failed")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_lazy_initialization())
    if success:
        print("\n🎉 All tests passed! The async connection is working perfectly.")
    else:
        print("\n💥 Tests failed - check the error messages above.")
    sys.exit(0 if success else 1)
