# scripts/test_entrypoint.py
"""
Test script to validate our entrypoint logic before running in Docker.

This allows us to debug initialization issues in a local Python environment
before dealing with container complexity.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.logging import setup_logging
from src.database.migrations import init_alembic

logger = setup_logging()

def test_alembic_initialization():
    """Test that Alembic initialization works correctly."""
    try:
        logger.info("🧪 Testing Alembic initialization...")

        # Test initialization
        result = init_alembic()

        if result:
            logger.info("✅ Alembic initialization test passed")

            # Verify files were created
            alembic_dir = project_root / "alembic"
            alembic_ini = project_root / "alembic.ini"

            if alembic_dir.exists() and alembic_ini.exists():
                logger.info("✅ Alembic files created successfully")
                logger.info(f"📁 {alembic_dir}")
                logger.info(f"⚙️  {alembic_ini}")
            else:
                logger.error("❌ Alembic files not found after initialization")
                return False

        return True

    except Exception as e:
        logger.error(f"❌ Alembic initialization test failed: {e}")
        return False

if __name__ == "__main__":
    if test_alembic_initialization():
        print("🎉 Entrypoint logic is ready for Docker!")
    else:
        print("💥 Issues found - fix before deploying")
        sys.exit(1)
