"""Service for managing table state data using SQLite."""

import json
import logging
import os
import aiosqlite
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio

from app.core.config import get_settings
from app.models.table_state import TableState

# Get settings
settings = get_settings()

# Set up logging
logger = logging.getLogger(__name__)

# Use the configured database path from settings
DB_PATH = settings.table_states_db_uri
logger.info(f"Using configured database path for aiosqlite: {DB_PATH}")

# Log the database path for debugging
# logger.info(f"Table states database path: {DB_PATH}")
# logger.info(f"Absolute database path: {os.path.abspath(DB_PATH)}")

# Ensure the directory exists with proper permissions
dir_path = os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else '.'
try:
    # Create directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)
    logger.info(f"Ensured directory exists: {dir_path}")
    
    # Try to set directory permissions (may fail if not running as root)
    try:
        os.chmod(dir_path, 0o777)
        logger.info(f"Attempted to set permissions on directory: {dir_path}")
    except Exception as e:
        logger.warning(f"Could not set permissions on directory {dir_path}: {e}")
    
    # Create the database file if it doesn't exist
    if not os.path.exists(DB_PATH):
        open(DB_PATH, 'a').close()
        logger.info(f"Created database file: {DB_PATH}")
        
        # Try to set file permissions
        try:
            os.chmod(DB_PATH, 0o666)
            logger.info(f"Attempted to set permissions on database file: {DB_PATH}")
        except Exception as e:
            logger.warning(f"Could not set permissions on database file {DB_PATH}: {e}")
    
    # Check if we can write to the directory
    test_file_path = os.path.join(dir_path, '.write_test')
    with open(test_file_path, 'w') as f:
        f.write('test')
    os.remove(test_file_path)
    logger.info(f"Successfully verified write access to {dir_path}")
except Exception as e:
    logger.error(f"Cannot create or access directory {dir_path}: {e}")
    logger.error(f"This will cause database operations to fail!")

async def init_db_async():
    """Initialize the SQLite database asynchronously."""
    async with aiosqlite.connect(DB_PATH) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('''
            CREATE TABLE IF NOT EXISTS table_states (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                user_id TEXT,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            ''')
        await conn.commit()
    logger.info(f"Asynchronously initialized SQLite database at {DB_PATH}")

class TableStateService:
    """Service for managing table state data using SQLite."""
    
    @staticmethod
    async def save_table_state(table_state: TableState) -> TableState:
        """Save a table state to the SQLite database."""
        table_state.updated_at = datetime.utcnow()
        data_json = json.dumps(table_state.data, default=str)
        
        async with aiosqlite.connect(DB_PATH) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT id FROM table_states WHERE id = ?", (table_state.id,))
                exists = await cursor.fetchone() is not None
                
                if exists:
                    await cursor.execute(
                        """UPDATE table_states SET name = ?, data = ?, updated_at = ? WHERE id = ?""",
                        (table_state.name, data_json, table_state.updated_at.isoformat(), table_state.id)
                    )
                else:
                    await cursor.execute(
                        """INSERT INTO table_states (id, name, user_id, data, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            table_state.id, table_state.name, table_state.user_id,
                            data_json, table_state.created_at.isoformat(),
                            table_state.updated_at.isoformat()
                        )
                    )
                await conn.commit()
            
            logger.info(f"Saved table state {table_state.id} to database via aiosqlite")
            return table_state
    
    @staticmethod
    async def get_table_state(table_id: str) -> Optional[TableState]:
        """Get a table state by ID from the SQLite database."""
        async with aiosqlite.connect(DB_PATH) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "SELECT id, name, user_id, data, created_at, updated_at FROM table_states WHERE id = ?",
                    (table_id,)
                )
                row = await cursor.fetchone()
                if not row:
                    logger.warning(f"Table state {table_id} not found in database (aiosqlite)")
                    return None
                data = json.loads(row[3])
                table_state = TableState(
                    id=row[0], name=row[1], user_id=row[2], data=data,
                    created_at=datetime.fromisoformat(row[4]),
                    updated_at=datetime.fromisoformat(row[5])
                )
                logger.info(f"Loaded table state {table_id} from database (aiosqlite)")
                return table_state
    
    @staticmethod
    async def list_table_states() -> List[Dict[str, Any]]:
        """List all table states (essential fields only) from the SQLite database."""
        states_list = []
        async with aiosqlite.connect(DB_PATH) as conn:
            # Select only needed columns, EXCLUDE 'data'
            conn.row_factory = aiosqlite.Row # Return rows as dictionary-like objects
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "SELECT id, name, user_id, created_at, updated_at FROM table_states ORDER BY updated_at DESC"
                )
                rows = await cursor.fetchall()
                for row in rows:
                    # Convert row to dictionary explicitly for clarity
                    states_list.append(dict(row))

        logger.info(f"Loaded {len(states_list)} table state items (without data) from database (aiosqlite)")
        # The endpoint will automatically validate these dicts against TableStateListItem
        return states_list
    
    @staticmethod
    async def delete_table_state(table_id: str) -> bool:
        """Delete a table state by ID from the SQLite database."""
        async with aiosqlite.connect(DB_PATH) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("DELETE FROM table_states WHERE id = ?", (table_id,))
                await conn.commit()
                deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted table state {table_id} from database (aiosqlite)")
            else:
                logger.warning(f"Table state {table_id} not found for deletion (aiosqlite).")
            return deleted
