"""Service for managing table state data using SQLite."""

import json
import logging
import os
import sqlite3 # Keep for OperationalError type hint if needed, or remove if aiosqlite.OperationalError is used directly
import asyncio # Added for retry delays
import aiosqlite # Added for async SQLite operations
from datetime import datetime
from typing import Dict, List, Optional, Any

from app.core.config import get_settings
from app.models.table_state import TableState

# Get settings
settings = get_settings()

# Set up logging
logger = logging.getLogger(__name__)

# Use the configured database path from settings
DB_PATH = settings.table_states_db_uri
logger.info(f"Using configured database path: {DB_PATH}")

# Log the database path for debugging
# logger.info(f"Table states database path: {DB_PATH}")
# logger.info(f"Absolute database path: {os.path.abspath(DB_PATH)}")

# File system operations will be moved into init_db

# Initialize the database
async def init_db():
    """Initialize the SQLite database asynchronously, including directory and file setup."""
    
    # Ensure the directory exists with proper permissions
    dir_path = os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else '.'
    try:
        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Ensured directory exists for DB: {dir_path}")
        
        # Try to set directory permissions (may fail if not running as root)
        try:
            os.chmod(dir_path, 0o755) # More restrictive permissions for the directory
            logger.info(f"Attempted to set permissions on directory: {dir_path}")
        except Exception as e:
            logger.warning(f"Could not set permissions on directory {dir_path}: {e}")
        
        # Create the database file if it doesn't exist
        if not os.path.exists(DB_PATH):
            # Create the file by opening in append mode and closing
            with open(DB_PATH, 'a') as f:
                pass # Just to create the file
            logger.info(f"Created database file: {DB_PATH}")
            
            # Try to set file permissions
            try:
                os.chmod(DB_PATH, 0o644) # More restrictive permissions for the DB file itself
                logger.info(f"Attempted to set permissions on database file: {DB_PATH}")
            except Exception as e:
                logger.warning(f"Could not set permissions on database file {DB_PATH}: {e}")
        
        # Check if we can write to the directory (optional, but good for diagnostics)
        test_file_path = os.path.join(dir_path, '.write_test_init_db')
        with open(test_file_path, 'w') as f:
            f.write('test')
        os.remove(test_file_path)
        logger.info(f"Successfully verified write access to directory {dir_path} during init_db")

    except Exception as e:
        logger.error(f"Critical error during file system setup for DB {DB_PATH}: {e}", exc_info=True)
        raise # Re-raise to allow FastAPI startup to fail fast

    # Proceed with table creation
    try:
        async with aiosqlite.connect(DB_PATH) as conn:
            await conn.execute('''
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
        logger.info(f"Initialized SQLite database at {DB_PATH}")
    except Exception as e:
        logger.error(f"Error initializing database {DB_PATH}: {e}", exc_info=True)
        # Depending on policy, you might want to raise this or handle it
        # For now, just logging, as the original code didn't explicitly raise from init_db

# Initialize the database when the module is loaded
# init_db() # IMPORTANT: This synchronous call to an async function will fail.
            # This should be called from an async context, e.g., FastAPI startup event.
            # For now, commenting out. Ensure this is handled in main.py or equivalent.

class TableStateService:
    """Service for managing table state data using SQLite."""
    
    # Cache the most recent table states
    _cache = {}
    
    @staticmethod
    async def save_table_state(table_state: TableState) -> TableState:
        """Save a table state to the SQLite database asynchronously."""
        table_state.updated_at = datetime.utcnow()
        data_json = json.dumps(table_state.data, default=str)
        
        max_retries = 3
        retry_delay = 0.1  # Initial delay in seconds

        for attempt in range(max_retries):
            try:
                async with aiosqlite.connect(DB_PATH) as conn:
                    async with conn.execute("SELECT id FROM table_states WHERE id = ?", (table_state.id,)) as cursor:
                        exists = await cursor.fetchone() is not None
                    
                    if exists:
                        await conn.execute(
                            """
                            UPDATE table_states
                            SET name = ?, data = ?, updated_at = ?
                            WHERE id = ?
                            """,
                            (table_state.name, data_json, table_state.updated_at.isoformat(), table_state.id)
                        )
                    else:
                        await conn.execute(
                            """
                            INSERT INTO table_states (id, name, user_id, data, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                table_state.id,
                                table_state.name,
                                table_state.user_id,
                                data_json,
                                table_state.created_at.isoformat(),
                                table_state.updated_at.isoformat()
                            )
                        )
                    await conn.commit()
                
                TableStateService._cache.clear()
                logger.info(f"Saved table state {table_state.id} to database")
                return table_state
            except aiosqlite.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Database locked on attempt {attempt + 1} for save {table_state.id}. Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Error saving table state {table_state.id} after {attempt + 1} attempts: {e}", exc_info=True)
                    raise
            except Exception as e: # Catch other potential errors
                logger.error(f"Unexpected error saving table state {table_state.id}: {e}", exc_info=True)
                raise
        # Should not be reached if max_retries > 0, but as a fallback:
        raise Exception(f"Failed to save table state {table_state.id} after {max_retries} retries.")


    @staticmethod
    async def get_table_state(table_id: str) -> Optional[TableState]:
        """Get a table state by ID from the SQLite database asynchronously."""
        if table_id in TableStateService._cache:
            return TableStateService._cache[table_id]
        
        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                async with conn.execute(
                    "SELECT id, name, user_id, data, created_at, updated_at FROM table_states WHERE id = ?",
                    (table_id,)
                ) as cursor:
                    row = await cursor.fetchone()
            
            if not row:
                logger.warning(f"Table state {table_id} not found in database")
                return None
            
            data = json.loads(row[3]) # Deserialization logic remains the same
            
            table_state = TableState(
                id=row[0],
                name=row[1],
                user_id=row[2],
                data=data,
                created_at=datetime.fromisoformat(row[4]), # Datetime parsing remains the same
                updated_at=datetime.fromisoformat(row[5])  # Datetime parsing remains the same
            )
            
            TableStateService._cache[table_id] = table_state
            logger.info(f"Loaded table state {table_id} from database")
            return table_state
        except Exception as e:
            logger.error(f"Error loading table state {table_id}: {e}", exc_info=True)
            return None

    @staticmethod
    async def list_table_states() -> List[TableState]:
        """List all table states from the SQLite database asynchronously."""
        table_states = []
        try:
            async with aiosqlite.connect(DB_PATH) as conn:
                async with conn.execute(
                    """
                    SELECT id, name, user_id, data, created_at, updated_at
                    FROM table_states
                    ORDER BY updated_at DESC
                    LIMIT 100
                    """
                ) as cursor:
                    rows = await cursor.fetchall()
            
            for row in rows:
                data = json.loads(row[3]) # Deserialization logic remains the same
                table_state = TableState(
                    id=row[0],
                    name=row[1],
                    user_id=row[2],
                    data=data,
                    created_at=datetime.fromisoformat(row[4]), # Datetime parsing remains the same
                    updated_at=datetime.fromisoformat(row[5])  # Datetime parsing remains the same
                )
                TableStateService._cache[table_state.id] = table_state
                table_states.append(table_state)
            
            logger.info(f"Listed {len(table_states)} table states from database")
            return table_states
        except Exception as e:
            logger.error(f"Error listing table states: {e}", exc_info=True)
            return []

    @staticmethod
    async def delete_table_state(table_id: str) -> bool:
        """Delete a table state by ID from the SQLite database asynchronously."""
        max_retries = 3
        retry_delay = 0.1  # Initial delay in seconds

        for attempt in range(max_retries):
            try:
                async with aiosqlite.connect(DB_PATH) as conn:
                    async with conn.execute("SELECT id FROM table_states WHERE id = ?", (table_id,)) as cursor:
                        exists = await cursor.fetchone() is not None
                    
                    if not exists:
                        logger.warning(f"Table state {table_id} not found in database for deletion")
                        return False # No need to retry if not found
                    
                    await conn.execute("DELETE FROM table_states WHERE id = ?", (table_id,))
                    await conn.commit()
                
                if table_id in TableStateService._cache:
                    del TableStateService._cache[table_id]
                logger.info(f"Deleted table state {table_id} from database")
                return True
            except aiosqlite.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Database locked on attempt {attempt + 1} for delete {table_id}. Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Error deleting table state {table_id} after {attempt + 1} attempts: {e}", exc_info=True)
                    return False # Return False on persistent failure
            except Exception as e: # Catch other potential errors
                logger.error(f"Unexpected error deleting table state {table_id}: {e}", exc_info=True)
                return False
        return False # Should be reached only if all retries fail
