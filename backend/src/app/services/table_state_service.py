"""Service for managing table state data using SQLite."""

import json
import logging
import os
import sqlite3
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

# Ensure the directory exists with proper permissions
dir_path = os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else '.'
try:
    # Create directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)
    logger.info(f"Ensured directory exists: {dir_path}")
    
    # Try to set directory permissions (may fail if not running as root)
    try:
        os.chmod(dir_path, 0o777)
        logger.info(f"Set permissions on directory: {dir_path}")
    except Exception as e:
        logger.warning(f"Could not set permissions on directory {dir_path}: {e}")
    
    # Create the database file if it doesn't exist
    if not os.path.exists(DB_PATH):
        open(DB_PATH, 'a').close()
        logger.info(f"Created database file: {DB_PATH}")
        
        # Try to set file permissions
        try:
            os.chmod(DB_PATH, 0o666)
            logger.info(f"Set permissions on database file: {DB_PATH}")
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

# Initialize the database
def init_db():
    """Initialize the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create the table_states table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS table_states (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        user_id TEXT,
        data TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()
    
    logger.info(f"Initialized SQLite database at {DB_PATH}")

# Initialize the database when the module is loaded
init_db()


class TableStateService:
    """Service for managing table state data using SQLite."""
    
    @staticmethod
    def save_table_state(table_state: TableState) -> TableState:
        """Save a table state to the SQLite database."""
        # Update the updated_at timestamp
        table_state.updated_at = datetime.utcnow()
        
        # Convert the data to a JSON string
        data_json = json.dumps(table_state.data, default=str)
        
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            # Check if the table state already exists
            cursor.execute("SELECT id FROM table_states WHERE id = ?", (table_state.id,))
            exists = cursor.fetchone() is not None
            
            if exists:
                # Update the existing table state
                cursor.execute(
                    "UPDATE table_states SET name = ?, user_id = ?, data = ?, updated_at = ? WHERE id = ?",
                    (
                        table_state.name,
                        table_state.user_id,
                        data_json,
                        table_state.updated_at.isoformat(),
                        table_state.id
                    )
                )
                logger.info(f"Updated table state {table_state.id} in database")
            else:
                # Insert a new table state
                cursor.execute(
                    "INSERT INTO table_states (id, name, user_id, data, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        table_state.id,
                        table_state.name,
                        table_state.user_id,
                        data_json,
                        table_state.created_at.isoformat(),
                        table_state.updated_at.isoformat()
                    )
                )
                logger.info(f"Inserted new table state {table_state.id} into database")
            
            # Commit the transaction
            conn.commit()
            
            return table_state
        except Exception as e:
            # Rollback the transaction on error
            conn.rollback()
            logger.error(f"Error saving table state {table_state.id}: {e}")
            raise
        finally:
            # Close the connection
            conn.close()
    
    @staticmethod
    def get_table_state(table_id: str) -> Optional[TableState]:
        """Get a table state by ID from the SQLite database."""
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            # Query the table state
            cursor.execute(
                "SELECT id, name, user_id, data, created_at, updated_at FROM table_states WHERE id = ?",
                (table_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                logger.warning(f"Table state {table_id} not found in database")
                return None
            
            # Parse the JSON data
            data = json.loads(row[3])
            
            # Create a TableState object
            table_state = TableState(
                id=row[0],
                name=row[1],
                user_id=row[2],
                data=data,
                created_at=datetime.fromisoformat(row[4]),
                updated_at=datetime.fromisoformat(row[5])
            )
            
            logger.info(f"Loaded table state {table_id} from database")
            
            return table_state
        except Exception as e:
            logger.error(f"Error loading table state {table_id}: {e}")
            return None
        finally:
            # Close the connection
            conn.close()
    
    @staticmethod
    def list_table_states() -> List[TableState]:
        """List all table states from the SQLite database."""
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            # Query all table states
            cursor.execute(
                "SELECT id, name, user_id, data, created_at, updated_at FROM table_states ORDER BY updated_at DESC"
            )
            rows = cursor.fetchall()
            
            # Convert rows to TableState objects
            table_states = []
            for row in rows:
                # Parse the JSON data
                data = json.loads(row[3])
                
                # Create a TableState object
                table_state = TableState(
                    id=row[0],
                    name=row[1],
                    user_id=row[2],
                    data=data,
                    created_at=datetime.fromisoformat(row[4]),
                    updated_at=datetime.fromisoformat(row[5])
                )
                
                table_states.append(table_state)
            
            logger.info(f"Listed {len(table_states)} table states from database")
            
            return table_states
        except Exception as e:
            logger.error(f"Error listing table states: {e}")
            return []
        finally:
            # Close the connection
            conn.close()
    
    @staticmethod
    def delete_table_state(table_id: str) -> bool:
        """Delete a table state by ID from the SQLite database."""
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            # Check if the table state exists
            cursor.execute("SELECT id FROM table_states WHERE id = ?", (table_id,))
            exists = cursor.fetchone() is not None
            
            if not exists:
                logger.warning(f"Table state {table_id} not found in database")
                return False
            
            # Delete the table state
            cursor.execute("DELETE FROM table_states WHERE id = ?", (table_id,))
            
            # Commit the transaction
            conn.commit()
            
            logger.info(f"Deleted table state {table_id} from database")
            
            return True
        except Exception as e:
            # Rollback the transaction on error
            conn.rollback()
            logger.error(f"Error deleting table state {table_id}: {e}")
            return False
        finally:
            # Close the connection
            conn.close()
