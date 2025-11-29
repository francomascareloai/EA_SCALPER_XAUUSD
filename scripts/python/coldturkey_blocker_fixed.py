"""
Get Pro Subscription of Cold Turkey Blocker - Fixed for Windows.

Author: Anshul Raj Verma (Modified)
Github: https://github.com/arv-anshul/ColdTurkeyBlocker-Pro
"""

import json
import logging
import os
import sqlite3
import typing
from pathlib import Path

MAC_DB_PATH = Path("/Library/Application Support/Cold Turkey/data-app.db")
WIN_DB_PATH = Path("C:/Program Files/Cold Turkey/data-app.db")
SystemType = typing.Literal["mac", "win"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s -> %(message)s")


def configure_db_path(system_type: SystemType) -> Path:
    db_path = MAC_DB_PATH if system_type == "mac" else WIN_DB_PATH
    if not db_path.exists():
        logging.info(f"{db_path = !s}")
        logging.error("Database does not exists.")
        exit(1)
    return db_path


def kill_blocker(system_type: SystemType):
    if system_type == "mac":
        kill_blocker_command = "/usr/bin/killall 'Cold Turkey Blocker'"
        os.system(kill_blocker_command)  # noqa: S605
        logging.critical("Open Blocker App")
    else:
        # For Windows, try to kill the process
        try:
            os.system('taskkill /f /im "Cold Turkey Blocker.exe" 2>nul')
            os.system('taskkill /f /im "coldturkey.exe" 2>nul')
            logging.critical("Cold Turkey processes terminated. Please restart the application.")
        except:
            logging.critical("Please manually close Cold Turkey Blocker and restart it.")


def list_all_settings(c: sqlite3.Cursor) -> None:
    """Debug function to see all available settings."""
    logging.info("Listing all settings in the database:")
    rows = c.execute("SELECT key, value FROM settings").fetchall()
    for key, value in rows:
        logging.info(f"Key: {key}")
        if len(str(value)) > 200:
            logging.info(f"Value (truncated): {str(value)[:200]}...")
        else:
            logging.info(f"Value: {value}")
        logging.info("-" * 50)


def upgrade_blocker(c: sqlite3.Cursor) -> None:
    try:
        # First, let's see what settings are available
        result = c.execute("SELECT value FROM settings WHERE key = 'settings'").fetchone()
        
        if result is None:
            logging.error("No 'settings' key found in database.")
            logging.info("Let's see what keys are available:")
            list_all_settings(c)
            return
        
        s = result[0]
        
        if not s:
            logging.error("Settings value is empty.")
            return
            
        logging.info(f"Raw settings data: {s[:200]}..." if len(s) > 200 else f"Raw settings data: {s}")
        
        data = json.loads(s)
        
        # Check if the expected structure exists
        if "additional" not in data:
            logging.error("'additional' key not found in settings.")
            logging.info(f"Available keys: {list(data.keys())}")
            return
            
        if "proStatus" not in data["additional"]:
            logging.error("'proStatus' not found in additional settings.")
            logging.info(f"Available keys in additional: {list(data['additional'].keys())}")
            return

        blocker_status = data["additional"]["proStatus"]
        logging.info(f"Current blocker status: {blocker_status!r}")

        if blocker_status == "pro":
            logging.info("Changing status from 'pro' to 'free'.")
            data["additional"]["proStatus"] = "free"
        elif blocker_status == "free":
            logging.info("Changing status from 'free' to 'pro'.")
            data["additional"]["proStatus"] = "pro"
        else:
            logging.warning(f"Unknown status '{blocker_status}'. Setting to 'pro'.")
            data["additional"]["proStatus"] = "pro"

        c.execute(
            "UPDATE settings SET value = ? WHERE key = 'settings'",
            (json.dumps(data),),
        )
        
        logging.info("Settings updated successfully!")
        
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON: {e}")
        logging.error("The settings data might be corrupted or in a different format.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")


def main():
    print("Cold Turkey Blocker Pro Activator")
    print("=" * 40)
    
    while True:
        system_type: SystemType = input("Choose your system [mac|win]: ").lower().strip()
        if system_type in ["mac", "win"]:
            break
        logging.warning("Please choose either 'mac' or 'win'")

    if system_type == "win":
        logging.info("Running on Windows - using enhanced error handling.")

    db_path = configure_db_path(system_type)
    logging.info(f"Using database path: {db_path}")
    
    try:
        # Make a backup first
        backup_path = db_path.with_suffix('.db.backup')
        logging.info(f"Creating backup at: {backup_path}")
        
        import shutil
        shutil.copy2(db_path, backup_path)
        logging.info("Backup created successfully!")
        
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            
            # Check database structure
            tables = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            logging.info(f"Available tables: {[table[0] for table in tables]}")
            
            upgrade_blocker(c)
            
            logging.info("Committing changes to database...")
            conn.commit()
            
            logging.info("Changes committed successfully!")
            kill_blocker(system_type)
            
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        logging.error("Failed to connect with Cold Turkey blocker database.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
