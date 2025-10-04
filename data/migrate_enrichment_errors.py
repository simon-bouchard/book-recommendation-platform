# data/migrate_enrichment_errors.py
"""
Migration script to update enrichment_errors table from simple structure
to Phase 1 structured error tracking.

Run this ONCE before deploying Phase 1 Kafka pipeline.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import engine
import pymysql

def migrate_enrichment_errors():
    """
    Migrate enrichment_errors table to Phase 1 structure.
    Safe to run multiple times (checks if columns exist).
    """
    conn = pymysql.connect(
        host=engine.url.host or "127.0.0.1",
        port=engine.url.port or 3306,
        user=engine.url.username,
        password=engine.url.password,
        database=engine.url.database,
        charset="utf8mb4"
    )
    
    try:
        cur = conn.cursor()
        
        print("Checking enrichment_errors table structure...")
        
        # Check if new columns already exist
        cur.execute("""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = %s 
            AND TABLE_NAME = 'enrichment_errors'
        """, (engine.url.database,))
        
        existing_columns = {row[0] for row in cur.fetchall()}
        
        if 'stage' in existing_columns:
            print("Table already migrated. Skipping.")
            return
        
        print("Migrating enrichment_errors table...")
        
        # Add new columns
        migrations = [
            # Timestamps
            """ALTER TABLE enrichment_errors 
               ADD COLUMN first_seen_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP""",
            
            """ALTER TABLE enrichment_errors 
               ADD COLUMN last_seen_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP 
               ON UPDATE CURRENT_TIMESTAMP""",
            
            """ALTER TABLE enrichment_errors 
               ADD COLUMN occurrence_count INT NOT NULL DEFAULT 1""",
            
            # Error classification
            """ALTER TABLE enrichment_errors 
               ADD COLUMN stage VARCHAR(64) NOT NULL DEFAULT 'unknown' 
               COMMENT 'fetch|llm_invoke|llm_parse|validate|postprocess|produce|consume|sql_merge'""",
            
            """ALTER TABLE enrichment_errors 
               ADD COLUMN error_code VARCHAR(64) NOT NULL DEFAULT 'UNKNOWN' 
               COMMENT 'INVALID_GENRE|TIMEOUT|JSON_PARSE|etc'""",
            
            """ALTER TABLE enrichment_errors 
               ADD COLUMN error_field VARCHAR(128) NULL 
               COMMENT 'Specific field that failed validation'""",
            
            # Rename 'error' to 'error_msg' for consistency
            """ALTER TABLE enrichment_errors 
               CHANGE COLUMN error error_msg TEXT NOT NULL""",
            
            # Context
            """ALTER TABLE enrichment_errors 
               ADD COLUMN tags_version VARCHAR(32) NOT NULL DEFAULT 'v1'""",
            
            """ALTER TABLE enrichment_errors 
               ADD COLUMN title VARCHAR(256) NULL""",
            
            """ALTER TABLE enrichment_errors 
               ADD COLUMN author VARCHAR(256) NULL""",
            
            """ALTER TABLE enrichment_errors 
               ADD COLUMN attempted JSON NULL 
               COMMENT 'Partial enrichment results if any'""",
            
            # Indexes
            """ALTER TABLE enrichment_errors 
               ADD INDEX idx_error_code (error_code)""",
            
            """ALTER TABLE enrichment_errors 
               ADD INDEX idx_stage (stage)""",
            
            """ALTER TABLE enrichment_errors 
               ADD INDEX idx_last_seen (last_seen_at)""",
        ]
        
        for i, sql in enumerate(migrations, 1):
            try:
                print(f"  [{i}/{len(migrations)}] Executing migration...")
                cur.execute(sql)
                conn.commit()
            except pymysql.err.OperationalError as e:
                if "Duplicate column" in str(e) or "Duplicate key" in str(e):
                    print(f"    Skipped (already exists)")
                else:
                    raise
        
        # Populate default values for existing rows from old 'error' column
        print("Populating default values for existing error records...")
        cur.execute("""
            UPDATE enrichment_errors 
            SET 
                stage = CASE 
                    WHEN error_msg LIKE '%timeout%' THEN 'llm_invoke'
                    WHEN error_msg LIKE '%parse%' OR error_msg LIKE '%JSON%' THEN 'llm_parse'
                    WHEN error_msg LIKE '%invalid%' OR error_msg LIKE '%validation%' THEN 'validate'
                    ELSE 'unknown'
                END,
                error_code = CASE
                    WHEN error_msg LIKE '%timeout%' THEN 'TIMEOUT'
                    WHEN error_msg LIKE '%parse%' OR error_msg LIKE '%JSON%' THEN 'JSON_PARSE'
                    WHEN error_msg LIKE '%genre%' THEN 'INVALID_GENRE'
                    WHEN error_msg LIKE '%tone%' THEN 'INVALID_TONE_ID'
                    WHEN error_msg LIKE '%vibe%' THEN 'VIBE_TOO_LONG'
                    ELSE 'UNKNOWN'
                END
            WHERE stage = 'unknown' OR error_code = 'UNKNOWN'
        """)
        conn.commit()
        
        print("Migration completed successfully!")
        print("\nNew table structure:")
        cur.execute("DESCRIBE enrichment_errors")
        for row in cur.fetchall():
            print(f"  {row[0]:20} {row[1]:20} {row[2]:5} {row[3]:5}")
        
    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    print("="*80)
    print("ENRICHMENT_ERRORS TABLE MIGRATION - PHASE 1")
    print("="*80 + "\n")
    
    migrate_enrichment_errors()
    
    print("\nNext steps:")
    print("  1. Review the new table structure above")
    print("  2. Test with a small enrichment run")
    print("  3. Deploy Phase 1 Kafka pipeline")
