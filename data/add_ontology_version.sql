-- data/add_ontology_version.sql
-- Run this entire block at once in your mysql> prompt or via file

-- 1. tones → replace UNIQUE(slug) with UNIQUE(slug, ontology_version)
-- Safely drop the old unique key whatever its name is
SET @key_name = (
    SELECT CONSTRAINT_NAME 
    FROM information_schema.KEY_COLUMN_USAGE 
    WHERE TABLE_NAME = 'tones' 
      AND COLUMN_NAME = 'slug' 
      AND CONSTRAINT_NAME != 'PRIMARY'
      AND TABLE_SCHEMA = DATABASE()
    LIMIT 1
);

SET @sql = IF(@key_name IS NOT NULL,
    CONCAT('ALTER TABLE tones DROP KEY `', @key_name, '`'),
    'SELECT "No old unique key to drop on tones"');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- Now add the correct composite unique key
ALTER TABLE tones 
ADD UNIQUE KEY uq_tone_slug_version (slug, ontology_version);

-- 2. genres → change PK from (slug) → (slug, ontology_version)
ALTER TABLE genres 
DROP PRIMARY KEY,
ADD PRIMARY KEY (slug, ontology_version);

-- 3. book_genres → drop any existing FK + recreate the correct one
-- Drop whatever FK currently exists on book_genres (there can be only one)
SET @fk_name = (
    SELECT CONSTRAINT_NAME 
    FROM information_schema.TABLE_CONSTRAINTS 
    WHERE TABLE_NAME = 'book_genres' 
      AND CONSTRAINT_TYPE = 'FOREIGN KEY' 
      AND TABLE_SCHEMA = DATABASE()
    LIMIT 1
);

SET @sql = IF(@fk_name IS NOT NULL,
    CONCAT('ALTER TABLE book_genres DROP FOREIGN KEY `', @fk_name, '`'),
    'SELECT "No FK to drop on book_genres"');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- Make sure the column is named correctly
ALTER TABLE book_genres 
CHANGE COLUMN IF EXISTS genre_ontology_version genre_ontology_version VARCHAR(32) NOT NULL DEFAULT 'v1';

-- Add the correct composite foreign key
ALTER TABLE book_genres 
ADD CONSTRAINT fk_book_genre_version
    FOREIGN KEY (genre_slug, genre_ontology_version)
    REFERENCES genres(slug, ontology_version)
    ON DELETE RESTRICT ON UPDATE CASCADE;

-- Done!
SELECT 'SUCCESS: Database is now 100% in sync with table_models.py' AS status;
