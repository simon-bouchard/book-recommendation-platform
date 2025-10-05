-- /add_tags_version_to_links.sql
-- Migration to support versioned enrichment datasets
-- Run this ONCE before deploying versioned Spark consumer

-- Add tags_version to book_tones
ALTER TABLE book_tones 
ADD COLUMN tags_version VARCHAR(32) NOT NULL DEFAULT 'v1' AFTER tone_id;

-- Drop old unique constraint and create new composite key
ALTER TABLE book_tones 
DROP INDEX uq_book_tone;

ALTER TABLE book_tones 
ADD CONSTRAINT uq_book_tone_version 
UNIQUE KEY (item_idx, tone_id, tags_version);

-- Add tags_version to book_genres
ALTER TABLE book_genres 
ADD COLUMN tags_version VARCHAR(32) NOT NULL DEFAULT 'v1' AFTER genre_slug;

-- Recreate primary key to include tags_version
ALTER TABLE book_genres 
DROP PRIMARY KEY,
ADD PRIMARY KEY (item_idx, tags_version);

-- Add tags_version to book_vibes
ALTER TABLE book_vibes 
ADD COLUMN tags_version VARCHAR(32) NOT NULL DEFAULT 'v1' AFTER vibe_id;

-- Recreate primary key to include tags_version
ALTER TABLE book_vibes 
DROP PRIMARY KEY,
ADD PRIMARY KEY (item_idx, tags_version);

-- Add tags_version to book_llm_subjects
ALTER TABLE book_llm_subjects 
ADD COLUMN tags_version VARCHAR(32) NOT NULL DEFAULT 'v1' AFTER llm_subject_idx;

-- Drop old unique constraint and create new composite key
ALTER TABLE book_llm_subjects 
DROP INDEX uq_book_llm_subject;

ALTER TABLE book_llm_subjects 
ADD CONSTRAINT uq_book_llm_subject_version 
UNIQUE KEY (item_idx, llm_subject_idx, tags_version);

-- Create indexes for querying by version
ALTER TABLE book_tones ADD INDEX idx_version (tags_version);
ALTER TABLE book_genres ADD INDEX idx_version (tags_version);
ALTER TABLE book_vibes ADD INDEX idx_version (tags_version);
ALTER TABLE book_llm_subjects ADD INDEX idx_version (tags_version);

-- Create view for current version (v1 initially)
CREATE OR REPLACE VIEW v_current_enrichment_version AS
SELECT 'v1' as current_version;

-- View: Books enriched in each version
CREATE OR REPLACE VIEW v_enrichment_coverage_by_version AS
SELECT 
    tags_version,
    COUNT(DISTINCT item_idx) as books_enriched,
    COUNT(DISTINCT CASE WHEN s.item_idx IS NOT NULL THEN s.item_idx END) as with_subjects,
    COUNT(DISTINCT CASE WHEN t.item_idx IS NOT NULL THEN t.item_idx END) as with_tones,
    COUNT(DISTINCT CASE WHEN g.item_idx IS NOT NULL THEN g.item_idx END) as with_genres,
    COUNT(DISTINCT CASE WHEN v.item_idx IS NOT NULL THEN v.item_idx END) as with_vibes
FROM (
    SELECT DISTINCT item_idx, tags_version FROM book_llm_subjects
    UNION
    SELECT DISTINCT item_idx, tags_version FROM book_tones
    UNION
    SELECT DISTINCT item_idx, tags_version FROM book_genres
    UNION
    SELECT DISTINCT item_idx, tags_version FROM book_vibes
) all_enriched
LEFT JOIN book_llm_subjects s USING (item_idx, tags_version)
LEFT JOIN book_tones t USING (item_idx, tags_version)
LEFT JOIN book_genres g USING (item_idx, tags_version)
LEFT JOIN book_vibes v USING (item_idx, tags_version)
GROUP BY tags_version
ORDER BY tags_version;

-- View: Books with multiple versions (for comparison)
CREATE OR REPLACE VIEW v_multi_version_books AS
SELECT 
    item_idx,
    GROUP_CONCAT(DISTINCT tags_version ORDER BY tags_version) as versions,
    COUNT(DISTINCT tags_version) as version_count
FROM (
    SELECT DISTINCT item_idx, tags_version FROM book_llm_subjects
    UNION
    SELECT DISTINCT item_idx, tags_version FROM book_tones
    UNION
    SELECT DISTINCT item_idx, tags_version FROM book_genres
    UNION
    SELECT DISTINCT item_idx, tags_version FROM book_vibes
) all_versions
GROUP BY item_idx
HAVING version_count > 1
ORDER BY version_count DESC, item_idx;

-- Migration complete
SELECT 
    'Migration completed successfully. All link tables now support tags_version.' as status,
    (SELECT COUNT(DISTINCT item_idx) FROM book_llm_subjects WHERE tags_version = 'v1') as v1_books,
    'Use tags_version in your queries to separate enrichment versions.' as note;
