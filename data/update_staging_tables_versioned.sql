-- data/update_staging_tables_versioned.sql
-- Update staging tables to support tags_version

-- tmp_book_vibes_load needs tags_version
ALTER TABLE tmp_book_vibes_load 
ADD COLUMN tags_version VARCHAR(32) NOT NULL DEFAULT 'v1' AFTER vibe_text;

ALTER TABLE tmp_book_vibes_load
DROP PRIMARY KEY,
ADD PRIMARY KEY (item_idx, tags_version);

-- tmp_book_tones_load needs tags_version
ALTER TABLE tmp_book_tones_load
ADD COLUMN tags_version VARCHAR(32) NOT NULL DEFAULT 'v1' AFTER tone_id;

ALTER TABLE tmp_book_tones_load
DROP INDEX uq_tmp_book_tone;

ALTER TABLE tmp_book_tones_load
ADD CONSTRAINT uq_tmp_book_tone_version 
UNIQUE KEY (item_idx, tone_id, tags_version);

-- tmp_book_genres_load needs tags_version
ALTER TABLE tmp_book_genres_load
ADD COLUMN tags_version VARCHAR(32) NOT NULL DEFAULT 'v1' AFTER genre_slug;

ALTER TABLE tmp_book_genres_load
DROP PRIMARY KEY,
ADD PRIMARY KEY (item_idx, tags_version);

-- tmp_book_llm_subjects_load needs tags_version
ALTER TABLE tmp_book_llm_subjects_load
ADD COLUMN tags_version VARCHAR(32) NOT NULL DEFAULT 'v1' AFTER llm_subject;

ALTER TABLE tmp_book_llm_subjects_load
DROP INDEX uq_tmp_book_llm_subject;

ALTER TABLE tmp_book_llm_subjects_load
ADD CONSTRAINT uq_tmp_book_llm_subject_version 
UNIQUE KEY (item_idx, llm_subject, tags_version);

-- Verify staging tables
SELECT 'Staging tables updated for versioned enrichment' as status;
