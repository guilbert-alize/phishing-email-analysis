--- Table emails
CREATE TABLE emails (
    email_id INTEGER PRIMARY KEY AUTOINCREMENT,
    email_text TEXT,
    email_type TEXT
);

-- temporary table to import the dataset that only has 2 columns
CREATE TABLE temp_emails (
    email_text TEXT,
    email_type TEXT
);

-- Import CSV (run in SQLite prompt)
-- .mode csv
-- .import --skip 1 data/Phishing_Email.csv temp_emails

-- Transfer to main table
INSERT INTO emails (email_text, email_type)
SELECT email_text, email_type FROM temp_emails;

-- Drop temp table
DROP TABLE temp_emails;

-- Manage missing and empty values
UPDATE emails
SET email_text = 'Unknown'
WHERE email_text IS NULL OR email_text = '';
DELETE FROM emails
WHERE email_type IS NULL OR email_type = '';

-- Standardize email_type to is_phishing (adjust based on actual email_type values)
ALTER TABLE emails ADD COLUMN is_phishing INTEGER;
UPDATE emails
SET is_phishing = CASE
    WHEN LOWER(email_type) LIKE '%phishing%' THEN 1
    WHEN LOWER(email_type) LIKE '%safe%' THEN 0
    ELSE NULL
END;
ALTER TABLE emails DROP COLUMN email_type;

-- Adding useful infos
ALTER TABLE emails ADD COLUMN text_length INTEGER;
UPDATE emails
SET text_length = LENGTH(email_text);

ALTER TABLE emails ADD COLUMN has_url INTEGER;
UPDATE emails
SET has_url = CASE WHEN email_text LIKE '%http%' THEN 1 ELSE 0 END;

ALTER TABLE emails ADD COLUMN keyword_count INTEGER;
UPDATE emails
SET keyword_count = (
    (email_text LIKE '%login%') +
    (email_text LIKE '%urgent%') +
    (email_text LIKE '%password%') +
    (email_text LIKE '%account%')
);

-- Export cleaned data
-- .headers on
-- .mode csv
-- .output data/cleaned_emails.csv
-- SELECT email_id, email_text, is_phishing, text_length, has_url, keyword_count FROM emails;
-- .output stdout