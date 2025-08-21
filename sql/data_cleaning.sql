--- Création d'une table emails
CREATE TABLE emails (
    email_id INTEGER PRIMARY KEY AUTOINCREMENT,
    email_text TEXT,
    email_type TEXT
);

-- Création table temporaire pour le fichier avec 2 colonnes
CREATE TABLE temp_emails (
    email_text TEXT,
    email_type TEXT
);

-- Transfert des données entre la table temporaire et la table final
INSERT INTO emails (email_text, email_type)
SELECT email_text, email_type FROM temp_emails;

-- Suprresion de la table inutile
DROP TABLE temp_emails;

-- Néttoyages des données
UPDATE emails
SET email_text = 'Unknown'
WHERE email_text IS NULL OR email_text = '';
DELETE FROM emails
WHERE email_type IS NULL OR email_type = '';

-- Normalisation de la colonne spam
ALTER TABLE emails ADD COLUMN is_phishing INTEGER;
UPDATE emails
SET is_phishing = CASE
    WHEN LOWER(email_type) LIKE '%phishing%' THEN 1
    WHEN LOWER(email_type) LIKE '%safe%' THEN 0
    ELSE NULL
END;
ALTER TABLE emails DROP COLUMN email_type;

-- Ajout d'infos pour analyse
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
