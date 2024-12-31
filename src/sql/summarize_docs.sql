-- SELECT SNOWFLAKE.CORTEX.COMPLETE(
--     'mistral-large',
--         CONCAT('Summarize this biz docs in less than 100 words. 
--         Put the product name and summary in JSON format: <doc_text>',  
--         doc_text, '</doc_text>')
-- ) FROM req_docs_chunks LIMIT 10;

SELECT SNOWFLAKE.CORTEX.SUMMARIZE(doc_text) FROM streamlit_docs_chunks LIMIT 10;

ALTER TABLE streamlit_docs_chunks 
ADD COLUMN "col_summarize" TEXT;

-- Show the table structure to find the actual column names
DESCRIBE TABLE streamlit_docs_chunks;

-- -- Then update using a row number approach without relying on ID
-- UPDATE streamlit_docs_chunks target
-- SET "col_summarize" = SNOWFLAKE.CORTEX.SUMMARIZE(doc_text)
-- WHERE EXISTS (
--     SELECT 1 
--     FROM (
--         SELECT doc_text, ROW_NUMBER() OVER (ORDER BY doc_text) as rnum 
--         FROM streamlit_docs_chunks 
--         WHERE "col_summarize" IS NULL
--     ) source 
--     WHERE source.doc_text = target.doc_text 
--     AND rnum <= 10
-- );

UPDATE streamlit_docs_chunks 
SET "col_summarize" = SNOWFLAKE.CORTEX.SUMMARIZE(doc_text)
WHERE "col_summarize" IS NULL;

SELECT doc_text, "col_summarize"
FROM streamlit_docs_chunks 
WHERE "col_summarize" IS NOT NULL;


ALTER TABLE sklearn_docs_chunks 
ADD COLUMN "col_summarize" TEXT;

-- Show the table structure to find the actual column names
DESCRIBE TABLE sklearn_docs_chunks;

-- -- Then update using a row number approach without relying on ID
-- UPDATE sklearn_docs_chunks target
-- SET "col_summarize" = SNOWFLAKE.CORTEX.SUMMARIZE(doc_text)
-- WHERE EXISTS (
--     SELECT 1 
--     FROM (
--         SELECT doc_text, ROW_NUMBER() OVER (ORDER BY doc_text) as rnum 
--         FROM sklearn_docs_chunks 
--         WHERE "col_summarize" IS NULL
--     ) source 
--     WHERE source.doc_text = target.doc_text 
--     AND rnum <= 10
-- );

UPDATE sklearn_docs_chunks 
SET "col_summarize" = SNOWFLAKE.CORTEX.SUMMARIZE(doc_text)
WHERE "col_summarize" IS NULL;

SELECT doc_text, "col_summarize"
FROM sklearn_docs_chunks 
WHERE "col_summarize" IS NOT NULL;