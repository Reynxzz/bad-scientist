-- CREATE OR REPLACE CORTEX SEARCH SERVICE sklearn_docs_search_svc
-- ON doc_text
-- WAREHOUSE = COMPUTE_WH
-- TARGET_LAG = '1 hour'
-- AS 
--     SELECT 
--         doc_text,
--     FROM sklearn_docs_chunks

CREATE OR REPLACE CORTEX SEARCH SERVICE sklearn_docs_search_svc
ON doc_text
WAREHOUSE = COMPUTE_WH
TARGET_LAG = '1 hour'
AS 
    SELECT 
        col_summarize,
    FROM sklearn_docs_chunks

-- CREATE OR REPLACE CORTEX SEARCH SERVICE streamlit_docs_search_svc
-- ON doc_text
-- WAREHOUSE = COMPUTE_WH
-- TARGET_LAG = '1 hour'
-- AS 
--     SELECT 
--         doc_text,
--     FROM streamlit_docs_chunks

CREATE OR REPLACE CORTEX SEARCH SERVICE streamlit_docs_search_svc
ON doc_text
WAREHOUSE = COMPUTE_WH
TARGET_LAG = '1 hour'
AS 
    SELECT 
        col_summarize,
    FROM streamlit_docs_chunks