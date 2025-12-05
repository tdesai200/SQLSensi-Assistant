# SELECT * Anti-pattern

SELECT * loads all columns, causing unnecessary I/O.
It prevents index-only scans.
It can break applications when columns are added or removed.
Use explicit column names instead to improve maintainability and performance.
