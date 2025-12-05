# Indexing Best Practices

Add indexes to columns used in JOIN conditions, WHERE filters, and foreign keys.
Indexing date and timestamp columns improves range filtering performance.
Avoid excessive indexing on tables with heavy write operations.
Use EXPLAIN plans to verify index usage.
