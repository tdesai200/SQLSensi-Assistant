# Join Performance Best Practices

Ensure both sides of a join have indexes on the join keys.
Avoid joining large tables without filters.
Consider denormalizing or using materialized views when queries involve multiple joins and aggregations.
Use INNER JOIN when possible for better performance.
