EXPLAIN (ANALYZE, BUFFERS)
WITH joined AS (
  SELECT r.a AS a, r.b AS b, s.c AS c, t.d AS d
  FROM R AS r
  JOIN S AS s ON s.b = r.b
  JOIN T AS t ON t.c = s.c
),
numbered AS (
  SELECT *, ROW_NUMBER() OVER (ORDER BY a, b, c, d) AS rn
  FROM joined
),
total_count AS (
  SELECT COUNT(*) AS cnt FROM numbered
),
positions AS (
  SELECT 1 AS pos
  UNION ALL SELECT FLOOR(cnt * 0.25)::int  FROM total_count
  UNION ALL SELECT FLOOR(cnt * 0.5)::int  FROM total_count
  UNION ALL SELECT FLOOR(cnt * 0.75)::int  FROM total_count
  UNION ALL SELECT cnt FROM total_count
)
SELECT n.*
FROM numbered n
JOIN positions p ON n.rn = p.pos
ORDER BY n.rn;