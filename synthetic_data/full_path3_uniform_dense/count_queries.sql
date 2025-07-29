SET statement_timeout = '0';
\timing
SELECT COUNT(*) FROM (SELECT r.a AS a, r.b AS b, s.c AS c, t.d AS d FROM R AS r
JOIN S AS s ON s.b = r.b
JOIN T AS t ON t.c = s.c) AS subq;
