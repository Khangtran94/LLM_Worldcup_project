import duckdb

con = duckdb.connect("data/processed/worldcup_ingest.duckdb")

result = con.sql("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'worldcup_staging'
    ORDER BY table_name
""").fetchdf()

print(result)