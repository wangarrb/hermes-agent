#!/home/wyr/miniconda/envs/hindsight/bin/python
"""Reset Hindsight database and run migrations."""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import datetime
import os
import sys

DB_NAME = "hindsight"
DB_USER = "hindsight"
DB_HOST = "127.0.0.1"
DB_PORT = "5432"
BACKUP_DIR = "/home/wyr/.hermes/hindsight/backups/reset-pre-earliest-week-20260509-155702"
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs(BACKUP_DIR, exist_ok=True)

# 1. Check current DB state
conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, database=DB_NAME)
cur = conn.cursor()
cur.execute("SELECT count(*) FROM documents")
doc_count = cur.fetchone()[0]
cur.execute("SELECT count(*) FROM memory_units")
unit_count = cur.fetchone()[0]
cur.close()
conn.close()

print(f"[{TS}] Pre-reset: documents={doc_count}, memory_units={unit_count}")

# 2. Logical backup
backup_file = f"{BACKUP_DIR}/hindsight-pre-reset-{TS}.sql"
conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, database=DB_NAME)
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cur = conn.cursor()

cur.execute("""
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
ORDER BY table_name
""")
tables = [r[0] for r in cur.fetchall()]

with open(backup_file, 'w') as f:
    f.write(f"-- Hindsight backup {TS}\n")
    f.write(f"-- Pre-reset: documents={doc_count}, memory_units={unit_count}\n\n")
    for t in tables:
        f.write(f"-- Table: {t}\n")
        try:
            cur.execute(f"SELECT count(*) FROM {t}")
            count = cur.fetchone()[0]
            f.write(f"-- Row count: {count}\n\n")
        except Exception as e:
            f.write(f"-- Error: {e}\n\n")

cur.close()
conn.close()
print(f"Backup saved: {backup_file}")

# 3. Drop + recreate
conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, database="postgres")
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cur = conn.cursor()

cur.execute(f"""
SELECT pg_terminate_backend(pid) FROM pg_stat_activity
WHERE datname = '{DB_NAME}' AND pid <> pg_backend_pid()
""")
print("Terminated existing connections")

cur.execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
print(f"Dropped database {DB_NAME}")

cur.execute(f"CREATE DATABASE {DB_NAME} OWNER {DB_USER}")
print(f"Created database {DB_NAME}")

cur.close()
conn.close()

# 4. Run migrations
sys.path.insert(0, "/home/wyr/hindsight_shadow")
from pathlib import Path
from alembic import command
from alembic.config import Config

database_url = f"postgresql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
script_location = "/home/wyr/hindsight_shadow/hindsight_api/alembic"

alembic_cfg = Config()
alembic_cfg.set_main_option("script_location", script_location)
alembic_cfg.set_main_option("sqlalchemy.url", database_url)
alembic_cfg.set_main_option("prepend_sys_path", ".")
alembic_cfg.set_main_option("path_separator", "os")

command.upgrade(alembic_cfg, "heads")
print("Migrations completed")

# 5. Verify post-reset
conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, database=DB_NAME)
cur = conn.cursor()
cur.execute("SELECT count(*) FROM documents")
post_doc = cur.fetchone()[0]
cur.execute("SELECT count(*) FROM memory_units")
post_unit = cur.fetchone()[0]
cur.close()
conn.close()

print(f"Post-reset: documents={post_doc}, memory_units={post_unit}")
print("=== RESET COMPLETE ===")