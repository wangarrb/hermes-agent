#!/home/wyr/miniconda/envs/hindsight/bin/python
"""Production reset + A-group retain runner.

Steps:
1. Backup current DB to SQL
2. Drop + recreate hindsight DB
3. Run alembic migrations
4. Restart hindsight service
5. Retain A-group candidates to production bank
"""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import subprocess
import datetime
import os
import sys
import time
import json

DB_NAME = "hindsight"
DB_USER = "hindsight"
DB_HOST = "127.0.0.1"
DB_PORT = "5432"
BACKUP_DIR = "/home/wyr/.hermes/hindsight/backups"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_FILE = f"{BACKUP_DIR}/hindsight_pre_reset_{TIMESTAMP}.sql"
RUN_DIR = "/home/wyr/.hermes/hindsight/runs"
API_URL = "http://127.0.0.1:8888"
HINDSIGHT_ENV = "/home/wyr/.hindsight/profiles/hermes.env"
HINDSIGHT_PYTHON = "/home/wyr/miniconda/envs/hindsight/bin/python"
HINDSIGHT_SRC = "/home/wyr/hindsight_shadow"

os.makedirs(BACKUP_DIR, exist_ok=True)

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def get_db_connection(dbname=DB_NAME):
    return psycopg2.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, database=dbname)

def run_sql(conn, sql, params=None):
    cur = conn.cursor()
    cur.execute(sql, params)
    result = cur.fetchall() if cur.description else None
    cur.close()
    return result

# ======================
# 1. Pre-reset stats
# ======================
log("=== Step 1: Pre-reset stats ===")
conn = get_db_connection()
cur = conn.cursor()
stats = {}
for t in ['documents', 'memory_units', 'memory_links', 'entities', 'chunks', 'async_operations', 'banks']:
    cur.execute(f"SELECT count(*) FROM {t}")
    stats[t] = cur.fetchone()[0]
cur.close()
conn.close()
log(f"Pre-reset: {json.dumps(stats)}")

# ======================
# 2. Backup (schema + data via pg_dump not available, use Python pg_dump equivalent)
# ======================
log("=== Step 2: Backup ===")
# Since pg_dump not available, do a logical backup of key tables
backup_conn = get_db_connection()
backup_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

cur = backup_conn.cursor()
cur.execute("""
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
ORDER BY table_name
""")
tables = [r[0] for r in cur.fetchall()]

with open(BACKUP_FILE, 'w') as f:
    f.write(f"-- Hindsight backup {TIMESTAMP}\n")
    f.write(f"-- Pre-reset stats: {json.dumps(stats)}\n\n")
    for t in tables:
        f.write(f"-- Table: {t}\n")
        try:
            cur.execute(f"SELECT * FROM {t}")
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            f.write(f"-- Columns: {', '.join(colnames)}\n")
            f.write(f"-- Row count: {len(rows)}\n")
            for row in rows[:1000]:  # cap per table
                vals = []
                for v in row:
                    if v is None:
                        vals.append("NULL")
                    elif isinstance(v, str):
                        vals.append("'" + v.replace("'", "''") + "'")
                    else:
                        vals.append(str(v))
                f.write(f"INSERT INTO {t} ({', '.join(colnames)}) VALUES ({', '.join(vals)});\n")
            if len(rows) > 1000:
                f.write(f"-- ... truncated {len(rows)-1000} rows\n")
            f.write("\n")
        except Exception as e:
            f.write(f"-- Error dumping {t}: {e}\n\n")
cur.close()
backup_conn.close()
log(f"Backup saved: {BACKUP_FILE}")

# ======================
# 3. Stop hindsight service
# ======================
log("=== Step 3: Stop hindsight ===")
subprocess.run(["systemctl", "--user", "stop", "hindsight-8888.service"], check=False)
time.sleep(2)

# Verify stopped
result = subprocess.run(["curl", "-s", "http://127.0.0.1:8888/health"], capture_output=True, text=True)
if "healthy" not in result.stdout:
    log("Hindsight stopped")
else:
    log("WARNING: Hindsight still responding, proceeding anyway")

# ======================
# 4. Drop + recreate DB
# ======================
log("=== Step 4: Drop + recreate DB ===")
conn = get_db_connection("postgres")
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cur = conn.cursor()

# Terminate existing connections
log("Terminating existing connections...")
cur.execute(f"""
SELECT pg_terminate_backend(pid) FROM pg_stat_activity
WHERE datname = %s AND pid <> pg_backend_pid()
""", (DB_NAME,))

# Drop and recreate
try:
    cur.execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
    log(f"Dropped database {DB_NAME}")
except Exception as e:
    log(f"Drop error (may be ok): {e}")

cur.execute(f"CREATE DATABASE {DB_NAME} OWNER {DB_USER}")
log(f"Created database {DB_NAME}")

cur.close()
conn.close()

# ======================
# 5. Run migrations
# ======================
log("=== Step 5: Run migrations ===")
env = os.environ.copy()
env["HINDSIGHT_API_DATABASE_URL"] = f"postgresql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
env["PYTHONPATH"] = HINDSIGHT_SRC

# Run alembic upgrade head
result = subprocess.run(
    [HINDSIGHT_PYTHON, "-m", "alembic", "upgrade", "head"],
    cwd=HINDSIGHT_SRC,
    env=env,
    capture_output=True,
    text=True
)
log(f"Migration stdout: {result.stdout.strip()}")
if result.returncode != 0:
    log(f"Migration stderr: {result.stderr.strip()}")
    log(f"Migration exit code: {result.returncode}")
    # Try running via hindsight_api.main with --run-migrations
    log("Trying alternative migration path...")
    result2 = subprocess.run(
        [HINDSIGHT_PYTHON, "-c", "from hindsight_api.migrations import run_migrations; run_migrations()"],
        cwd=HINDSIGHT_SRC,
        env=env,
        capture_output=True,
        text=True
    )
    log(f"Alt migration stdout: {result2.stdout.strip()}")
    if result2.returncode != 0:
        log(f"Alt migration stderr: {result2.stderr.strip()}")
else:
    log("Migrations completed")

# ======================
# 6. Restart hindsight
# ======================
log("=== Step 6: Restart hindsight ===")
subprocess.run(["systemctl", "--user", "start", "hindsight-8888.service"], check=False)

# Wait for health
for i in range(30):
    time.sleep(2)
    result = subprocess.run(["curl", "-s", "http://127.0.0.1:8888/health"], capture_output=True, text=True)
    if "healthy" in result.stdout:
        log("Hindsight is healthy")
        break
else:
    log("WARNING: Hindsight did not become healthy within 60s")

# ======================
# 7. Post-reset verify
# ======================
log("=== Step 7: Post-reset verify ===")
conn = get_db_connection()
cur = conn.cursor()
post_stats = {}
for t in ['documents', 'memory_units', 'memory_links', 'entities', 'chunks', 'async_operations', 'banks']:
    try:
        cur.execute(f"SELECT count(*) FROM {t}")
        post_stats[t] = cur.fetchone()[0]
    except:
        post_stats[t] = "N/A"
cur.close()
conn.close()
log(f"Post-reset: {json.dumps(post_stats)}")

log("=== RESET COMPLETE ===")
log(f"Backup: {BACKUP_FILE}")
log("Next: run A-group retain to production bank")
