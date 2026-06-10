# Hindsight Container Upgrade Pattern

## Overview

Step-by-step pattern for upgrading the Hindsight Docker container to a new version while preserving all configuration and data.

Key principle: Hindsight uses `--network host` and connects to a host PostgreSQL. Data is in PostgreSQL (persistent), not in the container. Upgrading the container image only changes the application layer.

## Upgrade Steps

### 1. Pull the new image

```bash
docker pull ghcr.io/vectorize-io/hindsight:<NEW_VERSION>
```

### 2. Verify the new image exists locally

```bash
docker images | grep hindsight
```

### 3. Dump current container environment

```bash
docker inspect hindsight --format '{{range .Config.Env}}{{.}}{{println}}{{end}}' > /tmp/hindsight_env_file.txt
```

This captures all env vars (including API keys) in `KEY=VALUE` format for `--env-file`.

### 4. Record volume mounts and network config

```bash
docker inspect hindsight --format '{{json .HostConfig.Binds}}' | python3 -c "import sys,json; [print(v) for v in json.loads(sys.stdin.read())]"
docker inspect hindsight --format '{{.HostConfig.NetworkMode}}'
docker inspect hindsight --format '{{json .HostConfig.RestartPolicy}}'
```

### 5. Stop and remove the old container

```bash
docker stop hindsight && docker rm hindsight
```

**Pitfall**: If `RestartPolicy` is `unless-stopped`, the container may auto-restart after `docker rm` before you create the new one. Do `docker stop && docker rm` in one line, then immediately create the new container.

### 6. Recreate with the new image

```bash
docker run -d \
  --name hindsight \
  --network host \
  --restart unless-stopped \
  -v /home/wyr/.cache/torch:/home/wyr/.cache/torch \
  -v /home/wyr/.cache/huggingface:/home/hindsight/.cache/huggingface \
  -v /home/wyr/.cache/huggingface:/home/wyr/.cache/huggingface \
  -v /home/wyr/.cache/torch:/home/hindsight/.cache/torch \
  --env-file /tmp/hindsight_env_file.txt \
  ghcr.io/vectorize-io/hindsight:<NEW_VERSION>
```

**Pitfall**: Do NOT reconstruct `-e` flags from the env dump — shell escaping issues with API keys and special characters will cause `docker run` to fail with confusing errors like `pull access denied for extract`. Use `--env-file` instead.

### 7. Wait for API health

```bash
# The container takes 10-60s to start (embedding model loading)
for i in 1 2 3 4 5 6 7 8; do
  sleep 15
  result=$(curl -s http://127.0.0.1:8888/health 2>/dev/null)
  if [ -n "$result" ]; then
    echo "Ready: $result"
    break
  fi
  echo "Waiting... ($i)"
done
```

### 8. Verify version

```bash
docker inspect hindsight --format '{{index .Config.Labels "org.opencontainers.image.version"}}'
```

### 9. Reapply patches if needed

If local patches (e.g., parallel consolidator, temporal FK guard) need to be reapplied after container recreate:

```bash
python3 $HERMES_HOME/scripts/patch_hindsight_consolidator_parallel.py
python3 $HERMES_HOME/scripts/patch_hindsight_retain_temporal_fk_guard.py
```

### 10. Clean up old containers

```bash
docker rm hindsight-old-<OLD_VERSION> 2>/dev/null
```

## Database Considerations

- **Data is in PostgreSQL** — container recreate does NOT affect data. The new container connects to the same database.
- **Schema migrations**: Check if the new version requires DB migrations:
  - `HINDSIGHT_API_RUN_MIGRATIONS_ON_STARTUP=false` is the default in this deployment
  - Check the new version's release notes for schema changes
  - If migrations are needed, temporarily set `RUN_MIGRATIONS_ON_STARTUP=true` or run Alembic manually
  - Current alembic version can be checked: `SELECT version_num FROM alembic_version`
- **The user's question "数据库同步到最新的了吗?"** usually means "is the data from offline pipeline runs visible in the new version?" — the answer is always yes if using the same PostgreSQL, regardless of container version.

## Common Pitfalls

1. **`docker rm` auto-restart race**: With `--restart unless-stopped`, the old image can auto-restart after removal. Stop+rm in one step, then immediately create the new container.
2. **Shell escaping with `-e` flags**: Reconstructing `-e KEY=VALUE` from `docker inspect` output breaks on API keys with special characters. Always use `--env-file`.
3. **Forgetting to reapply patches**: Container recreate wipes the writable layer. Local patches (parallel consolidator, temporal FK guard, JSON parser) must be reapplied.
4. **Trusting `:latest` tag**: `docker pull ghcr.io/vectorize-io/hindsight:latest` may not update if the tag already exists locally with an older digest. Always use explicit version tags and verify with `docker inspect` labels.
5. **Assuming data loss**: Users may worry that upgrading the container loses imported data. Reassure: data is in PostgreSQL, not in the container. The container is just the application layer.

## Verified Upgrades

| Date | From | To | Notes |
|------|------|----|-------|
| 2026-06-02 | 0.6.1 | 0.7.1 | Smooth; pyproject name changed to `hindsight-api-slim`; added `markitdown` and `cryptography` deps; no schema migration needed |
