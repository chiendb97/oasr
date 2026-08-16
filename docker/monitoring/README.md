# Monitoring an OASR server

`oasr-server` exposes Prometheus metrics at `GET /metrics` with no extra flags —
the endpoint is always on. Getting from there to graphs is three things:

1. **Prometheus** scraping the server.
2. A **Grafana datasource** pointing at that Prometheus.
3. The **dashboard** ([`grafana/oasr-overview.json`](grafana/oasr-overview.json))
   imported.

What each metric means, what to alert on, and why a panel can be legitimately
empty: [`docs/serving.md`](../../docs/serving.md) § Metrics.

---

## Path A — you have neither Prometheus nor Grafana

The compose file starts both and provisions the datasource and the dashboard.

```bash
# 1. Start OASR (any mode). The default bind is 0.0.0.0:8080, which is what
#    lets a container reach it.
oasr-server --ckpt-dir /path/to/ckpt --service-mode offline

# 2. Start the monitoring stack.
cd docker/monitoring
docker compose -f docker-compose.monitoring.yml up -d

# 3. Open Grafana. Dashboard: "OASR — Serving Overview", in the OASR folder.
xdg-open http://localhost:3000
```

### Ports

| Variable | Default | Publishes |
|---|---|---|
| `GRAFANA_PORT` / `GRAFANA_BIND` | `3000` / `0.0.0.0` | Grafana's UI |
| `PROMETHEUS_PORT` / `PROMETHEUS_BIND` | `9090` / `127.0.0.1` | Prometheus' UI and query API |

```bash
GRAFANA_PORT=9000 docker compose -f docker-compose.monitoring.yml up -d
```

Only the **host** side moves. Grafana keeps listening on 3000 inside the
container, and Prometheus on 9090 — mapping a host port straight through to a
container port nothing listens on (`9000:9000` rather than `9000:3000`)
publishes a port that refuses every connection. To move Grafana's listener too, set `GF_SERVER_HTTP_PORT` in
its `environment:` block *and* the container side of the mapping; nothing here
needs it.

Changing these does not affect the dashboard: Grafana reaches Prometheus over the
compose network as `http://prometheus:9090`, which is independent of what either
service publishes to the host.

### Pointing it at a different address

`OASR_TARGETS` sets the scrape target list; nothing has to be edited.

```bash
# Another host, or another port.
OASR_TARGETS=host:port docker compose -f docker-compose.monitoring.yml up -d

# Scale-out: one oasr-server process per GPU, one entry each.
OASR_TARGETS='host-0:port-0,host-1:port-1' docker compose -f docker-compose.monitoring.yml up -d
```

The default is `host.docker.internal:8080`, so a server on the local host needs
no variable. Re-running `up -d` re-renders the list; add
`--force-recreate prometheus` if the container was not replaced. To stop
repeating it, put `OASR_TARGETS=…` in `docker/monitoring/.env` — compose reads that
automatically, and `.env*` is gitignored.

The variable works because the targets live in a file-SD list that the compose
file generates, not in `prometheus.yml`. It has to be done out there: Prometheus
has no `--target` flag and does not expand environment variables in its own
config.

> Inline config `content` needs **Compose v2.23.1+** (`docker compose version`);
> on an older one the stack fails to parse. Replace the `configs:` block with a
> bind mount of `./prometheus/targets` and write the same one-line file
> yourself: `echo '- targets: [host:port]' > prometheus/targets/oasr.yml`.

### Before exposing it anywhere

**Grafana ships bound to `0.0.0.0` with anonymous admin**, so out of the box
anyone who can route to this host gets Grafana's admin UI. That combination is
fine on a workstation behind a firewall and nowhere else. Pick one before the
box is reachable: drop the three `GF_AUTH_ANONYMOUS_*`/`GF_AUTH_BASIC_*` lines
and set `GF_SECURITY_ADMIN_PASSWORD`, or keep it closed with
`GRAFANA_BIND=127.0.0.1` and reach it over an SSH tunnel
(`ssh -L 3000:localhost:3000 host`).

Prometheus is on the loopback by default and needs no such decision.
`/metrics` on `oasr-server` is itself unauthenticated — OASR has no auth on any
route yet — so on its default `0.0.0.0` bind, anyone who can reach the server can
read its metrics.

## Path B — you already run Prometheus and Grafana

**Add a scrape job.** One target per `oasr-server` process; scale-out is one
process per GPU, and the `engine` label keeps their series apart.

```yaml
scrape_configs:
  - job_name: oasr
    metrics_path: /metrics
    static_configs:
      - targets: ['oasr-host:8080']
```

Do not pin an `instance` label: Prometheus derives it from the address, and
overriding it collapses two replicas onto one series.

**Set the datasource's scrape interval** to whatever Prometheus uses (15 s
here). Every rate() in the dashboard uses `$__rate_interval`, which is derived
from *this* field and not from Prometheus — leave it unset and rate windows come
out shorter than the scrape period, which shows up as spiky or empty panels.
This is the single most common reason an imported dashboard looks broken.

**Import the dashboard.** Grafana → Dashboards → New → Import → upload
`grafana/oasr-overview.json`. Then set the **Datasource** variable at the top of
the dashboard to your Prometheus once — it defaults to the UID the compose file
provisions (`oasr-prometheus`), which will not exist in your Grafana.

Panels are editable in the browser, but provisioning overwrites them from disk on
restart: export back to `grafana/oasr-overview.json` (Export → **for sharing
externally: off**) or the changes are lost.

---

## Checking it works

```bash
# The server is exporting. The count grows with traffic rather than being fixed —
# the exporter renders nothing for a metric that has no samples yet.
curl -s localhost:8080/metrics | grep -c '^# TYPE'

# Prometheus resolved the target list you meant, and each one is up. `instance`
# is the address, so this is also how you confirm OASR_TARGETS took effect.
curl -s localhost:9090/api/v1/targets \
  | jq '.data.activeTargets[] | {job:.labels.job, instance:.labels.instance, health, lastError}'

# Prometheus has the data.
curl -s --get localhost:9090/api/v1/query \
  --data-urlencode 'query=sum(rate(oasr_audio_seconds_total[5m]))' | jq .data.result
```

A panel is empty until traffic reaches the surface it measures, and some stay
empty by design for a given mode or decode family — time to first partial on an
offline-only server, the paged KV pool on a `StatefulStreamingBackend`, decode
slots on a frame-synchronous family. `docs/serving.md` § Metrics says which.
