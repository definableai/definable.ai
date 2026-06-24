// Runs tab — KPI hero + 30-day activity strip + filter tabs + table.

function RunsTab({ runs, metrics, onSelectRun, selectedRunId, onSwitchTab }) {
  const [filter, setFilter] = React.useState('all');
  const I = window.Icons;

  const counts = React.useMemo(() => {
    const c = { all: runs.length, completed: 0, errored: 0, running: 0 };
    runs.forEach((r) => { c[r.status] = (c[r.status] || 0) + 1; });
    return c;
  }, [runs]);

  const filtered = React.useMemo(() => {
    if (filter === 'all') return runs;
    return runs.filter((r) => r.status === filter);
  }, [runs, filter]);

  const bars = React.useMemo(() => {
    // 30 buckets, oldest → newest, by day. Bucket index 29 = today.
    const now = Date.now() / 1000;
    const dayS = 86400;
    const buckets = new Array(30).fill(0);
    runs.forEach((r) => {
      const ageD = Math.floor((now - (r.started_at || 0)) / dayS);
      const idx = 29 - ageD;
      if (idx >= 0 && idx < 30) buckets[idx] += 1;
    });
    const max = Math.max(1, ...buckets);
    return buckets.map((v) => Math.round((v / max) * 100));
  }, [runs]);

  const m = metrics || { runs: 0, errors: 0, p50_ms: 0, p95_ms: 0, cost_usd: 0, input_tokens: 0, output_tokens: 0 };
  const successPct = m.runs > 0 ? (((m.runs - m.errors) / m.runs) * 100).toFixed(1) + '%' : '—';

  return (
    <div className="runs-view scroll-y">
      <div className="runs-inner">
        <div className="runs-hero">
          <div className="t-eyebrow" style={{ color: 'var(--stone)' }}>Last 24 hours</div>
          <h2 className="t-heading-md" style={{ margin: '8px 0 20px', color: 'var(--ink)' }}>Activity</h2>
          <div className="metric-row" style={{ marginBottom: 24 }}>
            <div className="metric">
              <div className="m-label">Runs</div>
              <div className="m-value">{m.runs}</div>
              <div className="m-foot">{m.errors} errored</div>
            </div>
            <div className="metric">
              <div className="m-label">Success rate</div>
              <div className="m-value">{successPct}</div>
              <div className="m-foot">{m.errors} failures</div>
            </div>
            <div className="metric">
              <div className="m-label">P50 / P95</div>
              <div className="m-value">{window.UI.fmtMs(m.p50_ms)}</div>
              <div className="m-foot">p95 · {window.UI.fmtMs(m.p95_ms)}</div>
            </div>
            <div className="metric">
              <div className="m-label">Spend</div>
              <div className="m-value">{window.UI.fmtCost(m.cost_usd)}</div>
              <div className="m-foot">{window.UI.fmtTokens(m.input_tokens + m.output_tokens)} tok</div>
            </div>
          </div>

          <div className="activity-strip">
            <div className="t-eyebrow" style={{ color: 'var(--stone)', marginBottom: 10 }}>Runs per day</div>
            <div className="bars">
              {bars.map((h, i) => (
                <div key={i} className="bar">
                  <div className="bar-fill" style={{ height: `${Math.max(2, h)}%` }} />
                </div>
              ))}
            </div>
            <div className="bar-axis">
              <span>-30d</span>
              <span>-15d</span>
              <span>today</span>
            </div>
          </div>
        </div>

        <div className="runs-filter">
          <div className="runs-tabs">
            {[
              { id: 'all', label: 'All', count: counts.all },
              { id: 'completed', label: 'Completed', count: counts.completed || 0 },
              { id: 'running', label: 'Running', count: counts.running || 0 },
              { id: 'errored', label: 'Errored', count: counts.errored || 0 },
            ].map((t) => (
              <button key={t.id} className={`runs-tab ${filter === t.id ? 'active' : ''}`} onClick={() => setFilter(t.id)}>
                {t.label} <span className="ct">{t.count}</span>
              </button>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 6, marginLeft: 'auto' }}>
            <button className="btn-ghost-sm" title="Export coming soon"><I.IconDownload size={13} /> Export</button>
          </div>
        </div>

        <div className="runs-table">
          <div className="rt-head">
            <div>Run</div>
            <div>Trigger</div>
            <div>Started</div>
            <div className="num">Duration</div>
            <div className="num">Tokens</div>
            <div className="num">Cost</div>
            <div className="num">Status</div>
          </div>
          {filtered.length === 0 ? (
            <div style={{ padding: 32, textAlign: 'center', color: 'var(--stone)', fontSize: 12 }}>
              No runs match this filter.
            </div>
          ) : filtered.map((r) => {
            const dur = (r.ended_at && r.started_at) ? (r.ended_at - r.started_at) * 1000 : null;
            const tok = (r.total_input_tokens || 0) + (r.total_output_tokens || 0);
            const isErr = r.status === 'errored';
            const summary = r.output || r.error || r.input || '—';
            return (
              <div
                key={r.id}
                className={`rt-row status-${r.status} ${r.id === selectedRunId ? 'active' : ''}`}
                onClick={() => { onSelectRun(r.id); if (onSwitchTab) onSwitchTab('details'); }}
              >
                <div>
                  <div className="rt-run-id">#{(r.id || '').slice(0, 8)}</div>
                  <div className="rt-run-sum">{summary}</div>
                </div>
                <div>
                  <div className="rt-trigger">
                    <span className="trigger-dot playground" />
                    playground
                  </div>
                </div>
                <div className="rt-cell">{window.UI.fmtRelative(r.started_at)}</div>
                <div className="rt-cell num mono">{dur != null ? window.UI.fmtMs(dur) : '—'}</div>
                <div className="rt-cell num mono">{tok > 0 ? window.UI.fmtTokens(tok) : '—'}</div>
                <div className="rt-cell num mono">{r.total_cost_usd > 0 ? window.UI.fmtCost(r.total_cost_usd) : '—'}</div>
                <div className="num">
                  <window.UI.Pill kind={r.status}>{r.status}</window.UI.Pill>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

window.RunsTab = RunsTab;
