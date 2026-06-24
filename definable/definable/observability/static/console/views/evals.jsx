// Right config — Evals tab body. Synthesised from observability metrics.

function EvalsView({ agent, runs, metrics }) {
  const UI = window.UI;
  if (!agent) {
    return (
      <div className="config-body scroll-y">
        <UI.EmptyState title="No agent selected" />
      </div>
    );
  }
  const m = metrics || {};
  const successRate = m.runs > 0 ? ((m.runs - m.errors) / m.runs) * 100 : 0;
  const errorRate = m.runs > 0 ? (m.errors / m.runs) * 100 : 0;
  // Latency scores: lower-is-better, map to 0-100.
  const latP50Score = m.p50_ms ? Math.max(0, 100 - Math.min(100, m.p50_ms / 100)) : 100;
  const latP95Score = m.p95_ms ? Math.max(0, 100 - Math.min(100, m.p95_ms / 300)) : 100;
  // Token efficiency: average tokens per run; lower-is-better up to 10k cap.
  const tokPerRun = m.runs > 0 ? (m.input_tokens + m.output_tokens) / m.runs : 0;
  const tokScore = tokPerRun ? Math.max(0, 100 - Math.min(100, tokPerRun / 200)) : 100;

  const overall = (successRate * 0.5 + latP50Score * 0.25 + tokScore * 0.25);

  const evals = [
    { name: 'Success rate', pass: successRate.toFixed(1), max: 100, trend: errorRate > 5 ? '-' : '+', desc: 'Fraction of runs that completed without error.' },
    { name: 'P50 latency', pass: latP50Score.toFixed(1), max: 100, trend: '0', desc: `Lower is better. Currently ${UI.fmtMs(m.p50_ms)}.` },
    { name: 'P95 latency', pass: latP95Score.toFixed(1), max: 100, trend: '0', desc: `Tail latency. Currently ${UI.fmtMs(m.p95_ms)}.` },
    { name: 'Error rate', pass: (100 - errorRate).toFixed(1), max: 100, trend: errorRate > 5 ? '-' : '+', desc: `${m.errors || 0} of ${m.runs || 0} runs errored.` },
    { name: 'Token efficiency', pass: tokScore.toFixed(1), max: 100, trend: '0', desc: `Average ${UI.fmtTokens(tokPerRun)} tokens per run.` },
  ];

  const recentCI = (runs || []).slice(0, 5).map((r) => ({
    id: (r.id || '').slice(0, 8),
    when: UI.fmtRelative(r.started_at),
    trigger: r.input || 'playground run',
    result: r.status === 'completed' ? 'passed' : r.status === 'errored' ? 'errored' : r.status,
    state: r.status === 'completed' ? 'pass' : r.status === 'errored' ? 'fail' : 'warn',
  }));

  return (
    <div className="config-body scroll-y">
      <div className="agent-hero">
        <div className="t-eyebrow eyebrow">Evals</div>
        <h1 className="name">How well it works</h1>
        <div className="handle">{evals.length} synthesised evals · derived from /api/metrics</div>
        <div className="agent-meta-row">
          <div className="stat">
            <span className="label">Overall</span>
            <span className="value">{overall.toFixed(1)}</span>
          </div>
          <div className="stat">
            <span className="label">Runs · 24h</span>
            <span className="value">{m.runs || 0}</span>
          </div>
          <div className="stat">
            <span className="label">Errors</span>
            <span className="value">{m.errors || 0}</span>
          </div>
        </div>
      </div>

      <UI.AccordionSection title="Eval suite" status={errorRate < 5 ? 'Passing' : 'Degraded'} statusOn={errorRate < 5}>
        {evals.map((e, i) => (
          <div key={i} className="eval-card">
            <div className="eval-head">
              <div className="eval-name">{e.name}</div>
              <div className="eval-score">
                <span className="big">{e.pass}</span>
                <span className="dim"> / {e.max}</span>
                <span className={`trend ${e.trend === '+' ? 'pos' : e.trend === '-' ? 'neg' : ''}`}>{e.trend}</span>
              </div>
            </div>
            <div className="eval-desc">{e.desc}</div>
            <div className="eval-track">
              <div className="eval-fill" style={{ width: `${Math.min(100, parseFloat(e.pass))}%` }} />
            </div>
          </div>
        ))}
      </UI.AccordionSection>

      <UI.AccordionSection title="Recent runs">
        {recentCI.length === 0
          ? <div style={{ color: 'var(--stone)', fontSize: 12, padding: '8px 0' }}>No runs yet.</div>
          : recentCI.map((ci) => (
              <div key={ci.id} className="ci-row">
                <span className={`ci-dot ${ci.state}`} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="ci-trigger">{ci.trigger.length > 40 ? ci.trigger.slice(0, 40) + '…' : ci.trigger}</div>
                  <div className="ci-when">#{ci.id} · {ci.when}</div>
                </div>
                <div className="ci-result">{ci.result}</div>
              </div>
            ))}
      </UI.AccordionSection>
    </div>
  );
}

window.EvalsView = EvalsView;
