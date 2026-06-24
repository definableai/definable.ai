// Details tab — for selected run: summary tiles + waterfall + timeline + cost breakdown.

function DetailsTab({ selectedRunId, runDetail, loading }) {
  if (!selectedRunId) {
    return (
      <div className="details-view scroll-y">
        <div className="details-inner">
          <window.UI.EmptyState title="No run selected" sub="Pick a run from Runs or from the sidebar Recents." />
        </div>
      </div>
    );
  }
  if (loading || !runDetail) {
    return (
      <div className="details-view scroll-y">
        <div className="details-inner">
          <window.UI.EmptyState title="Loading…" />
        </div>
      </div>
    );
  }
  const run = runDetail.run;
  const spans = runDetail.spans || [];
  const events = runDetail.events || [];

  const duration = (run.ended_at && run.started_at) ? (run.ended_at - run.started_at) * 1000 : null;
  const tok = (run.total_input_tokens || 0) + (run.total_output_tokens || 0);

  // Waterfall scaling.
  const t0 = spans.length ? Math.min(...spans.map((s) => s.start_ts)) : run.started_at;
  const tN = spans.length ? Math.max(...spans.map((s) => s.end_ts || s.start_ts)) : (run.ended_at || run.started_at);
  const totalMs = Math.max(1, (tN - t0) * 1000);

  // Cost breakdown by span kind.
  const breakdown = computeBreakdown(spans, run);

  return (
    <div className="details-view scroll-y">
      <div className="details-inner">
        <div className="t-eyebrow" style={{ color: 'var(--stone)' }}>Run · #{(run.id || '').slice(0, 8)}</div>
        <h2 className="t-heading-md" style={{ margin: '8px 0 14px', color: 'var(--ink)' }}>Run details</h2>
        <p style={{ color: 'var(--graphite)', fontSize: 14.5, lineHeight: 1.55, maxWidth: 640, margin: '0 0 32px' }}>
          Full breakdown — what triggered it, what the agent did, every tool call, every event.
        </p>

        <div className="metric-row" style={{ marginBottom: 32 }}>
          <div className="metric">
            <div className="m-label">Started</div>
            <div className="m-value-sm">{window.UI.fmtTime(run.started_at)}</div>
            <div className="m-foot">{window.UI.fmtTimestamp(run.started_at)}</div>
          </div>
          <div className="metric">
            <div className="m-label">Duration</div>
            <div className="m-value-sm">{duration != null ? window.UI.fmtMs(duration) : '—'}</div>
            <div className="m-foot">{run.ended_at ? `ended ${window.UI.fmtTime(run.ended_at)}` : 'still running'}</div>
          </div>
          <div className="metric">
            <div className="m-label">Cost</div>
            <div className="m-value-sm">{window.UI.fmtCost(run.total_cost_usd)}</div>
            <div className="m-foot">{window.UI.fmtTokens(tok)} tok</div>
          </div>
          <div className="metric">
            <div className="m-label">Status</div>
            <div className="m-value-sm" style={{ textTransform: 'capitalize' }}>{run.status}</div>
            <div className="m-foot">{run.turns || 0} turn{(run.turns || 0) === 1 ? '' : 's'}</div>
          </div>
        </div>

        <div className="details-grid">
          <div>
            <h3 className="t-heading-sm" style={{ margin: '0 0 12px' }}>Trigger</h3>
            <div className="info-list">
              <div className="info-row"><span className="k">Source</span><span className="v">Playground (HTTP)</span></div>
              <div className="info-row"><span className="k">Agent</span><span className="v mono">{run.agent_id}</span></div>
              <div className="info-row"><span className="k">Run ID</span><span className="v mono">{run.id}</span></div>
              {run.exit_reason && <div className="info-row"><span className="k">Exit reason</span><span className="v mono">{run.exit_reason}</span></div>}
              {run.error && <div className="info-row"><span className="k">Error</span><span className="v" style={{ color: 'var(--err)' }}>{run.error}</span></div>}
            </div>

            <h3 className="t-heading-sm" style={{ margin: '28px 0 12px' }}>Model</h3>
            <div className="info-list">
              <div className="info-row"><span className="k">Input tokens</span><span className="v mono">{window.UI.fmtTokens(run.total_input_tokens)}</span></div>
              <div className="info-row"><span className="k">Output tokens</span><span className="v mono">{window.UI.fmtTokens(run.total_output_tokens)}</span></div>
              <div className="info-row"><span className="k">Cached tokens</span><span className="v mono">{window.UI.fmtTokens(run.total_cached_tokens || 0)}</span></div>
              <div className="info-row"><span className="k">Total cost</span><span className="v mono">{window.UI.fmtCost(run.total_cost_usd)}</span></div>
            </div>

            <h3 className="t-heading-sm" style={{ margin: '28px 0 12px' }}>Waterfall</h3>
            <div className="wf">
              {spans.length === 0
                ? <div style={{ color: 'var(--stone)', fontSize: 12, padding: '8px 0' }}>No spans recorded.</div>
                : spans.map((s, i) => {
                    const left = ((s.start_ts - t0) * 1000 / totalMs) * 100;
                    const width = Math.max(0.5, ((s.duration_ms || 0) / totalMs) * 100);
                    const cls = s.status === 'err' ? 'err' : s.kind;
                    return (
                      <div className="wf-row" key={i}>
                        <div className="name">
                          <span className={`ico-sq ${cls}`} />
                          <span>{s.name}</span>
                        </div>
                        <div className="bar">
                          <div className={`f ${cls}`} style={{ left: `${left}%`, width: `${width}%` }} />
                        </div>
                        <div className="dur">{window.UI.fmtMs(s.duration_ms)}</div>
                      </div>
                    );
                  })}
            </div>
          </div>

          <div>
            <h3 className="t-heading-sm" style={{ margin: '0 0 12px' }}>Timeline</h3>
            <div className="timeline">
              {events.length === 0
                ? <div style={{ color: 'var(--stone)', fontSize: 12, padding: '8px 0' }}>No events.</div>
                : events.map((e, i) => {
                    const kind = eventKind(e.type);
                    return (
                      <div key={i} className="t-event">
                        <div className="t-time">{window.UI.fmtTime(e.timestamp)}</div>
                        <div className={`t-dot kind-${kind}`} />
                        <div>
                          <div className="t-label">{e.type}</div>
                          <div className="t-detail">{summarizeEvent(e)}</div>
                        </div>
                      </div>
                    );
                  })}
            </div>
          </div>
        </div>

        <h3 className="t-heading-sm" style={{ margin: '40px 0 12px' }}>Cost breakdown</h3>
        <div className="cost-bar">
          {breakdown.map((b, i) => (
            <div key={i} className="cost-seg" style={{ width: `${b.pct}%`, background: b.color }} />
          ))}
        </div>
        <div className="cost-legend">
          {breakdown.map((b) => (
            <div key={b.label} className="cost-item">
              <span className="d" style={{ background: b.color }} />
              <span className="l">{b.label}</span>
              <span className="v">{window.UI.fmtMs(b.ms)} · {b.pct.toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function eventKind(type) {
  if (!type) return 'thought';
  if (type === 'RunStarted') return 'trigger';
  if (type === 'RunCompleted') return 'done';
  if (type === 'RunErrored' || type === 'ToolCallFailed') return 'err';
  if (type.startsWith('Tool')) return 'tool';
  if (type === 'MemoryAccessed') return 'memory';
  if (type === 'ModelResponded' || type === 'TurnStarted' || type === 'StreamChunkEvent') return 'llm';
  return 'thought';
}

function summarizeEvent(e) {
  const p = e.payload || {};
  if (e.type === 'ToolCallStarted' && p.call) return `${p.call.name}(${JSON.stringify(p.call.args || {}).slice(0, 80)})`;
  if (e.type === 'ToolCallCompleted' && p.call) return `${p.call.name} → ${String(p.output || '').slice(0, 80)}`;
  if (e.type === 'ToolCallFailed' && p.call) return `${p.call.name} failed: ${String(p.error || '').slice(0, 80)}`;
  if (e.type === 'ModelResponded' && p.usage) return `usage: in ${p.usage.input_tokens || 0} · out ${p.usage.output_tokens || 0}`;
  if (e.type === 'MemoryAccessed') return `${p.op || 'access'} · ${p.key || ''}`;
  if (e.type === 'RunCompleted') return p.exit_reason ? `exit: ${p.exit_reason}` : 'run completed';
  if (e.type === 'RunErrored') return p.error || 'errored';
  if (e.type === 'StreamChunkEvent') return `chunk · ${(p.data || '').slice(0, 40)}`;
  return Object.keys(p).length ? JSON.stringify(p).slice(0, 80) : '';
}

function computeBreakdown(spans, run) {
  const kinds = { llm: 0, tool: 0, memory: 0 };
  spans.forEach((s) => { kinds[s.kind] = (kinds[s.kind] || 0) + (s.duration_ms || 0); });
  const total = Math.max(1, kinds.llm + kinds.tool + kinds.memory);
  return [
    { label: 'Model (LLM)', ms: kinds.llm, pct: (kinds.llm / total) * 100, color: 'var(--ink)' },
    { label: 'Tools', ms: kinds.tool, pct: (kinds.tool / total) * 100, color: 'var(--graphite)' },
    { label: 'Memory', ms: kinds.memory, pct: (kinds.memory / total) * 100, color: 'var(--stone)' },
  ];
}

window.DetailsTab = DetailsTab;
