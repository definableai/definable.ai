// Right config — Memory tab body.

function MemoryView({ agent, runs, selectedRun }) {
  const I = window.Icons;
  const UI = window.UI;
  if (!agent) {
    return (
      <div className="config-body scroll-y">
        <UI.EmptyState title="No agent selected" />
      </div>
    );
  }
  const totalRuns = (runs || []).length;
  // Pinned runs = top 5 most recent (proxy for "facts" — episodic anchors).
  const pinned = (runs || []).slice(0, 5);
  // Memory recalls = MemoryAccessed events from selectedRun (if any).
  const recalls = (selectedRun && selectedRun.events)
    ? selectedRun.events.filter((e) => e.type === 'MemoryAccessed')
    : [];

  return (
    <div className="config-body scroll-y">
      <div className="agent-hero">
        <div className="t-eyebrow eyebrow">Memory</div>
        <h1 className="name">What the agent remembers</h1>
        <div className="handle">{totalRuns} episodic entries · trace-backed</div>
        <div className="agent-meta-row">
          <div className="stat">
            <span className="label">Episodic</span>
            <span className="value">{totalRuns}</span>
          </div>
          <div className="stat">
            <span className="label">Recalls · this run</span>
            <span className="value">{recalls.length}</span>
          </div>
          <div className="stat">
            <span className="label">Namespace</span>
            <span className="value" style={{ fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }}>observability</span>
          </div>
        </div>
      </div>

      <UI.AccordionSection title="Pinned runs" status={`${pinned.length} pinned`}>
        {pinned.length === 0
          ? <div style={{ color: 'var(--stone)', fontSize: 12, padding: '8px 0' }}>No runs yet — episodic store is empty.</div>
          : pinned.map((r, i) => (
              <div key={r.id} className="fact-row">
                <div className={`fact-tag ${r.status === 'errored' ? 'tag-correction' : 'tag-pinned'}`}>
                  {r.status === 'errored' ? 'errored' : 'pinned'}
                </div>
                <div className="fact-text">{r.input || r.output || r.error || r.id}</div>
                <div className="fact-source">run #{(r.id || '').slice(0, 8)} · {UI.fmtRelative(r.started_at)}</div>
              </div>
            ))}
      </UI.AccordionSection>

      <UI.AccordionSection title="Recent recalls">
        {recalls.length === 0
          ? <div style={{ color: 'var(--stone)', fontSize: 12, padding: '8px 0' }}>
              No MemoryAccessed events on the current run.
            </div>
          : recalls.map((e, i) => (
              <div key={i} className="recall-row">
                <div className="recall-q">{(e.payload && e.payload.op) || 'access'} · {(e.payload && e.payload.key) || '—'}</div>
                <div className="recall-hit">→ {String((e.payload && e.payload.value) || '').slice(0, 80) || '—'}</div>
                <div className="recall-time">{UI.fmtTime(e.timestamp)}</div>
              </div>
            ))}
      </UI.AccordionSection>

      <UI.AccordionSection title="Episodic store">
        <div className="ep-row">
          <div>
            <div className="ep-label">Total runs stored</div>
            <div className="ep-sub">trace store · process-scoped</div>
          </div>
          <div className="ep-val">{totalRuns}</div>
        </div>
        <div className="ep-row">
          <div>
            <div className="ep-label">Storage backend</div>
            <div className="ep-sub">aiosqlite · namespaced</div>
          </div>
          <div className="ep-val">SQLite</div>
        </div>
        <div className="ep-row">
          <div>
            <div className="ep-label">Embedding model</div>
            <div className="ep-sub">not configured for this agent</div>
          </div>
          <button className="btn-text" style={{ fontSize: 12 }} title="PATCH /api/agents/{id}/memory — coming soon">Change</button>
        </div>
        <div style={{ display: 'flex', gap: 8, marginTop: 14 }}>
          <button className="btn btn-ghost" style={{ height: 32, padding: '0 14px' }} title="Coming soon">Export memory</button>
          <button className="btn btn-ghost" style={{ height: 32, padding: '0 14px' }} title="Coming soon">Reset all</button>
        </div>
      </UI.AccordionSection>
    </div>
  );
}

window.MemoryView = MemoryView;
