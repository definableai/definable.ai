// Settings tab — 6 sections (Identity, Instructions, Tools, Memory, Deploy, Billing).
// Read-only; each editable field carries an `apiHint` showing the future PATCH/POST.

function SettingsTab({ agent, runs, metrics, selectedRun }) {
  const [section, setSection] = React.useState('identity');
  const UI = window.UI;
  const sections = [
    { id: 'identity', label: 'Identity' },
    { id: 'instructions', label: 'Instructions' },
    { id: 'tools', label: 'Tools & apps' },
    { id: 'memory', label: 'Memory' },
    { id: 'deploy', label: 'Deployment' },
    { id: 'billing', label: 'Billing' },
  ];

  if (!agent) {
    return (
      <div className="settings-view scroll-y">
        <div className="settings-inner" style={{ gridTemplateColumns: '1fr' }}>
          <UI.EmptyState title="No agent selected" sub="Register an Agent(observability=True) to configure it." />
        </div>
      </div>
    );
  }

  return (
    <div className="settings-view scroll-y">
      <div className="settings-inner">
        <aside className="settings-rail">
          <div className="t-eyebrow" style={{ color: 'var(--stone)', padding: '0 12px 10px' }}>Configuration</div>
          {sections.map((s) => (
            <button key={s.id} className={`settings-rail-item ${section === s.id ? 'active' : ''}`} onClick={() => setSection(s.id)}>
              {s.label}
            </button>
          ))}
        </aside>
        <div className="settings-body">
          {section === 'identity' && <SettingsIdentity agent={agent} />}
          {section === 'instructions' && <SettingsInstructions agent={agent} />}
          {section === 'tools' && <SettingsTools selectedRun={selectedRun} />}
          {section === 'memory' && <SettingsMemory runs={runs} />}
          {section === 'deploy' && <SettingsDeploy agent={agent} />}
          {section === 'billing' && <SettingsBilling metrics={metrics} />}
        </div>
      </div>
    </div>
  );
}

function SectionHead({ eyebrow, title, lede }) {
  return (
    <div className="section-head">
      <div className="t-eyebrow" style={{ color: 'var(--stone)' }}>{eyebrow}</div>
      <h2 className="t-heading-md" style={{ margin: '8px 0 12px', color: 'var(--ink)' }}>{title}</h2>
      {lede && <p style={{ color: 'var(--graphite)', maxWidth: 620, margin: 0, fontSize: 14.5, lineHeight: 1.55 }}>{lede}</p>}
    </div>
  );
}

function Field({ label, helper, children }) {
  return (
    <div className="field">
      <label className="field-label">{label}</label>
      {children}
      {helper && <div className="field-help">{helper}</div>}
    </div>
  );
}

function SettingsIdentity({ agent }) {
  return (
    <React.Fragment>
      <SectionHead
        eyebrow="01 — Identity"
        title="What this agent is"
        lede="A name, a one-line purpose, and a handle that downstream APIs use to address it."
      />
      <div className="settings-card">
        <div className="field-row two">
          <Field label="Display name" helper="PATCH /api/agents/{id} — coming soon">
            <input className="field-input" value={agent.id} readOnly />
          </Field>
          <Field label="Handle" helper="Used as ?agent= in stream + playground endpoints.">
            <div className="field-input" style={{ fontFamily: 'JetBrains Mono, monospace' }}>@{agent.id}</div>
          </Field>
        </div>
        <Field label="Model">
          <input className="field-input" value={agent.model} readOnly style={{ fontFamily: 'JetBrains Mono, monospace' }} />
        </Field>
        <Field label="Registered" helper="When this agent first attached to the observability server.">
          <div className="field-input" style={{ fontFamily: 'JetBrains Mono, monospace' }}>
            {window.UI.fmtTimestamp(agent.registered_at)}
          </div>
        </Field>
      </div>
    </React.Fragment>
  );
}

function SettingsInstructions({ agent }) {
  return (
    <React.Fragment>
      <SectionHead
        eyebrow="02 — Instructions"
        title="System prompt"
        lede="Read on every run. Mutation via PATCH /api/agents/{id}/instructions — coming soon."
      />
      <div className="settings-card">
        <Field label="System prompt">
          <textarea
            className="field-textarea"
            rows={12}
            value={agent.instructions || ''}
            readOnly
            placeholder="No system prompt set."
          />
        </Field>
      </div>
    </React.Fragment>
  );
}

function SettingsTools({ selectedRun }) {
  const tools = window.collectToolNames ? window.collectToolNames(selectedRun) : [];
  return (
    <React.Fragment>
      <SectionHead
        eyebrow="03 — Tools & apps"
        title="What the agent can reach"
        lede="Tools observed from recent spans. Attach/detach via POST /api/agents/{id}/tools — coming soon."
      />
      <div className="settings-card">
        {tools.length === 0 ? (
          <div style={{ color: 'var(--stone)', fontSize: 13 }}>
            No tool calls observed yet. They appear here as the agent uses them.
          </div>
        ) : (
          <div className="tool-grid">
            {tools.map((t) => (
              <div key={t.name} className="tool-tile">
                <div className="tool-tile-head">
                  <div style={{ width: 28, height: 28, borderRadius: 6, background: 'var(--canvas-warm)', border: '1px solid var(--hairline-soft)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'JetBrains Mono, monospace', fontWeight: 600 }}>{t.name[0].toUpperCase()}</div>
                  <window.UI.Toggle on={true} />
                </div>
                <div className="tool-tile-name">{t.name}</div>
                <div className="tool-tile-desc">{t.calls} call{t.calls === 1 ? '' : 's'} · {window.UI.fmtMs(t.totalMs)}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </React.Fragment>
  );
}

function SettingsMemory({ runs }) {
  return (
    <React.Fragment>
      <SectionHead
        eyebrow="04 — Memory"
        title="What the agent remembers"
        lede="Backed by the trace store. Mutation via PATCH /api/agents/{id}/memory — coming soon."
      />
      <div className="settings-card">
        <div className="setting-toggle">
          <div>
            <div className="st-label">Episodic memory</div>
            <div className="st-desc">Every run is recorded. Currently <strong>{(runs || []).length} entries</strong>.</div>
          </div>
          <window.UI.Toggle on={true} />
        </div>
        <div className="setting-toggle">
          <div>
            <div className="st-label">Semantic recall</div>
            <div className="st-desc">Vector search over past runs and pinned facts.</div>
          </div>
          <window.UI.Toggle on={false} />
        </div>
        <div className="setting-toggle">
          <div>
            <div className="st-label">Forget after</div>
            <div className="st-desc">Automatically discard runs older than the threshold.</div>
          </div>
          <select className="field-select" style={{ width: 140 }} defaultValue="never" disabled>
            <option value="30">30 days</option>
            <option value="90">90 days</option>
            <option value="never">Never</option>
          </select>
        </div>
      </div>
    </React.Fragment>
  );
}

function SettingsDeploy({ agent }) {
  const origin = window.location.origin;
  return (
    <React.Fragment>
      <SectionHead eyebrow="05 — Deployment" title="Where this agent runs" />
      <div className="settings-card">
        <Field label="Environment">
          <div className="env-chips">
            <div className="env-chip active"><span className="d" /> Local <span className="v">process</span></div>
            <div className="env-chip"><span className="d" /> Staging <span className="v">—</span></div>
            <div className="env-chip"><span className="d" /> Production <span className="v">—</span></div>
          </div>
        </Field>
        <Field label="Playground URL" helper="POST to this URL to invoke the agent.">
          <div className="field-input field-input-flex">
            <code style={{ flex: 1, fontFamily: 'JetBrains Mono, monospace', fontSize: 12.5, color: 'var(--ink)', wordBreak: 'break-all' }}>
              {origin}/api/playground/run
            </code>
            <button className="btn-text" onClick={() => navigator.clipboard.writeText(`${origin}/api/playground/run`)}>Copy</button>
          </div>
        </Field>
        <Field label="Event stream" helper="GET SSE — long-lived; auto-reconnects.">
          <div className="field-input field-input-flex">
            <code style={{ flex: 1, fontFamily: 'JetBrains Mono, monospace', fontSize: 12.5, color: 'var(--ink)', wordBreak: 'break-all' }}>
              {origin}/api/stream?agent={agent.id}
            </code>
            <button className="btn-text" onClick={() => navigator.clipboard.writeText(`${origin}/api/stream?agent=${agent.id}`)}>Copy</button>
          </div>
        </Field>
        <Field label="Example curl">
          <pre style={{ background: 'var(--canvas-warm)', border: '1px solid var(--hairline)', borderRadius: 'var(--r-sm)', padding: 12, fontFamily: 'JetBrains Mono, monospace', fontSize: 12, lineHeight: 1.55, color: 'var(--ink)', overflow: 'auto', margin: 0 }}>
{`curl -N -X POST ${origin}/api/playground/run \\
  -H 'content-type: application/json' \\
  -d '{"agent":"${agent.id}","input":"hello"}'`}
          </pre>
        </Field>
      </div>
    </React.Fragment>
  );
}

function SettingsBilling({ metrics }) {
  const m = metrics || {};
  const tok = (m.input_tokens || 0) + (m.output_tokens || 0);
  return (
    <React.Fragment>
      <SectionHead
        eyebrow="06 — Billing"
        title="Spend on this agent"
        lede="Live from /api/metrics?range=24h."
      />
      <div className="settings-card">
        <div className="metric-row">
          <div className="metric">
            <div className="m-label">Runs · 24h</div>
            <div className="m-value">{m.runs || 0}</div>
            <div className="m-foot">{m.errors || 0} errored</div>
          </div>
          <div className="metric">
            <div className="m-label">USD · 24h</div>
            <div className="m-value">{window.UI.fmtCost(m.cost_usd)}</div>
            <div className="m-foot">avg {window.UI.fmtCost((m.runs && m.cost_usd) ? m.cost_usd / m.runs : 0)} / run</div>
          </div>
          <div className="metric">
            <div className="m-label">Tokens · 24h</div>
            <div className="m-value">{window.UI.fmtTokens(tok)}</div>
            <div className="m-foot">{window.UI.fmtTokens(m.input_tokens || 0)} in · {window.UI.fmtTokens(m.output_tokens || 0)} out</div>
          </div>
          <div className="metric">
            <div className="m-label">Throughput</div>
            <div className="m-value">{(m.rps || 0).toFixed(3)}</div>
            <div className="m-foot">runs / sec</div>
          </div>
        </div>
      </div>
    </React.Fragment>
  );
}

window.SettingsTab = SettingsTab;
