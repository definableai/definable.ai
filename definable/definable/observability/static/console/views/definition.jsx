// Right config — Definition tab body.
// Reads live agent state + metrics + recent runs. Editable fields are read-only
// today but each carries an apiHint pointing at the future PATCH endpoint.

function DefinitionView({ agent, runs, metrics, selectedRun }) {
  const I = window.Icons;
  const UI = window.UI;
  if (!agent) {
    return (
      <div className="config-body scroll-y">
        <UI.EmptyState title="No agent selected" sub="Once an Agent is registered the definition appears here." />
      </div>
    );
  }
  const m = metrics || {};
  const successRate = m.runs > 0 ? (((m.runs - m.errors) / m.runs) * 100).toFixed(1) + '%' : '—';
  const avgCost = m.runs > 0 ? m.cost_usd / m.runs : 0;
  const live = (runs || []).some((r) => r.status === 'running');

  // Aggregate distinct tool names from recent runs' spans — not available here directly,
  // but we synthesise from selectedRun if any, plus event-derived names later.
  const tools = collectToolNames(selectedRun);

  return (
    <div className="config-body scroll-y">
      <div className="agent-hero">
        <div className="t-eyebrow eyebrow">Managed agent</div>
        <h1 className="name">
          <UI.EditableField value={agent.id} apiHint={`PATCH /api/agents/${agent.id} {"name":"…"}`} />
        </h1>
        <div className="handle">handle: <code>@{agent.id}</code> · <code>{agent.model}</code></div>
        <div className="agent-meta-row">
          <div className="stat">
            <span className="label">Status</span>
            <span className={`value ${live ? 'live' : ''}`}>{live ? 'Running' : 'Idle'}</span>
          </div>
          <div className="stat">
            <span className="label">Runs · 24h</span>
            <span className="value">{m.runs || 0}</span>
          </div>
          <div className="stat">
            <span className="label">Success</span>
            <span className="value">{successRate}</span>
          </div>
          <div className="stat">
            <span className="label">Avg cost</span>
            <span className="value">{UI.fmtCost(avgCost)}</span>
          </div>
        </div>
      </div>

      <UI.AccordionSection title="Preferences">
        <div className="model-row">
          <div className="star"><I.IconSparkle size={14} stroke={2} /></div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="name">
              <UI.EditableField value={agent.model} apiHint={`PATCH /api/agents/${agent.id} {"model":"…"}`} />
            </div>
            <div className="sub">registered {UI.fmtRelative(agent.registered_at)}</div>
          </div>
          <I.IconChevronRight size={16} className="chev-r" />
        </div>
        <div className="prompt-box">
          <strong>System prompt</strong>
          <div style={{ marginTop: 6 }}>
            {agent.instructions || <em style={{ color: 'var(--stone)' }}>No system prompt set.</em>}
          </div>
          {agent.instructions && agent.instructions.length > 220 && <div className="fade" />}
          <button className="edit" title="PATCH /api/agents/{id} {instructions: ...} — coming soon">
            <I.IconEdit size={12} />
          </button>
        </div>
        <div className="toggle-row" style={{ marginTop: 14 }}>
          <div className="ico-wrap"><I.IconGitBranch size={14} /></div>
          <div className="label-stack">
            <div className="l1">Allow self-updates</div>
            <div className="l2">Agent can propose changes to its own prompt + skills</div>
          </div>
          <UI.Toggle on={false} />
        </div>
        <div className="toggle-row">
          <div className="ico-wrap"><I.IconUsers size={14} /></div>
          <div className="label-stack">
            <div className="l1">Human-in-the-loop</div>
            <div className="l2">Require approval before destructive tool calls</div>
          </div>
          <UI.Toggle on={false} />
        </div>
      </UI.AccordionSection>

      <UI.AccordionSection title="Endpoints" status="Live">
        <div className="trigger-row">
          <div className="ico"><I.IconBolt size={16} /></div>
          <div className="label">
            <div className="nm">Playground</div>
            <div className="when">POST /api/playground/run · agent={agent.id}</div>
          </div>
          <span className="pill-status on">Live</span>
        </div>
        <div className="trigger-row">
          <div className="ico"><I.IconWebhook size={16} /></div>
          <div className="label">
            <div className="nm">Event stream</div>
            <div className="when">GET /api/stream?agent={agent.id}</div>
          </div>
          <span className="pill-status on">Live</span>
        </div>
        <div className="trigger-row">
          <div className="ico"><I.IconClock size={16} /></div>
          <div className="label">
            <div className="nm">Schedule</div>
            <div className="when">POST /api/agents/{agent.id}/schedule — coming soon</div>
          </div>
          <span className="pill-status">Off</span>
        </div>
        <button className="add-row" title="Triggers API coming soon"><I.IconPlus size={13} /> Add trigger</button>
      </UI.AccordionSection>

      <UI.AccordionSection title="Tools" status={tools.length ? `${tools.length} active` : 'None'}>
        {tools.length === 0 ? (
          <div style={{ color: 'var(--stone)', fontSize: 12, padding: '8px 0' }}>
            No tool calls observed yet. They appear here as the agent uses them.
          </div>
        ) : tools.map((t) => (
          <div key={t.name} className="app-card">
            <div className="logo">{t.name[0].toUpperCase()}</div>
            <div className="info">
              <div className="nm">{t.name}</div>
              <div className="sub">{t.calls} call{t.calls === 1 ? '' : 's'} · {UI.fmtMs(t.totalMs)}</div>
            </div>
            <div className="right">
              <span className="pill-status on">Active</span>
            </div>
          </div>
        ))}
        <button className="add-row" title="POST /api/agents/{id}/tools — coming soon">
          <I.IconPlus size={13} /> Attach tool
        </button>
      </UI.AccordionSection>

      <UI.AccordionSection title="Capabilities" defaultOpen={false}>
        <div className="toggle-row compact">
          <div className="ico-wrap"><I.IconBolt size={14} /></div>
          <div className="label-stack"><div className="l1">Streaming</div></div>
          <span className="pill-status on">On</span>
        </div>
        <div className="toggle-row compact">
          <div className="ico-wrap"><I.IconCube size={14} /></div>
          <div className="label-stack"><div className="l1">Tool calls</div></div>
          <span className="pill-status on">On</span>
        </div>
        <div className="toggle-row compact">
          <div className="ico-wrap"><I.IconChat size={14} /></div>
          <div className="label-stack"><div className="l1">Memory access</div></div>
          <span className="pill-status on">On</span>
        </div>
        <div className="toggle-row compact">
          <div className="ico-wrap"><I.IconGlobe size={14} /></div>
          <div className="label-stack"><div className="l1">Web fetch</div></div>
          <span className="pill-status">Tool-defined</span>
        </div>
      </UI.AccordionSection>

      <div style={{ padding: '20px 24px 32px', borderTop: '1px solid var(--hairline)' }}>
        <div className="t-eyebrow" style={{ color: 'var(--stone)' }}>Telemetry · 24h</div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginTop: 10 }}>
          <div className="t-meta" style={{ color: 'var(--graphite)' }}>P50 latency</div>
          <div className="t-meta" style={{ color: 'var(--ink)' }}>{UI.fmtMs(m.p50_ms)}</div>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginTop: 6 }}>
          <div className="t-meta" style={{ color: 'var(--graphite)' }}>P95 latency</div>
          <div className="t-meta" style={{ color: 'var(--ink)' }}>{UI.fmtMs(m.p95_ms)}</div>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginTop: 6 }}>
          <div className="t-meta" style={{ color: 'var(--graphite)' }}>Tokens · 24h</div>
          <div className="t-meta" style={{ color: 'var(--ink)' }}>{UI.fmtTokens(m.input_tokens || 0)} in · {UI.fmtTokens(m.output_tokens || 0)} out</div>
        </div>
      </div>
    </div>
  );
}

function collectToolNames(selectedRun) {
  if (!selectedRun || !selectedRun.spans) return [];
  const map = {};
  selectedRun.spans.forEach((s) => {
    if (s.kind !== 'tool') return;
    if (!map[s.name]) map[s.name] = { name: s.name, calls: 0, totalMs: 0 };
    map[s.name].calls += 1;
    map[s.name].totalMs += s.duration_ms || 0;
  });
  return Object.values(map);
}

window.DefinitionView = DefinitionView;
window.collectToolNames = collectToolNames;
