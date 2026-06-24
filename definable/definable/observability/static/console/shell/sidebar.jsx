// Left sidebar — brand, new-run CTA, agent dropdown, recents (live), workspace footer.

function Sidebar({
  agents,
  agentId,
  onSelectAgent,
  runs,
  selectedRunId,
  onSelectRun,
  onNewRun,
  theme,
  onToggleTheme,
  onCloseNav,
}) {
  const I = window.Icons;
  const [agentMenuOpen, setAgentMenuOpen] = React.useState(false);
  const [query, setQuery] = React.useState('');
  const selectedAgent = agents.find((a) => a.id === agentId) || null;

  const filteredRuns = React.useMemo(() => {
    const q = query.trim().toLowerCase();
    const src = runs || [];
    if (!q) return src;
    return src.filter((r) => {
      return (
        (r.id || '').toLowerCase().includes(q) ||
        (r.input || '').toLowerCase().includes(q) ||
        (r.output || '').toLowerCase().includes(q)
      );
    });
  }, [runs, query]);

  const totalCost = React.useMemo(() => {
    return (runs || []).reduce((sum, r) => sum + (r.total_cost_usd || 0), 0);
  }, [runs]);
  const totalTok = React.useMemo(() => {
    return (runs || []).reduce((sum, r) => sum + (r.total_input_tokens || 0) + (r.total_output_tokens || 0), 0);
  }, [runs]);

  return (
    <React.Fragment>
      <div className="nav-brand">
        <div className="wordmark">
          <span className="dot" />definable
        </div>
        <button className="btn-icon" title="Collapse sidebar" onClick={onCloseNav}>
          <I.IconSidebar size={16} />
        </button>
      </div>

      <div className="nav-section" style={{ paddingTop: 16 }}>
        <button className="nav-cta" onClick={onNewRun}>
          <I.IconPlus size={14} stroke={2} />
          New agent run
        </button>
        <div style={{ height: 8 }} />
        <div style={{ position: 'relative' }}>
          <div className="agent-select" onClick={() => setAgentMenuOpen((v) => !v)}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="nm">{selectedAgent ? selectedAgent.id : 'No agent registered'}</div>
              <div className="sub">{selectedAgent ? selectedAgent.model : 'observability=True'}</div>
            </div>
            <I.IconChevronDown size={14} className="chev" />
          </div>
          {agentMenuOpen && agents.length > 0 && (
            <div className="agent-dropdown">
              {agents.map((a) => (
                <div
                  key={a.id}
                  className={`row ${a.id === agentId ? 'active' : ''}`}
                  onClick={() => { onSelectAgent(a.id); setAgentMenuOpen(false); }}
                >
                  <div className="nm">{a.id}</div>
                  <div className="sub">{a.model}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="nav-section" style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', paddingBottom: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 12px 6px' }}>
          <div className="t-eyebrow" style={{ color: 'var(--stone)' }}>Recents</div>
          <span className="t-meta" style={{ color: 'var(--stone)', fontFamily: 'JetBrains Mono, monospace' }}>
            {(runs || []).length}
          </span>
        </div>
        <div className="nav-search">
          <I.IconSearch size={14} />
          <input
            placeholder="Search runs…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <div className="scroll-y" style={{ flex: 1, minHeight: 0, padding: '0 4px 12px' }}>
          {filteredRuns.length === 0 ? (
            <div style={{ padding: '16px 12px', color: 'var(--stone)', fontSize: 12, lineHeight: 1.5 }}>
              {(runs || []).length === 0
                ? 'No runs yet. Call agent.arun(…) to start.'
                : 'No matches.'}
            </div>
          ) : filteredRuns.map((r) => {
            const live = r.status === 'running';
            const err = r.status === 'errored';
            const cls = `recent-item ${r.id === selectedRunId ? 'active' : ''} ${live ? 'live' : ''} ${err ? 'errored' : ''}`;
            const title = r.input || r.output || r.error || r.id;
            return (
              <div key={r.id} className={cls} onClick={() => onSelectRun(r.id)}>
                <div className="dot" />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="title">{title}</div>
                  <div className="meta">{window.UI.fmtRelative(r.started_at)} · {window.UI.fmtCost(r.total_cost_usd)}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="nav-footer">
        <div className="workspace-bar">
          <div className="row">
            <span className="label">Workspace spend</span>
            <span className="value">{(runs || []).length} runs</span>
          </div>
          <div className="amount">{window.UI.fmtCost(totalCost)}<span> · {window.UI.fmtTokens(totalTok)} tok</span></div>
          <div className="foot">
            <span className="t-meta" style={{ color: 'var(--stone)' }}>local · process-scoped</span>
            <button className="theme-toggle" onClick={onToggleTheme} title="Toggle theme">
              {theme === 'ink' ? <I.IconSun size={12} /> : <I.IconMoon size={12} />}
              {theme === 'ink' ? 'Light' : 'Dark'}
            </button>
          </div>
        </div>
        <div className="user-row">
          <div className="avatar">D</div>
          <div style={{ minWidth: 0 }}>
            <div className="name">observability</div>
            <div className="role">local dashboard · {agents.length} agent{agents.length === 1 ? '' : 's'}</div>
          </div>
          <div className="more"><I.IconMoreV size={14} /></div>
        </div>
      </div>
    </React.Fragment>
  );
}

window.Sidebar = Sidebar;
