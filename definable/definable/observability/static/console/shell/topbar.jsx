// Center top bar — title block + share/clone stubs + tab strip + mobile triggers.

function TopBar({ tab, onTab, agent, selectedRun, onToggleNav, onToggleConfig }) {
  const I = window.Icons;
  const TABS = [
    { k: 'agent', label: 'Agent' },
    { k: 'settings', label: 'Settings' },
    { k: 'runs', label: 'Runs' },
    { k: 'details', label: 'Details' },
  ];
  const eyebrow = selectedRun
    ? `Run · #${(selectedRun.id || '').slice(0, 8)}`
    : agent
      ? `Agent · ${agent.model}`
      : 'Observability';
  const subline = selectedRun
    ? `${window.UI.fmtTokens((selectedRun.total_input_tokens || 0) + (selectedRun.total_output_tokens || 0))} tok · ${window.UI.fmtCost(selectedRun.total_cost_usd)}`
    : 'process-scoped';
  const title = agent ? agent.id : '—';
  return (
    <div className="topbar">
      <button className="btn-icon topbar-mobile-nav-btn" onClick={onToggleNav} title="Menu">
        <I.Icon size={16}><path d="M3 6h18M3 12h18M3 18h18" /></I.Icon>
      </button>
      <div className="title-block">
        <div className="title-row">
          <span className="t-eyebrow" style={{ color: 'var(--stone)' }}>{eyebrow}</span>
          <span className="pipe">·</span>
          <span className="t-meta" style={{ color: 'var(--graphite)' }}>{subline}</span>
        </div>
        <div className="title">{title}</div>
      </div>
      <div className="topbar-actions">
        <button className="btn btn-ghost" style={{ height: 32, padding: '0 12px' }} title="Share read-only link (coming soon)">
          <I.IconShare size={14} /> Share
        </button>
        <button className="btn btn-primary" style={{ height: 32, padding: '0 14px' }} title="Clone agent (coming soon)">
          <I.IconCopy size={13} /> Clone
        </button>
        <button className="btn-icon topbar-mobile-config-btn" onClick={onToggleConfig} title="Config">
          <I.IconChevronsRight size={16} />
        </button>
        <div className="tabs">
          {TABS.map((t) => (
            <button key={t.k} className={`tab ${tab === t.k ? 'active' : ''}`} onClick={() => onTab(t.k)}>{t.label}</button>
          ))}
        </div>
      </div>
      <div className="topbar-mobile-tabs">
        {TABS.map((t) => (
          <button key={t.k} className={`tab ${tab === t.k ? 'active' : ''}`} onClick={() => onTab(t.k)}>{t.label}</button>
        ))}
      </div>
    </div>
  );
}

window.TopBar = TopBar;
