// APP SHELL — sidebar agent list + topbar agent header + tab bar + tab content.
// Sidebar polls /api/agents every 5s. Default tab = playground.
const { useState, useEffect } = React;

function Sidebar({ agents, selectedId, onSelect }) {
  return (
    <div className="sb">
      <div className="sb-brand">
        <div className="mark"><span>D</span></div>
        <div className="name">DEFINABLE</div>
        <div className="role">CONSOLE</div>
      </div>

      <div className="sb-sec">
        <div className="sb-sec-hd"><span>WORKSPACE</span></div>
        <div style={{ padding: '0 14px', display: 'grid', gap: 4 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--ink-2)' }}>
            <span>definable / local</span><span style={{ color: 'var(--accent)' }}>▾</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--ink-3)' }}>
            <span>{agents.length} agent{agents.length === 1 ? '' : 's'}</span>
            <span>process</span>
          </div>
        </div>
      </div>

      <div className="sb-sec" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <div className="sb-sec-hd"><span>AGENTS · {agents.length}</span></div>
        <div className="sb-search">
          <span style={{ color: 'var(--ink-3)' }}>⌕</span>
          <input placeholder="search agents…" />
          <span className="kbd">⌘K</span>
        </div>
        <div className="sb-list scroll">
          {agents.length === 0 ? (
            <div style={{ padding: 16, color: 'var(--ink-3)', fontSize: 11, lineHeight: 1.6 }}>
              No agents registered yet. Construct an Agent with <span style={{ color: 'var(--accent)' }}>observability=True</span> to begin.
            </div>
          ) : agents.map((a) => (
            <div key={a.id} className={`sb-row ${selectedId === a.id ? 'active' : ''}`} onClick={() => onSelect(a.id)}>
              <div className="dot ok" />
              <div className="nm">{a.id}<span className="v">{a.model}</span></div>
            </div>
          ))}
        </div>
      </div>

      <div className="sb-foot">
        <div className="who">
          <div className="av">D</div>
          <div style={{ flex: 1 }}>
            <div>observability</div>
            <div style={{ color: 'var(--ink-3)', fontSize: 10 }}>local dashboard</div>
          </div>
          <span className="caps" style={{ color: 'var(--accent)' }}>LIVE</span>
        </div>
        <div className="links">
          <a href="https://docs.definable.ai" target="_blank" rel="noreferrer">DOCS</a>
          <a href="https://github.com/definable-ai/definable" target="_blank" rel="noreferrer">SDK</a>
        </div>
      </div>
    </div>
  );
}

function TopBar({ agent }) {
  return (
    <div className="tb">
      <div className="tb-l">
        <div className="tb-crumbs">
          <span>WORKSPACE</span><span>/</span><span>AGENTS</span><span>/</span>
          <span className="cur">{agent ? agent.id.toUpperCase() : '—'}</span>
        </div>
        {agent && (
          <div className="tb-agent">
            <div className="h">{agent.id}</div>
            <span className="ver" style={{ color: 'var(--ink-3)' }}>{agent.model}</span>
          </div>
        )}
      </div>
      <div className="tb-r">
        <div className="stat"><Dot kind="ok" /> <span>HEALTHY</span></div>
        <span className="pill live">● LIVE</span>
      </div>
    </div>
  );
}

function TabBar({ tab, setTab }) {
  const TABS = [
    { k: 'playground', label: 'PLAYGROUND' },
    { k: 'traces', label: 'TRACES' },
    { k: 'metrics', label: 'METRICS' },
  ];
  return (
    <div className="tabs">
      {TABS.map((t) => (
        <button key={t.k} className={`tab ${tab === t.k ? 'active' : ''}`} onClick={() => setTab(t.k)}>
          {t.label}
        </button>
      ))}
      <div className="spacer" />
      <div className="toolset">
        <span className="pill">PY SDK</span>
        <span className="pill"><span style={{ color: 'var(--accent)' }}>agent.arun(...)</span></span>
      </div>
    </div>
  );
}

function App() {
  const [tab, setTab] = useState('playground');
  const [agents, setAgents] = useState([]);
  const [agentId, setAgentId] = useState(null);

  useEffect(() => {
    let live = true;
    const refresh = () => {
      window.DEFINABLE_API.listAgents().then((rows) => {
        if (!live) return;
        setAgents(rows);
        if (!agentId && rows.length) setAgentId(rows[0].id);
      });
    };
    refresh();
    const id = setInterval(refresh, 5_000);
    return () => { live = false; clearInterval(id); };
  }, [agentId]);

  const agent = agents.find((a) => a.id === agentId) || agents[0] || null;

  return (
    <div className="app">
      <Sidebar agents={agents} selectedId={agent && agent.id} onSelect={setAgentId} />
      <div className="main">
        <TopBar agent={agent} />
        <TabBar tab={tab} setTab={setTab} />
        <div style={{ overflow: 'hidden' }}>
          {tab === 'playground' && <PlaygroundTab agent={agent} />}
          {tab === 'traces' && <TracesTab agent={agent} />}
          {tab === 'metrics' && <MetricsTab />}
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
