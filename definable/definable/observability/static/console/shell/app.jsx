// App root — theme + drawer state + agent/run selection state + data hooks + mount.

function App() {
  // Theme persisted in localStorage; default 'paper'.
  const [theme, setTheme] = window.useLocalStorage('definable_theme', 'paper');
  React.useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme === 'ink' ? 'ink' : 'paper');
  }, [theme]);

  // Drawer state (mobile only — collapses on resize).
  const [navOpen, setNavOpen] = React.useState(false);
  const [configOpen, setConfigOpen] = React.useState(false);
  React.useEffect(() => {
    const onResize = () => {
      if (window.innerWidth > 1024) setConfigOpen(false);
      if (window.innerWidth > 820) setNavOpen(false);
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // Tab state.
  const [tab, setTab] = React.useState('agent');
  const [configTab, setConfigTab] = React.useState('definition');

  // Selection state.
  const [agentId, setAgentId] = React.useState(null);
  const [selectedRunId, setSelectedRunId] = React.useState(null);

  // Data hooks.
  const agentsHook = window.useAgents(5_000);
  const agents = agentsHook.data || [];
  const currentAgentId = agentId || (agents[0] && agents[0].id) || null;
  const agent = agents.find((a) => a.id === currentAgentId) || null;

  const runsHook = window.useRuns(currentAgentId);
  const runs = runsHook.data || [];

  const runDetailHook = window.useRun(selectedRunId);
  const runDetail = runDetailHook.data;

  const metricsHook = window.useMetrics('24h');
  const metrics = metricsHook.data;

  // SSE-driven live updates: refresh runs whenever an event arrives for the
  // current agent. Trace store broadcasts every persisted event.
  window.useStream(React.useCallback(() => {
    runsHook.refresh();
    if (selectedRunId) runDetailHook.refresh();
  }, [runsHook, runDetailHook, selectedRunId]), currentAgentId);

  // When agent changes, clear selected run.
  React.useEffect(() => { setSelectedRunId(null); }, [currentAgentId]);

  // Aggregated tool / skill names from runs (for composer chips).
  const toolNames = React.useMemo(() => {
    if (!runDetail || !runDetail.spans) return [];
    return Array.from(new Set(runDetail.spans.filter((s) => s.kind === 'tool').map((s) => s.name)));
  }, [runDetail]);

  const onNewRun = React.useCallback(() => {
    setSelectedRunId(null);
    setTab('agent');
  }, []);

  const onToggleTheme = React.useCallback(() => {
    setTheme(theme === 'ink' ? 'paper' : 'ink');
  }, [theme, setTheme]);

  return (
    <div className="app">
      <aside className={`nav ${navOpen ? 'open' : ''}`}>
        <window.Sidebar
          agents={agents}
          agentId={currentAgentId}
          onSelectAgent={(id) => { setAgentId(id); setNavOpen(false); }}
          runs={runs}
          selectedRunId={selectedRunId}
          onSelectRun={(id) => { setSelectedRunId(id); setTab('details'); setNavOpen(false); }}
          onNewRun={onNewRun}
          theme={theme}
          onToggleTheme={onToggleTheme}
          onCloseNav={() => setNavOpen(false)}
        />
      </aside>
      <div className={`nav-backdrop ${navOpen ? 'open' : ''}`} onClick={() => setNavOpen(false)} />

      <main className="main">
        <window.TopBar
          tab={tab}
          onTab={setTab}
          agent={agent}
          selectedRun={runDetail && runDetail.run}
          onToggleNav={() => setNavOpen(!navOpen)}
          onToggleConfig={() => setConfigOpen(!configOpen)}
        />
        {tab === 'agent' && <window.AgentTab agent={agent} recentToolNames={toolNames} recentSkillNames={[]} />}
        {tab === 'settings' && <window.SettingsTab agent={agent} runs={runs} metrics={metrics} selectedRun={runDetail} />}
        {tab === 'runs' && <window.RunsTab
          runs={runs}
          metrics={metrics}
          onSelectRun={setSelectedRunId}
          selectedRunId={selectedRunId}
          onSwitchTab={setTab}
        />}
        {tab === 'details' && <window.DetailsTab
          selectedRunId={selectedRunId}
          runDetail={runDetail}
          loading={runDetailHook.loading}
        />}
      </main>

      <aside className={`config ${configOpen ? 'open' : ''}`}>
        <window.ConfigPanel
          tab={configTab}
          onTab={setConfigTab}
          agent={agent}
          runs={runs}
          metrics={metrics}
          selectedRun={runDetail}
          onSelectRun={setSelectedRunId}
        />
      </aside>
      <div className={`config-backdrop ${configOpen ? 'open' : ''}`} onClick={() => setConfigOpen(false)} />
    </div>
  );
}

window.App = App;
ReactDOM.createRoot(document.getElementById('root')).render(<App />);
