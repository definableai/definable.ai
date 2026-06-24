// Right-side config panel host — tab switcher (Definition / Memory / Evals) + body router.

function ConfigPanel({ tab, onTab, agent, runs, metrics, selectedRun, onSelectRun }) {
  const I = window.Icons;
  return (
    <React.Fragment>
      <div className="config-top">
        <div className="config-tabs">
          {[
            { k: 'definition', label: 'Definition' },
            { k: 'memory', label: 'Memory' },
            { k: 'evals', label: 'Evals' },
          ].map((t) => (
            <button key={t.k} className={`config-tab ${tab === t.k ? 'active' : ''}`} onClick={() => onTab(t.k)}>{t.label}</button>
          ))}
        </div>
        <button className="btn-icon" title="More"><I.IconMoreV size={16} /></button>
      </div>
      {tab === 'definition' && <window.DefinitionView agent={agent} runs={runs} metrics={metrics} selectedRun={selectedRun} />}
      {tab === 'memory' && <window.MemoryView agent={agent} runs={runs} selectedRun={selectedRun} />}
      {tab === 'evals' && <window.EvalsView agent={agent} runs={runs} metrics={metrics} />}
    </React.Fragment>
  );
}

window.ConfigPanel = ConfigPanel;
