// METRICS TAB — aggregated KPIs over a window. Re-fetches every 10s.
const { useState: useMT, useEffect: useEffMT } = React;

function MetricsTab() {
  const [range, setRange] = useMT('1h');
  const [data, setData] = useMT(null);
  const [loading, setLoading] = useMT(true);

  useEffMT(() => {
    let live = true;
    const fetchOnce = () => {
      window.DEFINABLE_API.metrics(range).then((d) => {
        if (live) { setData(d); setLoading(false); }
      });
    };
    fetchOnce();
    const id = setInterval(fetchOnce, 10_000);
    return () => { live = false; clearInterval(id); };
  }, [range]);

  if (loading) {
    return <div style={{ padding: 32 }}><UI.EmptyState title="LOADING…" /></div>;
  }
  if (!data) {
    return <UI.EmptyState title="NO METRICS" />;
  }

  return (
    <div className="scroll" style={{ height: '100%' }}>
      <div className="filterbar">
        {['1h', '6h', '24h'].map((r) => (
          <button key={r} className="flt" onClick={() => setRange(r)} style={{ cursor: 'pointer', color: range === r ? 'var(--accent)' : 'var(--ink-2)' }}>
            <span className="lbl">RANGE</span> {r}
          </button>
        ))}
        <div className="flt search" />
      </div>

      <div className="statgrid">
        <UI.StatCell label="RUNS" val={String(data.runs)} delta={`${data.errors} errored`} deltaDir={data.errors > 0 ? 'dn' : 'up'} />
        <UI.StatCell label="REQ / SEC" val={(data.rps || 0).toFixed(3)} />
        <UI.StatCell label="P50 LATENCY" val={UI.fmtMs(data.p50_ms)} />
        <UI.StatCell label="P95 LATENCY" val={UI.fmtMs(data.p95_ms)} />
        <UI.StatCell label="SPEND" val={UI.fmtCost(data.cost_usd)} delta={`${UI.fmtTokens(data.input_tokens + data.output_tokens)} tok`} />
      </div>

      <div className="met-grid">
        <div className="met-card">
          <h3>TOKENS
            <div className="leg">
              <div className="it"><span className="sw" style={{ background: 'var(--accent)' }} /> input</div>
              <div className="it"><span className="sw" style={{ background: 'var(--info)' }} /> output</div>
            </div>
          </h3>
          <div className="chart" style={{ display: 'flex', alignItems: 'flex-end', gap: 12, padding: 8 }}>
            <BarBlock label="IN" value={data.input_tokens} accent="var(--accent)" />
            <BarBlock label="OUT" value={data.output_tokens} accent="var(--info)" />
          </div>
        </div>
        <div className="met-card">
          <h3>RUNS BREAKDOWN</h3>
          <div className="chart" style={{ display: 'flex', alignItems: 'flex-end', gap: 12, padding: 8 }}>
            <BarBlock label="OK" value={data.runs - data.errors} accent="var(--ok)" />
            <BarBlock label="ERR" value={data.errors} accent="var(--err)" />
          </div>
        </div>
        <div className="met-card">
          <h3>LATENCY (P50 / P95)</h3>
          <div className="chart" style={{ display: 'flex', alignItems: 'flex-end', gap: 12, padding: 8 }}>
            <BarBlock label="P50" value={data.p50_ms} fmt={UI.fmtMs} accent="var(--ok)" />
            <BarBlock label="P95" value={data.p95_ms} fmt={UI.fmtMs} accent="var(--warn)" />
          </div>
        </div>
        <div className="met-card">
          <h3>COST</h3>
          <div className="chart" style={{ display: 'flex', alignItems: 'flex-end', padding: 8 }}>
            <BarBlock label="USD" value={data.cost_usd} fmt={UI.fmtCost} accent="var(--accent)" />
          </div>
        </div>
      </div>
    </div>
  );
}

function BarBlock({ label, value, accent, fmt }) {
  const display = fmt ? fmt(value) : UI.fmtTokens(value);
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', gap: 6 }}>
      <div style={{ height: 100, position: 'relative', background: 'var(--bg-3)' }}>
        <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, background: accent, height: `${Math.min(100, value > 0 ? Math.max(8, Math.log10(value + 1) * 22) : 4)}%` }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.1em' }}>
        <span>{label}</span><span style={{ color: 'var(--ink)' }}>{display}</span>
      </div>
    </div>
  );
}

window.MetricsTab = MetricsTab;
