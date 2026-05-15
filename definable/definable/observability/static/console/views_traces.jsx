// TRACES TAB — left run list, right run detail (waterfall + kv + events).
// Live SSE prepends new runs as they complete.
const { useState: useTS, useEffect: useEffTS, useRef: useRefTS } = React;

function TracesTab({ agent }) {
  const [runs, setRuns] = useTS([]);
  const [selId, setSelId] = useTS(null);
  const [detail, setDetail] = useTS(null);
  const [loading, setLoading] = useTS(true);
  const seenRef = useRefTS(new Set());

  // Initial fetch + SSE live append.
  useEffTS(() => {
    let live = true;
    setLoading(true);
    window.DEFINABLE_API.listRuns({ agent: agent && agent.id, limit: 200 }).then((rows) => {
      if (!live) return;
      seenRef.current = new Set(rows.map((r) => r.id));
      setRuns(rows);
      if (rows.length && !selId) setSelId(rows[0].id);
      setLoading(false);
    }).catch(() => setLoading(false));

    const unsub = window.DEFINABLE_API.stream((e) => {
      if (!live) return;
      if (agent && e.agent && e.agent !== agent.id) return;
      // RunCompleted / RunErrored — refresh the parent row (it may be new or updated).
      if (e.type === 'RunCompleted' || e.type === 'RunErrored') {
        window.DEFINABLE_API.listRuns({ agent: agent && agent.id, limit: 200 }).then((rows) => {
          if (!live) return;
          seenRef.current = new Set(rows.map((r) => r.id));
          setRuns(rows);
        });
      }
    }, agent && agent.id);

    return () => { live = false; unsub(); };
  }, [agent && agent.id]);

  useEffTS(() => {
    if (!selId) { setDetail(null); return; }
    let live = true;
    window.DEFINABLE_API.getRun(selId).then((d) => { if (live) setDetail(d); });
    return () => { live = false; };
  }, [selId]);

  // 60-bin histogram over the last 60 minutes.
  const histo = (() => {
    const bins = new Array(60).fill(0);
    const errs = new Array(60).fill(0);
    const now = Date.now() / 1000;
    runs.forEach((r) => {
      const ageM = (now - (r.started_at || 0)) / 60;
      const idx = 59 - Math.floor(ageM);
      if (idx < 0 || idx > 59) return;
      bins[idx] += 1;
      if (r.status === 'errored') errs[idx] += 1;
    });
    const max = Math.max(1, ...bins);
    return bins.map((v, i) => ({ v: (v / max) * 100, err: errs[i] > 0 }));
  })();

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="filterbar">
        <div className="flt"><span className="lbl">RANGE</span> last 60m</div>
        <div className="flt"><span className="lbl">AGENT</span> {agent ? agent.id : 'all'}</div>
        <div className="flt"><span className="lbl">STATUS</span> all</div>
        <div className="flt search">
          <span className="lbl">⌕</span>
          <input placeholder="search by run id, agent, output…" />
        </div>
        <div className="flt"><Pill kind="acc">↻ LIVE</Pill></div>
      </div>

      <div className="histo">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span className="caps">RUNS — LAST 60m</span>
          <span style={{ color: 'var(--ink-3)', fontSize: 10, letterSpacing: '0.1em' }}>
            {runs.length} runs ·{' '}
            <span style={{ color: 'var(--err)' }}>{runs.filter((r) => r.status === 'errored').length} errored</span>
          </span>
        </div>
        <div className="histo-bars">
          {histo.map((b, i) => (
            <div key={i} className={`bar ${b.err ? 'err' : b.v > 60 ? 'hot' : ''}`} style={{ height: `${Math.max(4, b.v)}%` }} />
          ))}
        </div>
        <div className="histo-foot"><span>-60m</span><span>-30m</span><span>NOW</span></div>
      </div>

      <div className="tr-app" style={{ flex: 1 }}>
        {/* LIST */}
        <div className="tr-list">
          {loading ? <UI.EmptyState title="LOADING…" /> : runs.length === 0 ? (
            <UI.EmptyState title="NO RUNS YET" sub="Call agent.arun(...) and refresh — runs appear here as soon as they complete." />
          ) : (
            <table className="brut">
              <thead>
                <tr>
                  <th style={{ width: 14 }}></th>
                  <th>RUN ID</th>
                  <th>TS</th>
                  <th>AGENT</th>
                  <th>STATUS</th>
                  <th className="num">TURNS</th>
                  <th className="num">TOK</th>
                  <th className="num">COST</th>
                  <th className="num">DUR</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.id} className={r.id === selId ? 'sel' : ''} onClick={() => setSelId(r.id)}>
                    <td><Dot kind={r.status === 'errored' ? 'err' : r.status === 'running' ? 'warn' : 'ok'} /></td>
                    <td style={{ color: 'var(--accent)' }}>{r.id.slice(0, 8)}</td>
                    <td style={{ color: 'var(--ink-3)' }}>{UI.fmtRelative(r.started_at)}</td>
                    <td>{r.agent_id}</td>
                    <td><Pill kind={r.status === 'errored' ? 'err' : r.status === 'running' ? 'warn' : 'ok'}>{r.status.toUpperCase()}</Pill></td>
                    <td className="num">{r.turns || 0}</td>
                    <td className="num">{UI.fmtTokens((r.total_input_tokens || 0) + (r.total_output_tokens || 0))}</td>
                    <td className="num">{UI.fmtCost(r.total_cost_usd)}</td>
                    <td className="num">{r.ended_at ? UI.fmtMs((r.ended_at - r.started_at) * 1000) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* DETAIL */}
        <div className="tr-detail">
          {!detail ? <UI.EmptyState title="—" sub="Select a run to inspect its waterfall and events." /> : (
            <RunDetail detail={detail} />
          )}
        </div>
      </div>
    </div>
  );
}

function RunDetail({ detail }) {
  const run = detail.run;
  const spans = detail.spans || [];
  const events = detail.events || [];
  const t0 = spans.length ? Math.min(...spans.map((s) => s.start_ts)) : run.started_at;
  const tN = spans.length ? Math.max(...spans.map((s) => s.end_ts || s.start_ts)) : (run.ended_at || run.started_at);
  const totalMs = Math.max(1, (tN - t0) * 1000);

  return (
    <div>
      <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--rule)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, display: 'flex', gap: 8, alignItems: 'center' }}>
            <Dot kind={run.status === 'errored' ? 'err' : 'ok'} />
            <span style={{ color: 'var(--accent)' }}>{run.id}</span>
            <span style={{ color: 'var(--ink-3)', fontWeight: 400 }}>{run.agent_id}</span>
          </div>
          <div style={{ marginTop: 4, color: 'var(--ink-2)', fontSize: 11 }}>{run.output || run.error || '—'}</div>
        </div>
        <div style={{ display: 'flex', gap: 16, fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.1em' }}>
          <span>{run.turns} turns</span>
          <span>{UI.fmtTokens((run.total_input_tokens || 0) + (run.total_output_tokens || 0))} tok</span>
          <span>{UI.fmtCost(run.total_cost_usd)}</span>
        </div>
      </div>

      <div className="hdr-row"><h2>◇ WATERFALL</h2></div>
      <div className="wf">
        {spans.length === 0 ? <div style={{ padding: 16, color: 'var(--ink-3)', fontSize: 11 }}>No spans recorded.</div> : spans.map((s, i) => {
          const left = ((s.start_ts - t0) * 1000 / totalMs) * 100;
          const width = Math.max(0.5, ((s.duration_ms || 0) / totalMs) * 100);
          return (
            <div className="wf-row" key={i}>
              <div className="name">
                <span className={`ico ${s.kind} ${s.status === 'err' ? 'err' : ''}`} />
                <span>{s.name}</span>
              </div>
              <div className="bar">
                <div className={`f ${s.status === 'err' ? 'err' : s.kind}`} style={{ left: `${left}%`, width: `${width}%` }} />
              </div>
              <div className="dur">{UI.fmtMs(s.duration_ms)}</div>
            </div>
          );
        })}
      </div>

      <div className="hdr-row"><h2>⧉ ATTRIBUTES</h2></div>
      <div className="kvgrid">
        <div className="k">started</div><div className="v">{UI.fmtTime(run.started_at)}</div>
        <div className="k">ended</div><div className="v">{UI.fmtTime(run.ended_at)}</div>
        <div className="k">input tok</div><div className="v">{UI.fmtTokens(run.total_input_tokens)}</div>
        <div className="k">output tok</div><div className="v">{UI.fmtTokens(run.total_output_tokens)}</div>
        <div className="k">cached tok</div><div className="v">{UI.fmtTokens(run.total_cached_tokens)}</div>
        <div className="k">cost</div><div className="v">{UI.fmtCost(run.total_cost_usd)}</div>
        <div className="k">exit reason</div><div className="v">{run.exit_reason || '—'}</div>
        {run.error && <><div className="k">error</div><div className="v" style={{ color: 'var(--err)' }}>{run.error}</div></>}
      </div>

      <div className="hdr-row"><h2>⌘ EVENT LOG</h2></div>
      <div style={{ padding: '0 14px 24px' }}>
        {events.length === 0 ? <div style={{ color: 'var(--ink-3)', fontSize: 11, padding: 8 }}>No events.</div> : events.map((e, i) => (
          <div key={i} style={{ display: 'grid', gridTemplateColumns: '80px 1fr', gap: 12, padding: '4px 0', borderBottom: '1px solid var(--rule)', fontSize: 11 }}>
            <span style={{ color: 'var(--ink-3)' }}>{UI.fmtTime(e.timestamp)}</span>
            <span><span style={{ color: 'var(--accent)' }}>{e.type}</span> <span style={{ color: 'var(--ink-3)' }}>{e.payload && Object.keys(e.payload).length ? JSON.stringify(e.payload).slice(0, 160) : ''}</span></span>
          </div>
        ))}
      </div>
    </div>
  );
}

window.TracesTab = TracesTab;
