// PLAYGROUND TAB — left config, center chat, right live trace.
// Sends user input to POST /api/playground/run and streams events back via SSE.
const { useState: usePG, useRef: useRefPG, useEffect: useEffPG } = React;

function PlaygroundTab({ agent }) {
  const [messages, setMessages] = usePG([]); // {role, content, tokens, cost, tools}
  const [draft, setDraft] = usePG('');
  const [busy, setBusy] = usePG(false);
  const [error, setError] = usePG(null);
  const [trace, setTrace] = usePG([]); // event log for live pane (current run only)
  const msgsRef = useRefPG(null);

  useEffPG(() => {
    if (msgsRef.current) msgsRef.current.scrollTop = msgsRef.current.scrollHeight;
  }, [messages, busy]);

  if (!agent) {
    return (
      <div style={{ height: '100%', display: 'grid', placeItems: 'center' }}>
        <UI.EmptyState title="NO AGENT SELECTED" sub="Construct an Agent with observability=True to begin." />
      </div>
    );
  }

  const send = async () => {
    const input = draft.trim();
    if (!input || busy) return;
    setDraft('');
    setError(null);
    setMessages((m) => m.concat([{ role: 'user', content: input }]));
    setBusy(true);
    setTrace([]);

    // Pending assistant message that we'll fill as ModelResponded events arrive.
    let assistantContent = '';
    let totals = { input_tokens: 0, output_tokens: 0, cost_usd: 0 };
    let tools = [];
    const pendingTools = {}; // id -> {name, args, started_at}

    setMessages((m) => m.concat([{ role: 'assistant', content: '', tokens: totals, tools }]));

    try {
      await window.DEFINABLE_API.playgroundRun(agent.id, input, (evt) => {
        const t = evt.type;
        setTrace((arr) => arr.concat([evt]));
        if (t === 'StreamChunkEvent' && evt.data && evt.data.kind === 'content') {
          assistantContent += evt.data.data || '';
        } else if (t === 'ModelResponded') {
          if (evt.data && typeof evt.data.content === 'string') {
            assistantContent = evt.data.content;
          }
          const usage = evt.data && evt.data.usage;
          if (usage) {
            totals.input_tokens += usage.input_tokens || 0;
            totals.output_tokens += usage.output_tokens || 0;
          }
        } else if (t === 'ToolCallStarted') {
          const c = evt.data && evt.data.call;
          if (c) pendingTools[c.id] = { name: c.name, args: c.args, started_at: evt.timestamp };
        } else if (t === 'ToolCallCompleted') {
          const c = evt.data && evt.data.call;
          if (c && pendingTools[c.id]) {
            const open = pendingTools[c.id];
            tools.push({ name: open.name, args: open.args, output: evt.data.output, status: 'ok', duration_ms: (evt.timestamp - open.started_at) * 1000 });
            delete pendingTools[c.id];
          }
        } else if (t === 'ToolCallFailed') {
          const c = evt.data && evt.data.call;
          if (c && pendingTools[c.id]) {
            const open = pendingTools[c.id];
            tools.push({ name: open.name, args: open.args, output: null, status: 'err', error: evt.data.error, duration_ms: (evt.timestamp - open.started_at) * 1000 });
            delete pendingTools[c.id];
          }
        } else if (t === 'RunCompleted') {
          if (evt.data && typeof evt.data.content === 'string' && !assistantContent) {
            assistantContent = evt.data.content;
          }
        }
        // Push interim state every event so the UI updates live.
        setMessages((m) => {
          const cp = m.slice();
          const last = cp[cp.length - 1];
          if (last && last.role === 'assistant') {
            cp[cp.length - 1] = Object.assign({}, last, { content: assistantContent, tokens: { ...totals }, tools: tools.slice() });
          }
          return cp;
        });
      });
    } catch (e) {
      setError(String(e && e.message ? e.message : e));
    } finally {
      setBusy(false);
    }
  };

  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="pg">
      {/* CONFIG */}
      <div className="pg-config scroll">
        <div className="cfg-section">
          <h4>AGENT</h4>
          <div className="cfg-row"><span className="k">name</span><span className="v">{agent.id}</span></div>
          <div className="cfg-row"><span className="k">model</span><span className="v">{agent.model}</span></div>
          <div className="cfg-row"><span className="k">registered</span><span className="v">{UI.fmtRelative(agent.registered_at)}</span></div>
        </div>
        {agent.instructions && (
          <div className="cfg-section">
            <h4>SYSTEM PROMPT</h4>
            <div className="cfg-promptbox">{agent.instructions}</div>
          </div>
        )}
      </div>

      {/* CHAT */}
      <div className="pg-chat">
        <div className="pg-msgs scroll" ref={msgsRef}>
          {messages.length === 0 && <UI.EmptyState title="PLAYGROUND" sub="Send a message to invoke the live agent. Events stream back as the loop runs." />}
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              <div className="who">{m.role.toUpperCase()}</div>
              <div>
                <div className="body">{m.content || (busy && i === messages.length - 1 ? '…' : '')}</div>
                {(m.tools || []).map((t, j) => (
                  <div className="tool" key={j}>
                    <span><span className="n">▸ {t.name}</span></span>
                    <pre>{JSON.stringify(t.args)}</pre>
                    <span style={{ color: t.status === 'err' ? 'var(--err)' : 'var(--ink-3)', fontSize: 10 }}>{UI.fmtMs(t.duration_ms)}</span>
                  </div>
                ))}
                {m.tokens && (m.tokens.input_tokens || m.tokens.output_tokens) ? (
                  <div className="meta">
                    <span>IN {UI.fmtTokens(m.tokens.input_tokens)}</span>
                    <span>OUT {UI.fmtTokens(m.tokens.output_tokens)}</span>
                  </div>
                ) : null}
              </div>
            </div>
          ))}
          {error && <div className="msg assistant"><div className="who">ERROR</div><div><div className="body" style={{ color: 'var(--err)' }}>{error}</div></div></div>}
        </div>
        <div className="pg-input">
          <div className="box">
            <textarea
              placeholder="Send a message…"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKey}
              rows={2}
              disabled={busy}
            />
            <div className="row">
              <div className="l">{busy ? <span style={{ color: 'var(--accent)' }}>● RUNNING</span> : <span>↵ to send  ·  ⇧↵ newline</span>}</div>
              <button className="send" onClick={send} disabled={busy || !draft.trim()}>SEND</button>
            </div>
          </div>
        </div>
      </div>

      {/* LIVE TRACE */}
      <div className="pg-trace scroll">
        <div className="tr-head">
          <h4>LIVE TRACE</h4>
          {busy ? <div className="live"><span className="pulse" /> STREAMING</div> : <span className="caps">IDLE</span>}
        </div>
        {trace.length === 0 ? (
          <UI.EmptyState title="—" sub="Events appear here while the agent loop runs." />
        ) : (
          trace.map((e, i) => {
            const cls = e.type === 'ModelResponded' || e.type === 'TurnStarted' || e.type === 'StreamChunkEvent' ? 'llm'
              : e.type === 'ToolCallStarted' || e.type === 'ToolCallCompleted' ? 'tool'
              : e.type === 'ToolCallFailed' || e.type === 'RunErrored' ? 'err'
              : e.type === 'MemoryAccessed' ? 'kb' : '';
            return (
              <div className={`span ${cls}`} key={i}>
                <div className="marker" />
                <div className="body">
                  <div className="line">
                    <span className="name">{e.type}</span>
                    <span className="meta">{UI.fmtTime(e.timestamp)}</span>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

window.PlaygroundTab = PlaygroundTab;
