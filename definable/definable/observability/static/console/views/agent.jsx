// Agent tab — Conversation (chat with playground SSE) + Composer.

function AgentTab({ agent, recentToolNames, recentSkillNames }) {
  const [messages, setMessages] = React.useState([]);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState(null);
  const scrollRef = React.useRef(null);

  React.useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, busy]);

  // Reset chat whenever the user picks a different agent.
  React.useEffect(() => {
    setMessages([]);
    setError(null);
  }, [agent && agent.id]);

  const send = async (text) => {
    if (!agent) return;
    const input = (text || '').trim();
    if (!input || busy) return;
    setError(null);
    setMessages((m) => m.concat([{ role: 'user', content: input, ts: Date.now() / 1000 }]));
    setBusy(true);

    let assistantContent = '';
    const totals = { input_tokens: 0, output_tokens: 0 };
    const tools = [];
    const pendingTools = {};
    setMessages((m) => m.concat([{ role: 'assistant', content: '', tokens: totals, tools, ts: Date.now() / 1000 }]));

    try {
      await window.API.playgroundRun(agent.id, input, (evt) => {
        const t = evt.type;
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
            tools.push({ name: open.name, args: open.args, status: 'err', error: evt.data.error, duration_ms: (evt.timestamp - open.started_at) * 1000 });
            delete pendingTools[c.id];
          }
        } else if (t === 'RunCompleted') {
          if (evt.data && typeof evt.data.content === 'string' && !assistantContent) {
            assistantContent = evt.data.content;
          }
        } else if (t === 'RunErrored' || t === 'error') {
          setError(evt.data && (evt.data.error || evt.error) || 'run failed');
        }
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

  return (
    <div className="chat-region">
      <div className="chat-scroll scroll-y" ref={scrollRef}>
        <div className="chat-inner">
          <IntroMessage agent={agent} />
          {messages.map((m, i) => m.role === 'user'
            ? <UserMsg key={i} msg={m} />
            : <AgentMsg key={i} msg={m} agent={agent} busy={busy && i === messages.length - 1} />)}
        </div>
      </div>
      <window.Composer
        onSend={send}
        busy={busy}
        error={error}
        recentToolNames={recentToolNames}
        recentSkillNames={recentSkillNames}
        disabled={!agent}
      />
    </div>
  );
}

function IntroMessage({ agent }) {
  if (!agent) {
    return (
      <window.UI.EmptyState
        title="No agent registered"
        sub="Construct an Agent with observability=True to begin."
      />
    );
  }
  const initials = window.UI.fmtInitials(agent.id);
  const intro = agent.instructions ? agent.instructions.slice(0, 320) : null;
  return (
    <div className="msg msg-agent">
      <div className="msg-meta">
        <div className="avatar" style={{ width: 22, height: 22, fontSize: 9 }}>{initials}</div>
        <span className="who">{agent.id} · live</span>
        <span className="time">{agent.model}</span>
      </div>
      <div className="content">
        {intro ? (
          <p style={{ whiteSpace: 'pre-wrap' }}>{intro}{agent.instructions.length > 320 ? '…' : ''}</p>
        ) : (
          <p>This agent has no system prompt set. Send a message to begin.</p>
        )}
      </div>
    </div>
  );
}

function UserMsg({ msg }) {
  return (
    <div className="msg msg-user">
      <div className="msg-meta">
        <span className="who">You · {window.UI.fmtTime(msg.ts)}</span>
      </div>
      <div className="bubble">{msg.content}</div>
    </div>
  );
}

function AgentMsg({ msg, agent, busy }) {
  const I = window.Icons;
  const initials = window.UI.fmtInitials(agent && agent.id);
  const hasContent = msg.content && msg.content.length > 0;
  return (
    <div className="msg msg-agent">
      <div className="msg-meta">
        <div className="avatar" style={{ width: 22, height: 22, fontSize: 9 }}>{initials}</div>
        <span className="who">{(agent && agent.id) || 'agent'} · {window.UI.fmtTime(msg.ts)}</span>
        {busy && <span className="time">streaming…</span>}
      </div>
      <div className="content">
        {(msg.tools || []).map((t, i) => (
          <div key={i} className={`tool-call ${t.status === 'err' ? 'err' : 'success'}`}>
            <div className="ico"><I.IconCube size={12} /></div>
            <div style={{ minWidth: 0, flex: 1 }}>
              <span className="name">{t.name}</span>
              {t.status === 'err'
                ? <span className="dim">— error: {String(t.error || '').slice(0, 80)}</span>
                : t.args && Object.keys(t.args).length > 0
                  ? <span className="dim">— {JSON.stringify(t.args).slice(0, 80)}</span>
                  : null}
            </div>
            <span className="meta">{window.UI.fmtMs(t.duration_ms)}</span>
          </div>
        ))}
        {hasContent && <p className="body">{msg.content}</p>}
        {!hasContent && busy && <p style={{ color: 'var(--stone)' }}>…</p>}
        {msg.tokens && (msg.tokens.input_tokens || msg.tokens.output_tokens) ? (
          <div className="tokens-foot">
            <span>IN <strong>{window.UI.fmtTokens(msg.tokens.input_tokens)}</strong></span>
            <span>OUT <strong>{window.UI.fmtTokens(msg.tokens.output_tokens)}</strong></span>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function Composer({ onSend, busy, error, recentToolNames, recentSkillNames, disabled }) {
  const I = window.Icons;
  const [text, setText] = React.useState('');
  const [openMenu, setOpenMenu] = React.useState(null); // 'attach' | 'skill' | 'tool' | 'schedule' | null
  const taRef = React.useRef(null);

  const submit = () => {
    if (!text.trim() || busy || disabled) return;
    const v = text;
    setText('');
    onSend(v);
  };

  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const insertAtCursor = (snippet) => {
    setText((cur) => (cur ? cur + ' ' + snippet : snippet));
    if (taRef.current) taRef.current.focus();
    setOpenMenu(null);
  };

  return (
    <div>
      <div className="composer-wrap">
        <div className="composer">
          <textarea
            ref={taRef}
            placeholder={disabled ? 'No agent — register an Agent(observability=True) first.' : 'Reply to your agent — or ask it to do something new'}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKey}
            rows={2}
            disabled={disabled || busy}
          />
          <div className="composer-foot">
            <ComposerChip
              icon={<I.IconPaperclip size={13} />}
              label="Attach"
              menu={openMenu === 'attach'}
              onClick={() => setOpenMenu((v) => v === 'attach' ? null : 'attach')}
            >
              <div className="empty">File attachments — upload endpoint coming soon. The selected file's name will be referenced inline.</div>
              <button className="item" onClick={() => insertAtCursor('[attachment: report.csv]')}>[attachment: report.csv]</button>
              <button className="item" onClick={() => insertAtCursor('[attachment: image.png]')}>[attachment: image.png]</button>
            </ComposerChip>
            <ComposerChip
              icon={<I.IconSkill size={13} />}
              label="Skill"
              menu={openMenu === 'skill'}
              onClick={() => setOpenMenu((v) => v === 'skill' ? null : 'skill')}
            >
              {(recentSkillNames || []).length === 0
                ? <div className="empty">No skills attached to this agent yet.</div>
                : (recentSkillNames || []).map((nm) => (
                    <button key={nm} className="item" onClick={() => insertAtCursor(`@skill:${nm}`)}>@skill:{nm}</button>
                  ))}
            </ComposerChip>
            <ComposerChip
              icon={<I.IconCube size={13} />}
              label="Tool"
              menu={openMenu === 'tool'}
              onClick={() => setOpenMenu((v) => v === 'tool' ? null : 'tool')}
            >
              {(recentToolNames || []).length === 0
                ? <div className="empty">No tools used by this agent yet.</div>
                : (recentToolNames || []).map((nm) => (
                    <button key={nm} className="item" onClick={() => insertAtCursor(`@tool:${nm}`)}>@tool:{nm}</button>
                  ))}
            </ComposerChip>
            <ComposerChip
              icon={<I.IconCal size={13} />}
              label="Schedule"
              menu={openMenu === 'schedule'}
              onClick={() => setOpenMenu((v) => v === 'schedule' ? null : 'schedule')}
            >
              <div className="empty">Scheduling endpoint: <code>POST /api/agents/{'{id}'}/schedule</code> — coming soon.</div>
              <button className="item" onClick={() => insertAtCursor('[schedule: every 1h]')}>[schedule: every 1h]</button>
              <button className="item" onClick={() => insertAtCursor('[schedule: cron 0 9 * * *]')}>[schedule: cron 0 9 * * *]</button>
            </ComposerChip>
            <button className="send" onClick={submit} disabled={!text.trim() || busy || disabled} title="Send">
              <I.IconArrowUp size={15} stroke={2.2} />
            </button>
          </div>
        </div>
        <div className="composer-help">
          {error ? <span className="err">Error: {error}</span> : <span>Press ↵ to send · ⇧↵ for newline</span>}
        </div>
      </div>
    </div>
  );
}

function ComposerChip({ icon, label, menu, onClick, children }) {
  return (
    <div style={{ position: 'relative' }}>
      <button className={`composer-chip ${menu ? 'active' : ''}`} onClick={onClick} type="button">
        {icon} {label}
      </button>
      {menu && <div className="chip-menu">{children}</div>}
    </div>
  );
}

window.AgentTab = AgentTab;
window.Composer = Composer;
