// Shared presentational atoms used by every view. Brutalist, monospace, no Tailwind.
// All components are stateless and pure — they accept props, return JSX, never fetch.

function Dot({ kind }) {
  const cls = (kind === 'ok' || kind === 'warn' || kind === 'err' || kind === 'idle') ? kind : 'idle';
  return <span className={`dot ${cls}`} />;
}

function Pill({ kind, children }) {
  const k = (kind === 'ok' || kind === 'warn' || kind === 'err' || kind === 'info' || kind === 'acc') ? kind : '';
  return <span className={`tag ${k}`}>{children}</span>;
}

// Inline SVG sparkline. `data` is an array of numbers; auto-scales y-axis.
function Spark({ data, w, h, color }) {
  const width = w || 64;
  const height = h || 16;
  const stroke = color || 'var(--accent)';
  if (!data || !data.length) {
    return <svg className="spark" width={width} height={height} />;
  }
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = (max - min) || 1;
  const step = width / Math.max(1, data.length - 1);
  const pts = data.map((v, i) => {
    const x = i * step;
    const y = height - ((v - min) / range) * height;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  return (
    <svg className="spark" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <polyline fill="none" stroke={stroke} strokeWidth="1.25" points={pts} />
    </svg>
  );
}

// Single KPI cell used in the .statgrid layout.
function StatCell({ label, val, unit, delta, deltaDir, sparkData, sparkColor }) {
  return (
    <div className="stat-cell">
      <div className="label">{label}</div>
      <div className="val">
        <span>{val}</span>
        {unit && <span className="unit">{unit}</span>}
      </div>
      {delta && <div className={`delta ${deltaDir === 'up' ? 'up' : deltaDir === 'dn' ? 'dn' : ''}`}>{delta}</div>}
      {sparkData && sparkData.length > 0 && (
        <div className="spark">
          <Spark data={sparkData} w={180} h={28} color={sparkColor} />
        </div>
      )}
    </div>
  );
}

function Section({ title, right, children }) {
  return (
    <div>
      <div className="hdr-row">
        <h2>{title}</h2>
        {right}
      </div>
      {children}
    </div>
  );
}

function EmptyState({ title, sub }) {
  return (
    <div className="empty">
      <div className="em-title">{title}</div>
      {sub && <div className="em-sub">{sub}</div>}
    </div>
  );
}

// Format helpers ----------------------------------------------------------

function fmtMs(ms) {
  if (ms == null) return '—';
  if (ms < 1000) return Math.round(ms) + 'ms';
  return (ms / 1000).toFixed(2) + 's';
}

function fmtCost(usd) {
  if (usd == null || usd === 0) return '$0.00';
  if (usd < 0.01) return '<$0.01';
  return '$' + usd.toFixed(usd < 1 ? 4 : 2);
}

function fmtTokens(n) {
  if (n == null) return '0';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return String(n);
}

function fmtTime(ts) {
  if (!ts) return '—';
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
}

function fmtRelative(ts) {
  if (!ts) return '—';
  const d = (Date.now() / 1000) - ts;
  if (d < 60) return Math.round(d) + 's ago';
  if (d < 3600) return Math.round(d / 60) + 'm ago';
  return Math.round(d / 3600) + 'h ago';
}

window.UI = {
  Dot, Pill, Spark, StatCell, Section, EmptyState,
  fmtMs, fmtCost, fmtTokens, fmtTime, fmtRelative,
};
