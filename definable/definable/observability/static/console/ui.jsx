// Shared atoms + fmt helpers. Stateless, pure.
// Cross-file access via window.UI.

const { IconChevronDown, IconEdit } = window.Icons;

function AccordionSection({ title, status, statusOn = true, defaultOpen = true, children }) {
  const [open, setOpen] = React.useState(defaultOpen);
  return (
    <div className="acc-section">
      <div className="acc-head" data-open={open} onClick={() => setOpen(!open)}>
        <IconChevronDown className="chev" size={14} />
        <span className="name">{title}</span>
        {status && (
          <span className={`pill ${statusOn ? '' : 'off'}`}>
            <span className="dotted" /> {status}
          </span>
        )}
      </div>
      {open && <div className="acc-body">{children}</div>}
    </div>
  );
}

function Toggle({ on, onChange }) {
  return (
    <button
      type="button"
      className={`toggle ${on ? 'on' : ''}`}
      onClick={onChange ? () => onChange(!on) : undefined}
      aria-pressed={on}
    />
  );
}

function Pill({ kind, children }) {
  return <span className={`status-pill ${kind || ''}`}>{children}</span>;
}

function EmptyState({ title, sub }) {
  return (
    <div className="empty-state">
      <div className="em-title">{title}</div>
      {sub && <div className="em-sub">{sub}</div>}
    </div>
  );
}

// Editable field affordance — read-only today, click pencil for "coming soon" hint.
// Each call carries a literal `apiHint` showing the future PATCH/POST so the lego
// concept is visible even before mutation endpoints land.
function EditableField({ value, apiHint, label, className }) {
  const [showHint, setShowHint] = React.useState(false);
  return (
    <span className={`editable ${className || ''}`}>
      {value}
      {apiHint && (
        <span style={{ position: 'relative' }}>
          <button
            type="button"
            className="edit-pencil"
            title={apiHint}
            onClick={(e) => { e.stopPropagation(); setShowHint((v) => !v); }}
          >
            <IconEdit size={12} />
          </button>
          {showHint && <span className="api-hint-tooltip">{apiHint}</span>}
        </span>
      )}
    </span>
  );
}

// Format helpers — keep pure, no I/O.
function fmtMs(ms) {
  if (ms == null) return '—';
  if (ms < 1000) return Math.round(ms) + 'ms';
  if (ms < 60_000) return (ms / 1000).toFixed(2) + 's';
  return Math.floor(ms / 60_000) + 'm ' + Math.round((ms % 60_000) / 1000) + 's';
}

function fmtCost(usd) {
  if (usd == null || usd === 0) return '$0.00';
  if (usd < 0.01) return '<$0.01';
  if (usd < 1) return '$' + usd.toFixed(4);
  return '$' + usd.toFixed(2);
}

function fmtTokens(n) {
  if (n == null) return '0';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return String(Math.round(n));
}

function fmtTime(ts) {
  if (!ts) return '—';
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
}

function fmtTimestamp(ts) {
  if (!ts) return '—';
  const d = new Date(ts * 1000);
  return d.toLocaleString([], { dateStyle: 'medium', timeStyle: 'medium' });
}

function fmtRelative(ts) {
  if (!ts) return '—';
  const d = (Date.now() / 1000) - ts;
  if (d < 5) return 'just now';
  if (d < 60) return Math.round(d) + 's ago';
  if (d < 3600) return Math.round(d / 60) + 'm ago';
  if (d < 86400) return Math.round(d / 3600) + 'h ago';
  return Math.round(d / 86400) + 'd ago';
}

function fmtInitials(name) {
  if (!name) return '·';
  const parts = String(name).split(/[\s_\-]+/).filter(Boolean);
  if (parts.length === 0) return '·';
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

window.UI = {
  AccordionSection, Toggle, Pill, EmptyState, EditableField,
  fmtMs, fmtCost, fmtTokens, fmtTime, fmtTimestamp, fmtRelative, fmtInitials,
};
