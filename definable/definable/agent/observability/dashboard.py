"""Observability dashboard — self-contained HTML frontend for agent monitoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.agent.observability.config import ObservabilityConfig


def get_dashboard_html(config: "ObservabilityConfig") -> str:
  """Return a self-contained HTML dashboard for agent observability.

  The HTML includes all CSS and JS inline. Dynamic data is fetched
  via JS fetch() calls to the /obs/api/* endpoints — no server-side
  templating is used (XSS-safe by design).

  Args:
    config: ObservabilityConfig with theme preference.

  Returns:
    Complete HTML document as a string.
  """
  theme = getattr(config, "theme", "dark")
  return f"""<!DOCTYPE html>
<html lang="en" data-theme="{theme}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Definable Observability</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
{_css()}
</head>
<body>
<div id="app" x-data="obsApp()" x-init="init()">
{_sidebar_html()}
<main class="main">
{_header_html()}
<div class="content">
{_overview_page()}
{_live_events_page()}
{_sessions_page()}
{_run_detail_page()}
{_compare_page()}
{_tools_page()}
{_models_page()}
</div>
</main>
</div>
{_alpine_app()}
</body>
</html>"""


def _css() -> str:
  return """<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root,[data-theme="dark"]{
  --bg:#0d0d0d;--surface:#1a1a1a;--surface-hover:#252525;
  --accent:#ff6b00;--accent-dim:rgba(255,107,0,.15);
  --text:#e0e0e0;--text-muted:#888;--border:#333;
  --success:#4ade80;--error:#f87171;--warning:#fbbf24;
  --info:#60a5fa;--purple:#c084fc;
  --font:'JetBrains Mono','Fira Code','SF Mono','Cascadia Code',monospace;
}
[data-theme="light"]{
  --bg:#fafafa;--surface:#ffffff;--surface-hover:#f0f0f0;
  --accent:#1a1a1a;--accent-dim:rgba(26,26,26,.08);
  --text:#1a1a1a;--text-muted:#666;--border:#e0e0e0;
  --success:#16a34a;--error:#dc2626;--warning:#d97706;
  --info:#2563eb;--purple:#9333ea;
}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--font);font-size:13px;line-height:1.5}
a{color:var(--accent);text-decoration:none}
#app{display:flex;height:100vh;overflow:hidden}

/* Sidebar */
.sidebar{width:220px;min-width:220px;background:var(--surface);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow-y:auto}
.sidebar-logo{padding:20px 16px;font-size:15px;font-weight:700;letter-spacing:1px;border-bottom:1px solid var(--border);white-space:nowrap}
.sidebar-logo span{color:var(--accent)}
.sidebar-nav{padding:8px 0;flex:1}
.nav-item{display:flex;align-items:center;gap:10px;padding:10px 16px;cursor:pointer;transition:background .15s;font-size:13px;color:var(--text-muted)}
.nav-item:hover{background:var(--surface-hover)}
.nav-item.active{background:var(--accent-dim);color:var(--text)}
.nav-dot{width:6px;height:6px;border-radius:50%;background:var(--border);flex-shrink:0}
.nav-item.active .nav-dot{background:var(--accent)}
.sidebar-footer{padding:12px 16px;border-top:1px solid var(--border);font-size:11px;color:var(--text-muted)}

/* Main */
.main{flex:1;display:flex;flex-direction:column;overflow:hidden}

/* Header */
.header{display:flex;align-items:center;justify-content:space-between;padding:12px 24px;border-bottom:1px solid var(--border);gap:16px;flex-shrink:0}
.header-left{display:flex;align-items:center;gap:16px;min-width:0;flex:1}
.breadcrumb{color:var(--text-muted);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.header-right{display:flex;align-items:center;gap:12px;flex-shrink:0}
.search-input{background:var(--surface);border:1px solid var(--border);color:var(--text);
  padding:6px 12px;border-radius:4px;font-family:var(--font);font-size:12px;width:200px}
.search-input::placeholder{color:var(--text-muted)}
.search-input:focus{outline:none;border-color:var(--accent)}
.theme-btn{background:var(--surface);border:1px solid var(--border);color:var(--text);
  padding:6px 10px;border-radius:4px;cursor:pointer;font-family:var(--font);font-size:12px}
.theme-btn:hover{background:var(--surface-hover)}
.agent-badge{display:flex;align-items:center;gap:6px;background:var(--surface);
  border:1px solid var(--border);padding:6px 12px;border-radius:4px;font-size:12px}
.status-dot{width:8px;height:8px;border-radius:50%;background:var(--success)}
.status-dot.error{background:var(--error)}

/* Content */
.content{flex:1;overflow-y:auto;padding:24px}

/* Page title */
.page-title{font-size:22px;font-weight:700;margin-bottom:4px}
.page-title span{color:var(--accent)}
.page-subtitle{color:var(--text-muted);font-size:12px;margin-bottom:24px}

/* Stat cards */
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:16px 20px}
.stat-label{font-size:12px;color:var(--text-muted);margin-bottom:8px}
.stat-label em{font-style:normal;color:var(--accent)}
.stat-value{font-size:32px;font-weight:700;line-height:1.1;margin-bottom:6px}
.stat-delta{font-size:11px;color:var(--success)}
.stat-delta.negative{color:var(--error)}

/* Section header */
.section-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.section-title{color:var(--text-muted);font-size:12px}
.section-actions{display:flex;gap:8px}

/* Split layout */
.split{display:grid;grid-template-columns:1fr 1fr;gap:24px}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:6px;overflow:hidden}
.panel-header{padding:12px 16px;border-bottom:1px solid var(--border);font-size:12px;
  color:var(--text-muted);display:flex;align-items:center;justify-content:space-between}

/* Run list */
.run-row{display:flex;align-items:center;gap:12px;padding:10px 16px;border-bottom:1px solid var(--border);cursor:pointer;transition:background .15s}
.run-row:hover{background:var(--surface-hover)}
.run-row:last-child{border-bottom:none}
.run-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.run-dot.completed{background:var(--success)}
.run-dot.error{background:var(--error)}
.run-dot.paused{background:var(--warning)}
.run-dot.cancelled{background:var(--text-muted)}
.run-dot.running{background:var(--info)}
.run-id{font-size:12px;font-weight:500;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.run-meta{font-size:11px;color:var(--text-muted);white-space:nowrap}
.badge{font-size:10px;font-weight:600;padding:2px 6px;border-radius:3px;letter-spacing:.5px}
.badge-completed{color:var(--success);border:1px solid var(--success)}
.badge-error{color:var(--error);border:1px solid var(--error)}
.badge-paused{color:var(--warning);border:1px solid var(--warning)}
.badge-cancelled{color:var(--text-muted);border:1px solid var(--text-muted)}
.badge-running{color:var(--info);border:1px solid var(--info)}
.badge-pending{color:var(--text-muted);border:1px solid var(--border)}

/* Chart (SVG bars) */
.chart-container{padding:16px;height:200px;display:flex;align-items:flex-end;gap:4px}
.chart-container svg{width:100%;height:100%}

/* Event feed */
.event-row{display:flex;align-items:flex-start;gap:10px;padding:8px 16px;border-bottom:1px solid var(--border);font-size:12px}
.event-time{color:var(--text-muted);white-space:nowrap;flex-shrink:0;width:80px}
.event-type{font-size:10px;font-weight:600;padding:2px 6px;border-radius:3px;flex-shrink:0;min-width:90px;text-align:center}
.event-type.lifecycle{background:rgba(74,222,128,.15);color:var(--success)}
.event-type.tool{background:rgba(255,107,0,.15);color:var(--accent)}
.event-type.model{background:rgba(96,165,250,.15);color:var(--info)}
.event-type.error{background:rgba(248,113,113,.15);color:var(--error)}
.event-type.guardrail{background:rgba(192,132,252,.15);color:var(--purple)}
.event-type.phase{background:rgba(251,191,36,.15);color:var(--warning)}
.event-type.memory{background:rgba(96,165,250,.15);color:var(--info)}
.event-type.knowledge{background:rgba(192,132,252,.15);color:var(--purple)}
.event-detail{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.event-run{color:var(--text-muted);flex-shrink:0;width:100px;overflow:hidden;text-overflow:ellipsis}

/* Filters */
.filter-bar{display:flex;gap:8px;padding:12px 16px;border-bottom:1px solid var(--border);flex-wrap:wrap;align-items:center}
.filter-bar select,.filter-bar input{background:var(--surface);border:1px solid var(--border);
  color:var(--text);padding:4px 8px;border-radius:4px;font-family:var(--font);font-size:11px}
.filter-bar select:focus,.filter-bar input:focus{outline:none;border-color:var(--accent)}
.filter-btn{background:var(--surface);border:1px solid var(--border);color:var(--text);
  padding:4px 10px;border-radius:4px;cursor:pointer;font-family:var(--font);font-size:11px}
.filter-btn:hover{background:var(--surface-hover)}
.filter-btn.active{border-color:var(--accent);color:var(--accent)}

/* Table */
.data-table{width:100%;border-collapse:collapse}
.data-table th{text-align:left;padding:10px 16px;font-size:11px;font-weight:600;
  color:var(--text-muted);border-bottom:1px solid var(--border);text-transform:uppercase;letter-spacing:.5px}
.data-table td{padding:10px 16px;border-bottom:1px solid var(--border);font-size:12px}
.data-table tr:hover td{background:var(--surface-hover)}
.data-table tr{cursor:pointer}

/* Run detail */
.detail-header{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:24px;align-items:center}
.detail-title{font-size:18px;font-weight:700}
.detail-meta{display:flex;gap:16px;font-size:12px;color:var(--text-muted)}
.detail-meta-item{display:flex;align-items:center;gap:4px}

/* Phase timeline */
.phase-timeline{display:flex;gap:2px;margin-bottom:24px;border-radius:4px;overflow:hidden;height:32px}
.phase-bar{display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:500;transition:flex .3s;min-width:2px;position:relative}
.phase-bar:hover{opacity:.85}
.phase-bar .phase-label{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:0 4px}

/* Tool calls list */
.tool-call{background:var(--surface);border:1px solid var(--border);border-radius:6px;margin-bottom:8px;overflow:hidden}
.tool-call-header{display:flex;align-items:center;gap:12px;padding:10px 16px;cursor:pointer}
.tool-call-header:hover{background:var(--surface-hover)}
.tool-call-name{font-weight:600;font-size:13px}
.tool-call-dur{font-size:11px;color:var(--text-muted);margin-left:auto}
.tool-call-body{padding:12px 16px;border-top:1px solid var(--border);font-size:11px;background:var(--bg)}
.tool-call-body pre{white-space:pre-wrap;word-break:break-all;max-height:200px;overflow-y:auto}

/* Collapsible */
.collapsible{margin-bottom:16px}
.collapsible-header{display:flex;align-items:center;gap:8px;padding:10px 16px;background:var(--surface);
  border:1px solid var(--border);border-radius:6px;cursor:pointer;font-size:13px;font-weight:500}
.collapsible-header:hover{background:var(--surface-hover)}
.collapsible-body{border:1px solid var(--border);border-top:none;border-radius:0 0 6px 6px;padding:12px 16px;background:var(--bg);font-size:12px}
.collapsible-body pre{white-space:pre-wrap;word-break:break-all;max-height:300px;overflow-y:auto}

/* Compare */
.compare-select{display:flex;gap:16px;margin-bottom:24px;align-items:center}
.compare-select select{background:var(--surface);border:1px solid var(--border);color:var(--text);
  padding:8px 12px;border-radius:4px;font-family:var(--font);font-size:12px;flex:1;max-width:300px}
.compare-vs{color:var(--text-muted);font-weight:600}
.diff-block{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:16px;margin-bottom:16px;font-size:12px}
.diff-block pre{white-space:pre-wrap;word-break:break-all}
.diff-add{color:var(--success)}
.diff-del{color:var(--error)}
.diff-meta{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:16px}
.diff-meta-card{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:12px 16px;text-align:center}
.diff-meta-label{font-size:11px;color:var(--text-muted);margin-bottom:4px}
.diff-meta-value{font-size:18px;font-weight:700}
.diff-meta-value.positive{color:var(--error)}
.diff-meta-value.negative{color:var(--success)}

/* Empty state */
.empty-state{text-align:center;padding:48px 24px;color:var(--text-muted)}
.empty-state-icon{font-size:32px;margin-bottom:12px;opacity:.5}
.empty-state-text{font-size:14px}

/* Loading */
.loading{display:flex;align-items:center;justify-content:center;padding:24px;color:var(--text-muted);gap:8px}
.spinner{width:16px;height:16px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* Back button */
.back-btn{display:inline-flex;align-items:center;gap:4px;color:var(--text-muted);font-size:12px;cursor:pointer;margin-bottom:16px;padding:4px 0}
.back-btn:hover{color:var(--text)}

/* Responsive */
@media(max-width:768px){
  .sidebar{position:fixed;left:-220px;top:0;bottom:0;z-index:100;transition:left .2s}
  .sidebar.open{left:0}
  .stat-grid{grid-template-columns:repeat(2,1fr)}
  .split{grid-template-columns:1fr}
  .diff-meta{grid-template-columns:1fr}
  .phase-timeline{height:24px}
  .phase-bar .phase-label{font-size:8px}
}
</style>"""


def _sidebar_html() -> str:
  return """<aside class="sidebar" :class="{'open': sidebarOpen}">
  <div class="sidebar-logo"><span>&gt;</span> DEFINABLE_OBS <span>|</span></div>
  <nav class="sidebar-nav">
    <div class="nav-item" :class="{'active': page==='overview'}" @click="navigate('overview')">
      <div class="nav-dot"></div><span>overview</span>
    </div>
    <div class="nav-item" :class="{'active': page==='live_events'}" @click="navigate('live_events')">
      <div class="nav-dot"></div><span>live_events</span>
    </div>
    <div class="nav-item" :class="{'active': page==='sessions'}" @click="navigate('sessions')">
      <div class="nav-dot"></div><span>sessions</span>
    </div>
    <div class="nav-item" :class="{'active': page==='tools'}" @click="navigate('tools')">
      <div class="nav-dot"></div><span>tools</span>
    </div>
    <div class="nav-item" :class="{'active': page==='models'}" @click="navigate('models')">
      <div class="nav-dot"></div><span>models</span>
    </div>
    <div class="nav-item" :class="{'active': page==='compare'}" @click="navigate('compare')">
      <div class="nav-dot"></div><span>compare</span>
    </div>
  </nav>
  <div class="sidebar-footer">definable observability v1</div>
</aside>"""


def _header_html() -> str:
  return """<header class="header">
  <div class="header-left">
    <span class="breadcrumb" x-text="'~/agent/' + agentName + '/' + page"></span>
  </div>
  <div class="header-right">
    <input class="search-input" type="text" placeholder="/ search..." x-model="searchQuery" @keyup.enter="applySearch()">
    <button class="theme-btn" @click="toggleTheme()" x-text="theme === 'dark' ? '&#9788;' : '&#9790;'"></button>
    <div class="agent-badge">
      <div class="status-dot" :class="{'error': !connected}"></div>
      <span x-text="agentName"></span>
    </div>
  </div>
</header>"""


def _overview_page() -> str:
  return """<div x-show="page==='overview'" x-cloak>
  <div class="page-title"><span>&gt;</span> OVERVIEW<span>|</span></div>
  <div class="page-subtitle"># agent runtime metrics and recent activity</div>

  <div class="stat-grid">
    <div class="stat-card">
      <div class="stat-label"><em>const</em> total_runs =</div>
      <div class="stat-value" x-text="metrics.total_runs || 0"></div>
      <div class="stat-delta" x-text="'++ ' + (metrics.recent_runs || 0) + ' recent'"></div>
    </div>
    <div class="stat-card">
      <div class="stat-label"><em>const</em> success_rate =</div>
      <div class="stat-value" x-text="formatPercent(metrics.success_rate)"></div>
      <div class="stat-delta" x-text="'++ ' + formatPercent(metrics.success_rate) + ' overall'"></div>
    </div>
    <div class="stat-card">
      <div class="stat-label"><em>const</em> total_tokens =</div>
      <div class="stat-value" x-text="formatNumber(metrics.total_tokens)"></div>
      <div class="stat-delta" x-text="'++ ' + formatNumber(metrics.avg_tokens_per_run || 0) + ' avg/run'"></div>
    </div>
    <div class="stat-card">
      <div class="stat-label"><em>const</em> total_cost =</div>
      <div class="stat-value" x-text="'$' + formatCost(metrics.total_cost)"></div>
      <div class="stat-delta" x-text="'++ $' + formatCost(metrics.avg_cost_per_run || 0) + ' avg/run'"></div>
    </div>
  </div>

  <div class="split">
    <div class="panel">
      <div class="panel-header">
        <span>// recent_runs</span>
        <span x-text="(recentRuns || []).length + ' runs'"></span>
      </div>
      <template x-if="!recentRuns || recentRuns.length === 0">
        <div class="empty-state">
          <div class="empty-state-icon">&gt;_</div>
          <div class="empty-state-text">no runs recorded yet</div>
        </div>
      </template>
      <template x-for="run in (recentRuns || []).slice(0, 10)" :key="run.run_id">
        <div class="run-row" @click="viewRun(run.run_id, run.session_id)">
          <div class="run-dot" :class="run.status?.toLowerCase()"></div>
          <div class="run-id" x-text="run.run_id"></div>
          <span class="badge" :class="'badge-' + (run.status?.toLowerCase() || 'pending')"
                x-text="'[' + (run.status || 'PENDING') + ']'"></span>
          <span class="run-meta" x-text="formatDuration(run.duration)"></span>
          <span class="run-meta" x-text="formatNumber(run.tokens || 0) + ' tok'"></span>
        </div>
      </template>
    </div>
    <div class="panel">
      <div class="panel-header">
        <span>// run_cadence</span>
        <span>$ ls -la</span>
      </div>
      <div class="chart-container" x-html="renderBarChart(metrics.timeline || [])"></div>
    </div>
  </div>
</div>"""


def _live_events_page() -> str:
  return """<div x-show="page==='live_events'" x-cloak>
  <div class="page-title"><span>&gt;</span> LIVE EVENTS<span>|</span></div>
  <div class="page-subtitle"># real-time event stream from the agent pipeline</div>

  <div class="panel">
    <div class="filter-bar">
      <select x-model="eventFilter.type">
        <option value="">all events</option>
        <option value="lifecycle">lifecycle</option>
        <option value="tool">tool</option>
        <option value="model">model</option>
        <option value="error">error</option>
        <option value="guardrail">guardrail</option>
        <option value="phase">phase</option>
        <option value="memory">memory</option>
        <option value="knowledge">knowledge</option>
      </select>
      <input type="text" placeholder="run_id..." x-model="eventFilter.run_id" style="width:120px">
      <input type="text" placeholder="session_id..." x-model="eventFilter.session_id" style="width:120px">
      <button class="filter-btn" :class="{'active': !ssePaused}" @click="ssePaused = !ssePaused"
              x-text="ssePaused ? '&#9654; resume' : '&#10074;&#10074; pause'"></button>
      <button class="filter-btn" @click="liveEvents = []">clear</button>
      <span style="margin-left:auto;font-size:11px;color:var(--text-muted)"
            x-text="filteredEvents().length + ' events'"></span>
    </div>
    <div style="max-height:calc(100vh - 280px);overflow-y:auto">
      <template x-if="filteredEvents().length === 0">
        <div class="empty-state">
          <div class="empty-state-icon">&gt;_</div>
          <div class="empty-state-text">waiting for events...</div>
        </div>
      </template>
      <template x-for="evt in filteredEvents().slice(-500)" :key="evt._idx">
        <div class="event-row">
          <span class="event-time" x-text="formatTime(evt.created_at)"></span>
          <span class="event-type" :class="eventCategory(evt.event)" x-text="evt.event"></span>
          <span class="event-run" x-text="evt.run_id ? evt.run_id.substring(0,12) + '...' : '-'"></span>
          <span class="event-detail" x-text="eventSummary(evt)"></span>
        </div>
      </template>
    </div>
  </div>
</div>"""


def _sessions_page() -> str:
  return """<div x-show="page==='sessions'" x-cloak>
  <div class="page-title"><span>&gt;</span> SESSIONS<span>|</span></div>
  <div class="page-subtitle"># browse historical trace sessions and runs</div>

  <template x-if="!selectedSession">
    <div>
      <template x-if="loadingSessions">
        <div class="loading"><div class="spinner"></div> loading sessions...</div>
      </template>
      <template x-if="!loadingSessions && sessions.length === 0">
        <div class="empty-state">
          <div class="empty-state-icon">&gt;_</div>
          <div class="empty-state-text">no trace sessions found</div>
        </div>
      </template>
      <template x-if="!loadingSessions && sessions.length > 0">
        <div class="panel">
          <table class="data-table">
            <thead><tr>
              <th>session_id</th><th>runs</th><th>size</th><th>modified</th>
            </tr></thead>
            <tbody>
              <template x-for="s in sessions" :key="s.session_id">
                <tr @click="selectSession(s.session_id)">
                  <td x-text="s.session_id"></td>
                  <td x-text="s.run_count || '-'"></td>
                  <td x-text="formatBytes(s.file_size || 0)"></td>
                  <td x-text="formatTimestamp(s.modified_at)"></td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>
      </template>
    </div>
  </template>

  <template x-if="selectedSession && !selectedRun">
    <div>
      <div class="back-btn" @click="selectedSession=null;sessionRuns=[]">&larr; back to sessions</div>
      <div class="page-subtitle" x-text="'session: ' + selectedSession"></div>
      <template x-if="loadingRuns">
        <div class="loading"><div class="spinner"></div> loading runs...</div>
      </template>
      <template x-if="!loadingRuns && sessionRuns.length === 0">
        <div class="empty-state">
          <div class="empty-state-icon">&gt;_</div>
          <div class="empty-state-text">no runs in this session</div>
        </div>
      </template>
      <template x-if="!loadingRuns && sessionRuns.length > 0">
        <div class="panel">
          <table class="data-table">
            <thead><tr>
              <th>run_id</th><th>status</th><th>duration</th><th>tokens</th><th>cost</th>
            </tr></thead>
            <tbody>
              <template x-for="r in sessionRuns" :key="r.run_id">
                <tr @click="viewRun(r.run_id, selectedSession)">
                  <td x-text="r.run_id"></td>
                  <td><span class="badge" :class="'badge-' + (r.status?.toLowerCase() || 'pending')"
                        x-text="'[' + (r.status || '?') + ']'"></span></td>
                  <td x-text="formatDuration(r.duration)"></td>
                  <td x-text="formatNumber(r.tokens || 0)"></td>
                  <td x-text="'$' + formatCost(r.cost || 0)"></td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>
      </template>
    </div>
  </template>
</div>"""


def _run_detail_page() -> str:
  return """<div x-show="page==='run_detail'" x-cloak>
  <div class="back-btn" @click="page=selectedSession?'sessions':'overview';selectedRun=null">&larr; back</div>
  <template x-if="loadingRunDetail">
    <div class="loading"><div class="spinner"></div> loading run detail...</div>
  </template>
  <template x-if="!loadingRunDetail && runDetail">
    <div>
      <div class="detail-header">
        <div class="detail-title" x-text="runDetail.run_id"></div>
        <span class="badge" :class="'badge-' + (runDetail.status?.toLowerCase() || 'pending')"
              x-text="'[' + (runDetail.status || '?') + ']'"></span>
      </div>
      <div class="detail-meta" style="margin-bottom:24px">
        <div class="detail-meta-item">model: <strong x-text="runDetail.model || '-'"></strong></div>
        <div class="detail-meta-item">agent: <strong x-text="runDetail.agent_name || '-'"></strong></div>
        <div class="detail-meta-item">duration: <strong x-text="formatDuration(runDetail.duration)"></strong></div>
        <div class="detail-meta-item">tokens: <strong x-text="formatNumber((runDetail.tokens?.total_tokens) || 0)"></strong></div>
        <div class="detail-meta-item">cost: <strong x-text="'$' + formatCost(runDetail.cost || 0)"></strong></div>
      </div>

      <!-- Phase timeline -->
      <template x-if="runDetail.steps && runDetail.steps.length > 0">
        <div>
          <div class="section-header"><span class="section-title">// phase_timeline</span></div>
          <div class="phase-timeline">
            <template x-for="step in runDetail.steps" :key="step.name + step.started_at">
              <div class="phase-bar"
                   :style="'flex:' + Math.max(1, step.duration_ms || 1) + ';background:' + phaseColor(step.step_type)"
                   :title="step.name + ': ' + formatMs(step.duration_ms)">
                <span class="phase-label" x-text="step.name"></span>
              </div>
            </template>
          </div>
        </div>
      </template>

      <!-- Tool calls -->
      <template x-if="runDetail.tool_calls && runDetail.tool_calls.length > 0">
        <div style="margin-top:24px">
          <div class="section-header">
            <span class="section-title">// tool_calls</span>
            <span style="font-size:11px;color:var(--text-muted)" x-text="runDetail.tool_calls.length + ' calls'"></span>
          </div>
          <template x-for="(tc, i) in runDetail.tool_calls" :key="i">
            <div class="tool-call" x-data="{open:false}">
              <div class="tool-call-header" @click="open=!open">
                <span x-text="open ? '&#9660;' : '&#9654;'" style="font-size:10px;color:var(--text-muted)"></span>
                <span class="tool-call-name" x-text="tc.tool_name"></span>
                <template x-if="tc.error">
                  <span class="badge badge-error">[ERROR]</span>
                </template>
                <span class="tool-call-dur" x-text="formatMs(tc.duration_ms)"></span>
              </div>
              <div class="tool-call-body" x-show="open" x-cloak>
                <div style="margin-bottom:8px;color:var(--text-muted)">args:</div>
                <pre x-text="JSON.stringify(tc.tool_args, null, 2) || '-'"></pre>
                <div style="margin:8px 0;color:var(--text-muted)">result:</div>
                <pre x-text="truncate(typeof tc.result === 'string' ? tc.result : JSON.stringify(tc.result, null, 2) || '-', 1000)"></pre>
                <template x-if="tc.error">
                  <div>
                    <div style="margin:8px 0;color:var(--error)">error:</div>
                    <pre style="color:var(--error)" x-text="tc.error"></pre>
                  </div>
                </template>
              </div>
            </div>
          </template>
        </div>
      </template>

      <!-- Input -->
      <div class="collapsible" style="margin-top:24px" x-data="{open:false}">
        <div class="collapsible-header" @click="open=!open">
          <span x-text="open ? '&#9660;' : '&#9654;'" style="font-size:10px;color:var(--text-muted)"></span>
          // input
        </div>
        <div class="collapsible-body" x-show="open" x-cloak>
          <pre x-text="JSON.stringify(runDetail.input, null, 2) || 'no input recorded'"></pre>
        </div>
      </div>

      <!-- Output -->
      <div class="collapsible" x-data="{open:false}">
        <div class="collapsible-header" @click="open=!open">
          <span x-text="open ? '&#9660;' : '&#9654;'" style="font-size:10px;color:var(--text-muted)"></span>
          // output
        </div>
        <div class="collapsible-body" x-show="open" x-cloak>
          <pre x-text="typeof runDetail.content === 'string'
            ? runDetail.content
            : JSON.stringify(runDetail.content, null, 2) || 'no output'"></pre>
        </div>
      </div>
    </div>
  </template>
</div>"""


def _compare_page() -> str:
  return """<div x-show="page==='compare'" x-cloak>
  <div class="page-title"><span>&gt;</span> COMPARE<span>|</span></div>
  <div class="page-subtitle"># side-by-side run comparison</div>

  <div class="compare-select">
    <select x-model="compareA" @change="loadComparison()">
      <option value="">select run A</option>
      <template x-for="r in allRunIds" :key="r"><option :value="r" x-text="r"></option></template>
    </select>
    <span class="compare-vs">vs</span>
    <select x-model="compareB" @change="loadComparison()">
      <option value="">select run B</option>
      <template x-for="r in allRunIds" :key="r"><option :value="r" x-text="r"></option></template>
    </select>
  </div>

  <template x-if="loadingCompare">
    <div class="loading"><div class="spinner"></div> comparing runs...</div>
  </template>

  <template x-if="!loadingCompare && compareResult">
    <div>
      <div class="diff-meta">
        <div class="diff-meta-card">
          <div class="diff-meta-label">token delta</div>
          <div class="diff-meta-value" :class="compareResult.token_diff > 0 ? 'positive' : (compareResult.token_diff < 0 ? 'negative' : '')"
               x-text="(compareResult.token_diff > 0 ? '+' : '') + formatNumber(compareResult.token_diff || 0)"></div>
        </div>
        <div class="diff-meta-card">
          <div class="diff-meta-label">cost delta</div>
          <div class="diff-meta-value" :class="compareResult.cost_diff > 0 ? 'positive' : (compareResult.cost_diff < 0 ? 'negative' : '')"
               x-text="(compareResult.cost_diff > 0 ? '+$' : '-$') + formatCost(Math.abs(compareResult.cost_diff || 0))"></div>
        </div>
        <div class="diff-meta-card">
          <div class="diff-meta-label">duration delta</div>
          <div class="diff-meta-value" :class="compareResult.duration_diff > 0 ? 'positive' : (compareResult.duration_diff < 0 ? 'negative' : '')"
               x-text="(compareResult.duration_diff > 0 ? '+' : '') + formatDuration(compareResult.duration_diff)"></div>
        </div>
      </div>

      <template x-if="compareResult.content_diff">
        <div>
          <div class="section-header"><span class="section-title">// content_diff</span></div>
          <div class="diff-block"><pre x-html="renderDiff(compareResult.content_diff)"></pre></div>
        </div>
      </template>
      <template x-if="!compareResult.content_diff && compareA && compareB">
        <div class="diff-block" style="text-align:center;color:var(--text-muted)">outputs are identical</div>
      </template>

      <template x-if="compareResult.tool_calls_diff">
        <div>
          <div class="section-header"><span class="section-title">// tool_calls_diff</span></div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">
            <div class="diff-block">
              <div style="color:var(--success);margin-bottom:8px;font-weight:600">added
                (<span x-text="(compareResult.tool_calls_diff.added||[]).length"></span>)</div>
              <template x-for="tc in compareResult.tool_calls_diff.added||[]" :key="tc.tool_name">
                <div style="padding:2px 0" x-text="'+ ' + tc.tool_name"></div>
              </template>
              <template x-if="!(compareResult.tool_calls_diff.added||[]).length">
                <div style="color:var(--text-muted)">none</div>
              </template>
            </div>
            <div class="diff-block">
              <div style="color:var(--error);margin-bottom:8px;font-weight:600">removed
                (<span x-text="(compareResult.tool_calls_diff.removed||[]).length"></span>)</div>
              <template x-for="tc in compareResult.tool_calls_diff.removed||[]" :key="tc.tool_name">
                <div style="padding:2px 0" x-text="'- ' + tc.tool_name"></div>
              </template>
              <template x-if="!(compareResult.tool_calls_diff.removed||[]).length">
                <div style="color:var(--text-muted)">none</div>
              </template>
            </div>
            <div class="diff-block">
              <div style="color:var(--text-muted);margin-bottom:8px;font-weight:600">common
                (<span x-text="compareResult.tool_calls_diff.common || 0"></span>)</div>
            </div>
          </div>
        </div>
      </template>
    </div>
  </template>

  <template x-if="!loadingCompare && !compareResult && compareA && compareB">
    <div class="empty-state">
      <div class="empty-state-icon">&gt;_</div>
      <div class="empty-state-text">select two runs to compare</div>
    </div>
  </template>
</div>"""


def _tools_page() -> str:
  return """<div x-show="page==='tools'" x-cloak>
  <div class="page-title"><span>&gt;</span> TOOLS<span>|</span></div>
  <div class="page-subtitle"># tool call analytics — count, errors, latency</div>

  <template x-if="!metrics.tool_stats || Object.keys(metrics.tool_stats || {}).length === 0">
    <div class="empty-state">
      <div class="empty-state-icon">&gt;_</div>
      <div class="empty-state-text">no tool call data yet</div>
    </div>
  </template>
  <template x-if="metrics.tool_stats && Object.keys(metrics.tool_stats).length > 0">
    <div class="panel">
      <table class="data-table">
        <thead><tr>
          <th>tool_name</th><th>calls</th><th>errors</th><th>avg latency</th><th>error rate</th>
        </tr></thead>
        <tbody>
          <template x-for="[name, stats] in Object.entries(metrics.tool_stats)" :key="name">
            <tr>
              <td><strong x-text="name"></strong></td>
              <td x-text="stats.count || 0"></td>
              <td :style="(stats.error_count || 0) > 0 ? 'color:var(--error)' : ''" x-text="stats.error_count || 0"></td>
              <td x-text="formatMs(stats.avg_latency_ms)"></td>
              <td :style="(stats.error_count || 0) > 0 ? 'color:var(--error)' : ''"
                  x-text="stats.count ? formatPercent((stats.error_count || 0) / stats.count * 100) : '0%'"></td>
            </tr>
          </template>
        </tbody>
      </table>
    </div>
  </template>
</div>"""


def _models_page() -> str:
  return """<div x-show="page==='models'" x-cloak>
  <div class="page-title"><span>&gt;</span> MODELS<span>|</span></div>
  <div class="page-subtitle"># model call analytics — tokens, cost, latency</div>

  <template x-if="!metrics.model_stats || Object.keys(metrics.model_stats || {}).length === 0">
    <div class="empty-state">
      <div class="empty-state-icon">&gt;_</div>
      <div class="empty-state-text">no model call data yet</div>
    </div>
  </template>
  <template x-if="metrics.model_stats && Object.keys(metrics.model_stats).length > 0">
    <div class="panel">
      <table class="data-table">
        <thead><tr>
          <th>model_id</th><th>calls</th><th>avg tokens</th><th>total cost</th>
        </tr></thead>
        <tbody>
          <template x-for="[name, stats] in Object.entries(metrics.model_stats)" :key="name">
            <tr>
              <td><strong x-text="name"></strong></td>
              <td x-text="stats.count || 0"></td>
              <td x-text="formatNumber(stats.avg_tokens || 0)"></td>
              <td x-text="'$' + formatCost(stats.total_cost || 0)"></td>
            </tr>
          </template>
        </tbody>
      </table>
    </div>
  </template>
</div>"""


def _alpine_app() -> str:
  return """<script>
function obsApp() {
  return {
    // State
    page: 'overview',
    theme: localStorage.getItem('definable-obs-theme') || document.documentElement.getAttribute('data-theme') || 'dark',
    sidebarOpen: false,
    searchQuery: '',
    agentName: 'agent',
    connected: false,

    // Metrics
    metrics: {},
    recentRuns: [],

    // Live events
    liveEvents: [],
    eventCounter: 0,
    ssePaused: false,
    sseSource: null,
    eventFilter: { type: '', run_id: '', session_id: '' },

    // Sessions
    sessions: [],
    loadingSessions: false,
    selectedSession: null,
    sessionRuns: [],
    loadingRuns: false,

    // Run detail
    selectedRun: null,
    runDetail: null,
    loadingRunDetail: false,

    // Compare
    compareA: '',
    compareB: '',
    compareResult: null,
    loadingCompare: false,
    allRunIds: [],

    // Init
    async init() {
      document.documentElement.setAttribute('data-theme', this.theme);
      await this.fetchMetrics();
      await this.fetchSessions();
      this.connectSSE();
      this.startPolling();
    },

    // Navigation
    navigate(p) {
      this.page = p;
      this.sidebarOpen = false;
      if (p === 'sessions') this.fetchSessions();
      if (p === 'compare') this.fetchAllRunIds();
      if (p === 'overview') this.fetchMetrics();
    },

    // Theme
    toggleTheme() {
      this.theme = this.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', this.theme);
      localStorage.setItem('definable-obs-theme', this.theme);
    },

    // Fetch helpers
    async fetchJSON(url) {
      try {
        const resp = await fetch(url);
        if (!resp.ok) return null;
        return await resp.json();
      } catch { return null; }
    },

    // Metrics
    async fetchMetrics() {
      const data = await this.fetchJSON('/obs/api/metrics');
      if (data) {
        this.metrics = data;
        this.agentName = data.agent_name || 'agent';
      }
    },

    // Sessions
    async fetchSessions() {
      this.loadingSessions = true;
      const data = await this.fetchJSON('/obs/api/sessions');
      this.sessions = data || [];
      this.loadingSessions = false;
    },

    async selectSession(sid) {
      this.selectedSession = sid;
      this.loadingRuns = true;
      const data = await this.fetchJSON('/obs/api/sessions/' + encodeURIComponent(sid));
      this.sessionRuns = data || [];
      this.loadingRuns = false;
    },

    // Run detail
    async viewRun(runId, sessionId) {
      this.selectedRun = runId;
      this.selectedSession = sessionId || this.selectedSession;
      this.page = 'run_detail';
      this.loadingRunDetail = true;
      const data = await this.fetchJSON('/obs/api/runs/' + encodeURIComponent(runId));
      this.runDetail = data;
      this.loadingRunDetail = false;
    },

    // Compare
    async fetchAllRunIds() {
      const data = await this.fetchJSON('/obs/api/sessions');
      if (!data) return;
      const ids = [];
      for (const s of data) {
        const runs = await this.fetchJSON('/obs/api/sessions/' + encodeURIComponent(s.session_id));
        if (runs) {
          for (const r of runs) ids.push(r.run_id);
        }
      }
      this.allRunIds = ids;
    },

    async loadComparison() {
      if (!this.compareA || !this.compareB) { this.compareResult = null; return; }
      this.loadingCompare = true;
      const data = await this.fetchJSON('/obs/api/compare?a=' + encodeURIComponent(this.compareA) + '&b=' + encodeURIComponent(this.compareB));
      this.compareResult = data;
      this.loadingCompare = false;
    },

    // SSE
    connectSSE() {
      if (this.sseSource) { this.sseSource.close(); }
      try {
        this.sseSource = new EventSource('/obs/api/events');
        this.sseSource.onopen = () => { this.connected = true; };
        this.sseSource.onmessage = (e) => {
          if (this.ssePaused) return;
          try {
            const evt = JSON.parse(e.data);
            evt._idx = ++this.eventCounter;
            this.liveEvents.push(evt);
            if (this.liveEvents.length > 2000) {
              this.liveEvents = this.liveEvents.slice(-1000);
            }
          } catch {}
        };
        this.sseSource.onerror = () => { this.connected = false; };
      } catch { this.connected = false; }
    },

    // Polling
    startPolling() {
      setInterval(() => {
        if (this.page === 'overview') this.fetchMetrics();
      }, 10000);
    },

    // Filters
    filteredEvents() {
      return this.liveEvents.filter(evt => {
        if (this.eventFilter.type && this.eventCategory(evt.event) !== this.eventFilter.type) return false;
        if (this.eventFilter.run_id && !(evt.run_id || '').includes(this.eventFilter.run_id)) return false;
        if (this.eventFilter.session_id && !(evt.session_id || '').includes(this.eventFilter.session_id)) return false;
        return true;
      });
    },

    applySearch() {
      // Global search — filter events or navigate
      const q = this.searchQuery.trim().toLowerCase();
      if (!q) return;
      // Try to find as run_id
      if (q.length > 6) {
        this.viewRun(q, null);
      }
    },

    // Event categorization
    eventCategory(eventName) {
      if (!eventName) return 'lifecycle';
      const n = eventName.toLowerCase();
      if (n.includes('error') || n.includes('cancelled')) return 'error';
      if (n.includes('tool')) return 'tool';
      if (n.includes('model')) return 'model';
      if (n.includes('guardrail') || n.includes('guard')) return 'guardrail';
      if (n.includes('phase')) return 'phase';
      if (n.includes('memory') || n.includes('session_summary')) return 'memory';
      if (n.includes('knowledge')) return 'knowledge';
      return 'lifecycle';
    },

    eventSummary(evt) {
      if (evt.content) return typeof evt.content === 'string' ? evt.content.substring(0, 120) : JSON.stringify(evt.content).substring(0, 120);
      if (evt.tool) return (evt.tool.tool_name || '') + '(' + JSON.stringify(evt.tool.tool_args || {}).substring(0, 60) + ')';
      if (evt.agent_name) return evt.agent_name;
      if (evt.phase_name) return 'phase: ' + evt.phase_name;
      return '';
    },

    // Formatters
    formatNumber(n) {
      if (n == null) return '0';
      if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
      if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
      return String(Math.round(n));
    },

    formatPercent(n) {
      if (n == null) return '0%';
      return Number(n).toFixed(1) + '%';
    },

    formatCost(n) {
      if (n == null) return '0.00';
      return Number(n).toFixed(4);
    },

    formatDuration(sec) {
      if (sec == null) return '-';
      if (sec < 0.001) return '< 1ms';
      if (sec < 1) return Math.round(sec * 1000) + 'ms';
      if (sec < 60) return sec.toFixed(2) + 's';
      return Math.floor(sec / 60) + 'm ' + Math.round(sec % 60) + 's';
    },

    formatMs(ms) {
      if (ms == null) return '-';
      if (ms < 1) return '< 1ms';
      if (ms < 1000) return Math.round(ms) + 'ms';
      return (ms / 1000).toFixed(2) + 's';
    },

    formatTime(ts) {
      if (!ts) return '-';
      const d = new Date(typeof ts === 'number' ? (ts < 1e12 ? ts * 1000 : ts) : ts);
      return d.toLocaleTimeString('en-US', {hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'});
    },

    formatTimestamp(ts) {
      if (!ts) return '-';
      const d = new Date(typeof ts === 'number' ? (ts < 1e12 ? ts * 1000 : ts) : ts);
      return d.toLocaleDateString() + ' ' + d.toLocaleTimeString('en-US', {hour12: false});
    },

    formatBytes(b) {
      if (b < 1024) return b + ' B';
      if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KB';
      return (b / (1024 * 1024)).toFixed(1) + ' MB';
    },

    truncate(str, maxLen) {
      if (!str) return '-';
      if (str.length <= maxLen) return str;
      return str.substring(0, maxLen) + '...';
    },

    // Phase colors
    phaseColor(stepType) {
      const colors = {
        model_call: 'var(--info)',
        tool_call: 'var(--accent)',
        knowledge_retrieval: 'var(--purple)',
        memory_recall: 'var(--warning)',
      };
      return colors[stepType] || 'var(--border)';
    },

    // Bar chart renderer (inline SVG)
    renderBarChart(timeline) {
      if (!timeline || timeline.length === 0) {
        return '<svg viewBox="0 0 400 160"><text x="200" y="80" fill="var(--text-muted)" '
          + 'text-anchor="middle" font-size="12">no data</text></svg>';
      }
      const maxVal = Math.max(...timeline.map(t => t.count || 0), 1);
      const barW = Math.max(8, Math.floor(380 / timeline.length) - 4);
      let bars = '';
      timeline.forEach((t, i) => {
        const h = Math.max(2, ((t.count || 0) / maxVal) * 130);
        const x = i * (barW + 4) + 10;
        const y = 140 - h;
        const fill = (t.count || 0) > 0 ? 'var(--accent)' : 'var(--border)';
        bars += '<rect x="' + x + '" y="' + y + '" width="' + barW
          + '" height="' + h + '" rx="2" fill="' + fill
          + '" opacity="0.8"><title>' + (t.label||'') + ': '
          + (t.count||0) + '</title></rect>';
        if (timeline.length <= 24) {
          bars += '<text x="' + (x + barW/2) + '" y="155" fill="var(--text-muted)"'
            + ' text-anchor="middle" font-size="9">' + (t.label||'') + '</text>';
        }
      });
      return '<svg viewBox="0 0 ' + (timeline.length * (barW + 4) + 20) + ' 160" preserveAspectRatio="none">' + bars + '</svg>';
    },

    // Diff renderer
    renderDiff(diff) {
      if (!diff) return '';
      return diff.split('\\n').map(line => {
        if (line.startsWith('+') && !line.startsWith('+++')) return '<span class="diff-add">' + this.escapeHtml(line) + '</span>';
        if (line.startsWith('-') && !line.startsWith('---')) return '<span class="diff-del">' + this.escapeHtml(line) + '</span>';
        return this.escapeHtml(line);
      }).join('\\n');
    },

    escapeHtml(str) {
      return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    },
  };
}
</script>"""
