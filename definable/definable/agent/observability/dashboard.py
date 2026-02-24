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
{_chat_widget()}
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

/* Chart */
.chart-wrap{position:relative}
.chart-range-bar{display:flex;gap:4px;padding:0 16px 4px}
.chart-range-btn{background:none;border:1px solid var(--border);color:var(--text-muted);
  padding:2px 8px;border-radius:3px;font-family:var(--font);font-size:10px;cursor:pointer}
.chart-range-btn:hover{color:var(--text);border-color:var(--text-muted)}
.chart-range-btn.active{background:var(--accent);color:var(--bg);border-color:var(--accent)}
.chart-container{padding:8px 16px 12px;height:180px;display:flex;align-items:center;justify-content:center}
.chart-container svg{width:100%;height:100%;overflow:visible}
.chart-container svg rect.bar{cursor:pointer;transition:opacity .1s}
.chart-container svg rect.bar:hover{opacity:1 !important}
.chart-tooltip{position:absolute;z-index:50;pointer-events:none;background:var(--surface);
  border:1px solid var(--border);border-radius:6px;padding:8px 12px;font-size:11px;
  line-height:1.6;white-space:nowrap;box-shadow:0 4px 12px rgba(0,0,0,.3);opacity:0;transition:opacity .12s}
.chart-tooltip.visible{opacity:1}
.chart-tooltip-time{font-weight:600;color:var(--text);margin-bottom:2px}
.chart-tooltip-row{color:var(--text-muted);display:flex;gap:8px;justify-content:space-between}
.chart-tooltip-row span:last-child{color:var(--text);font-weight:500}

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
.compare-bar{display:flex;gap:12px;margin-bottom:24px;align-items:flex-end;flex-wrap:wrap}
.compare-field{display:flex;flex-direction:column;gap:4px;flex:1;min-width:200px;max-width:360px}
.compare-field label{font-size:10px;color:var(--text-muted);font-weight:600;text-transform:uppercase;letter-spacing:.5px}
.compare-field select{background:var(--surface);border:1px solid var(--border);color:var(--text);
  padding:8px 12px;border-radius:4px;font-family:var(--font);font-size:12px;width:100%}
.compare-vs{color:var(--text-muted);font-weight:600;padding-bottom:8px}
.compare-btn{background:var(--accent);color:var(--bg);border:none;padding:8px 20px;border-radius:4px;
  font-family:var(--font);font-size:12px;font-weight:600;cursor:pointer;white-space:nowrap;height:35px}
.compare-btn:hover{opacity:.9}
.compare-btn:disabled{opacity:.4;cursor:not-allowed}
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
.diff-tool-card{background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:10px 12px;margin-bottom:8px}
.diff-tool-name{font-weight:600;font-size:12px;margin-bottom:6px}
.diff-tool-detail{font-size:11px;line-height:1.6}
.diff-tool-detail pre{background:var(--surface);padding:6px 10px;border-radius:4px;margin:4px 0;
  white-space:pre-wrap;word-break:break-all;font-size:11px;max-height:150px;overflow-y:auto}
.diff-section{margin-bottom:20px}
.diff-section-title{font-size:12px;font-weight:600;margin-bottom:10px;padding-bottom:6px;
  border-bottom:1px solid var(--border)}

/* Waterfall */
.waterfall-run{background:var(--surface);border:1px solid var(--border);border-radius:6px;margin-bottom:8px;overflow:hidden}
.waterfall-run-header{display:flex;align-items:center;gap:10px;padding:10px 16px;cursor:pointer;font-size:12px;transition:background .15s}
.waterfall-run-header:hover{background:var(--surface-hover)}
.waterfall-run-header .run-status{margin-left:auto;display:flex;align-items:center;gap:8px}
.waterfall-events{border-top:1px solid var(--border);background:var(--bg)}
.waterfall-evt{display:flex;align-items:center;gap:10px;padding:6px 16px 6px 32px;border-bottom:1px solid var(--border);font-size:12px}
.waterfall-evt:last-child{border-bottom:none}
.waterfall-evt-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.waterfall-evt-dot.model{background:var(--info)}
.waterfall-evt-dot.tool{background:var(--accent)}
.waterfall-evt-dot.knowledge{background:var(--purple)}
.waterfall-evt-dot.memory{background:var(--warning)}
.waterfall-evt-dot.lifecycle{background:var(--success)}
.waterfall-evt-dot.error{background:var(--error)}
.waterfall-evt-dot.guardrail{background:var(--purple)}
.waterfall-evt-dot.phase{background:var(--warning)}
.waterfall-evt-time{color:var(--text-muted);white-space:nowrap;flex-shrink:0;width:70px;font-size:11px}
.waterfall-evt-desc{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.waterfall-evt-dur{color:var(--text-muted);font-size:11px;white-space:nowrap;flex-shrink:0}
.waterfall-evt-expand{font-size:9px;color:var(--text-muted);flex-shrink:0;width:12px;text-align:center}
.waterfall-evt-expandable{cursor:pointer;transition:background .15s}
.waterfall-evt-expandable:hover{background:var(--surface-hover)}
.waterfall-evt-body{padding:8px 16px 12px 50px;border-bottom:1px solid var(--border);background:var(--bg)}
.waterfall-code-section{margin-bottom:8px}
.waterfall-code-section:last-child{margin-bottom:0}
.waterfall-code-label{font-size:10px;color:var(--text-muted);margin-bottom:4px;font-weight:600}
.waterfall-code{font-family:var(--font);font-size:11px;line-height:1.6;padding:8px 12px;
  background:var(--surface);border:1px solid var(--border);border-radius:4px;
  white-space:pre-wrap;word-break:break-all;max-height:200px;overflow-y:auto;margin:0}
.waterfall-code-output{color:var(--success);background:rgba(74,222,128,.05);border-color:rgba(74,222,128,.2)}
.waterfall-code-error{color:var(--error);background:rgba(248,113,113,.05);border-color:rgba(248,113,113,.2)}
.waterfall-phase-desc{color:var(--text-muted);font-size:11px;font-style:italic;margin-left:4px}
.waterfall-evt.skipped{opacity:.45}
.waterfall-ungrouped{margin-top:16px}
.waterfall-ungrouped-title{font-size:11px;color:var(--text-muted);margin-bottom:8px;padding-left:4px}

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

/* Chat widget */
.chat-fab{position:fixed;bottom:24px;right:24px;width:48px;height:48px;border-radius:50%;
  background:var(--accent);color:var(--bg);border:none;cursor:pointer;z-index:200;
  display:flex;align-items:center;justify-content:center;font-size:20px;
  box-shadow:0 4px 16px rgba(0,0,0,.4);transition:transform .15s,box-shadow .15s}
.chat-fab:hover{transform:scale(1.08);box-shadow:0 6px 20px rgba(0,0,0,.5)}
.chat-panel{position:fixed;bottom:24px;right:24px;width:420px;height:560px;z-index:200;
  background:var(--bg);border:1px solid var(--border);border-radius:12px;
  display:flex;flex-direction:column;box-shadow:0 8px 32px rgba(0,0,0,.5);overflow:hidden}
.chat-header{display:flex;align-items:center;justify-content:space-between;
  padding:12px 16px;background:var(--surface);border-bottom:1px solid var(--border)}
.chat-header-title{font-size:12px;font-weight:600;display:flex;align-items:center;gap:8px}
.chat-header-actions{display:flex;gap:6px}
.chat-header-btn{background:none;border:1px solid var(--border);color:var(--text-muted);
  padding:3px 8px;border-radius:4px;font-family:var(--font);font-size:10px;cursor:pointer}
.chat-header-btn:hover{color:var(--text);border-color:var(--text-muted)}
.chat-close{background:none;border:none;color:var(--text-muted);cursor:pointer;font-size:16px;
  padding:2px 6px;border-radius:4px}
.chat-close:hover{color:var(--text);background:var(--surface-hover)}
.chat-messages{flex:1;overflow-y:auto;padding:12px 16px;display:flex;flex-direction:column;gap:10px}
.chat-msg{max-width:85%;padding:8px 12px;border-radius:8px;font-size:12px;line-height:1.6;word-break:break-word}
.chat-msg.user{align-self:flex-end;background:var(--accent);color:var(--bg);border-bottom-right-radius:2px}
.chat-msg.assistant{align-self:flex-start;background:var(--surface);border:1px solid var(--border);
  border-bottom-left-radius:2px}
.chat-msg.assistant pre{background:var(--bg);padding:6px 8px;border-radius:4px;margin:6px 0;
  font-size:11px;overflow-x:auto;white-space:pre-wrap;word-break:break-all}
.chat-msg.assistant code{background:var(--bg);padding:1px 4px;border-radius:3px;font-size:11px}
.chat-msg.error{align-self:center;color:var(--error);font-size:11px;font-style:italic}
.chat-msg-meta{font-size:9px;color:var(--text-muted);margin-top:4px}
.chat-typing{align-self:flex-start;padding:8px 12px;color:var(--text-muted);font-size:12px;font-style:italic}
.chat-typing .dot{display:inline-block;animation:chatdot 1.2s infinite;opacity:.3}
.chat-typing .dot:nth-child(2){animation-delay:.2s}
.chat-typing .dot:nth-child(3){animation-delay:.4s}
@keyframes chatdot{0%,100%{opacity:.3}50%{opacity:1}}
.chat-input-bar{display:flex;gap:8px;padding:10px 12px;border-top:1px solid var(--border);background:var(--surface)}
.chat-input{flex:1;background:var(--bg);border:1px solid var(--border);color:var(--text);
  padding:8px 12px;border-radius:6px;font-family:var(--font);font-size:12px;resize:none;
  min-height:36px;max-height:80px;outline:none}
.chat-input:focus{border-color:var(--accent)}
.chat-send{background:var(--accent);color:var(--bg);border:none;padding:0 14px;border-radius:6px;
  font-family:var(--font);font-size:12px;font-weight:600;cursor:pointer;white-space:nowrap}
.chat-send:hover{opacity:.9}
.chat-send:disabled{opacity:.4;cursor:not-allowed}
.chat-empty{text-align:center;color:var(--text-muted);padding:32px 16px;font-size:12px}
.chat-empty-icon{font-size:28px;margin-bottom:8px;opacity:.4}

/* Responsive */
@media(max-width:768px){
  .sidebar{position:fixed;left:-220px;top:0;bottom:0;z-index:100;transition:left .2s}
  .sidebar.open{left:0}
  .stat-grid{grid-template-columns:repeat(2,1fr)}
  .split{grid-template-columns:1fr}
  .diff-meta{grid-template-columns:1fr}
  .phase-timeline{height:24px}
  .phase-bar .phase-label{font-size:8px}
  .chat-panel{width:calc(100vw - 32px);right:16px;bottom:16px;height:70vh}
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
      <div class="stat-delta" x-text="'++ ' + (metrics.recent_runs ? metrics.recent_runs.length : 0) + ' recent'"></div>
    </div>
    <div class="stat-card">
      <div class="stat-label"><em>const</em> success_rate =</div>
      <div class="stat-value" x-text="formatPercent((metrics.success_rate || 0) * 100)"></div>
      <div class="stat-delta" x-text="'++ ' + formatPercent((metrics.success_rate || 0) * 100) + ' overall'"></div>
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
        <span x-text="chartBucketLabel()"></span>
      </div>
      <div class="chart-wrap">
        <div class="chart-range-bar">
          <template x-for="r in [{v:'auto',l:'auto'},{v:'1h',l:'1h'},{v:'6h',l:'6h'},{v:'24h',l:'24h'},{v:'7d',l:'7d'}]" :key="r.v">
            <button class="chart-range-btn" :class="{'active': chartRange === r.v}"
                    @click="chartRange = r.v; fetchTimeline()" x-text="r.l"></button>
          </template>
        </div>
        <div class="chart-container"
             @mouseleave="chartTooltipVisible = false"
             x-html="renderBarChart(metrics.timeline || [])">
        </div>
        <div class="chart-tooltip" :class="{'visible': chartTooltipVisible}"
             :style="'left:' + chartTooltipX + 'px;top:' + chartTooltipY + 'px'"
             x-html="chartTooltipHtml">
        </div>
      </div>
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
      <template x-if="filteredEvents().length > 0">
        <div>
          <template x-for="group in groupedEvents()" :key="group.run_id">
            <div class="waterfall-run" x-data="{open: true}">
              <div class="waterfall-run-header" @click="open = !open">
                <span x-text="open ? '&#9660;' : '&#9654;'" style="font-size:10px;color:var(--text-muted)"></span>
                <div class="run-dot" :class="group.status"></div>
                <strong x-text="group.agent_name || 'agent'" style="font-size:12px"></strong>
                <span style="color:var(--text-muted);font-size:11px" x-text="group.model || ''"></span>
                <div class="run-status">
                  <span class="badge" :class="'badge-' + group.status" x-text="'[' + group.status.toUpperCase() + ']'"></span>
                  <span style="font-size:11px;color:var(--text-muted)" x-text="formatNumber(group.tokens) + ' tok'"></span>
                  <span style="font-size:11px;color:var(--text-muted)" x-text="group.duration ? formatDuration(group.duration) : ''"></span>
                  <span style="font-size:11px;color:var(--text-muted)" x-text="group.events.length + ' events'"></span>
                </div>
              </div>
              <div class="waterfall-events" x-show="open" x-cloak>
                <template x-for="evt in group.events" :key="evt._idx">
                  <div x-show="evt.event !== 'PhaseStarted'" x-data="{exp: false}">
                    <!-- Tool call events: expandable with code preview -->
                    <template x-if="isToolEvent(evt)">
                      <div>
                        <div class="waterfall-evt waterfall-evt-expandable" @click="exp = !exp">
                          <div class="waterfall-evt-dot" :class="eventCategory(evt.event)"></div>
                          <span class="waterfall-evt-time" x-text="formatTime(evt.created_at)"></span>
                          <span class="event-type" :class="eventCategory(evt.event)"
                                x-text="evt.event" style="font-size:10px;min-width:80px"></span>
                          <span class="waterfall-evt-desc" x-text="richEventSummary(evt)"></span>
                          <span class="waterfall-evt-dur" x-text="eventDuration(evt)"></span>
                          <span class="waterfall-evt-expand" x-text="exp ? '\u25bc' : '\u25b6'"></span>
                        </div>
                        <div class="waterfall-evt-body" x-show="exp" x-cloak>
                          <div class="waterfall-code-section" x-show="evt.tool && evt.tool.tool_args">
                            <div class="waterfall-code-label">// arguments</div>
                            <pre class="waterfall-code"
                                 x-text="JSON.stringify(evt.tool?.tool_args || {}, null, 2)"></pre>
                          </div>
                          <div class="waterfall-code-section" x-show="evt.tool && evt.tool.result">
                            <div class="waterfall-code-label">// output</div>
                            <pre class="waterfall-code waterfall-code-output"
                                 x-text="evt.tool?.result || ''"></pre>
                          </div>
                          <div class="waterfall-code-section" x-show="evt.tool && evt.tool.tool_call_error">
                            <div class="waterfall-code-label">// error</div>
                            <pre class="waterfall-code waterfall-code-error"
                                 x-text="evt.tool?.tool_call_error || ''"></pre>
                          </div>
                        </div>
                      </div>
                    </template>
                    <!-- Phase completed: rich description -->
                    <template x-if="evt.event === 'PhaseCompleted'">
                      <div class="waterfall-evt" :class="{'skipped': evt.skipped}">
                        <div class="waterfall-evt-dot phase"></div>
                        <span class="waterfall-evt-time" x-text="formatTime(evt.created_at)"></span>
                        <span class="event-type phase" style="font-size:10px;min-width:80px"
                              x-text="evt.phase_name"></span>
                        <span class="waterfall-evt-desc">
                          <span x-text="phaseDescription(evt.phase_name)"></span>
                          <span class="waterfall-phase-desc"
                                x-show="evt.skipped" x-text="'[skipped]'"></span>
                        </span>
                        <span class="waterfall-evt-dur" x-text="eventDuration(evt)"></span>
                      </div>
                    </template>
                    <!-- Model call events: expandable with details -->
                    <template x-if="isModelEvent(evt)">
                      <div>
                        <div class="waterfall-evt waterfall-evt-expandable" @click="exp = !exp">
                          <div class="waterfall-evt-dot model"></div>
                          <span class="waterfall-evt-time" x-text="formatTime(evt.created_at)"></span>
                          <span class="event-type model" style="font-size:10px;min-width:80px"
                                x-text="evt.event"></span>
                          <span class="waterfall-evt-desc" x-text="richEventSummary(evt)"></span>
                          <span class="waterfall-evt-dur" x-text="eventDuration(evt)"></span>
                          <span class="waterfall-evt-expand" x-text="exp ? '\u25bc' : '\u25b6'"></span>
                        </div>
                        <div class="waterfall-evt-body" x-show="exp" x-cloak>
                          <template x-if="evt.event === 'ModelCallCompleted' && evt.metrics">
                            <div class="waterfall-code-section">
                              <div class="waterfall-code-label">// metrics</div>
                              <pre class="waterfall-code"
                                   x-text="JSON.stringify(evt.metrics, null, 2)"></pre>
                            </div>
                          </template>
                          <template x-if="evt.tool_calls && evt.tool_calls.length > 0">
                            <div class="waterfall-code-section">
                              <div class="waterfall-code-label">// tool calls requested</div>
                              <pre class="waterfall-code"
                                   x-text="JSON.stringify(evt.tool_calls, null, 2)"></pre>
                            </div>
                          </template>
                          <template x-if="evt.content">
                            <div class="waterfall-code-section">
                              <div class="waterfall-code-label">// content</div>
                              <pre class="waterfall-code waterfall-code-output"
                                   x-text="typeof evt.content === 'string' ? evt.content : JSON.stringify(evt.content, null, 2)"></pre>
                            </div>
                          </template>
                        </div>
                      </div>
                    </template>
                    <!-- All other events: standard row -->
                    <template x-if="!isToolEvent(evt) && !isModelEvent(evt) && evt.event !== 'PhaseCompleted'">
                      <div class="waterfall-evt">
                        <div class="waterfall-evt-dot" :class="eventCategory(evt.event)"></div>
                        <span class="waterfall-evt-time" x-text="formatTime(evt.created_at)"></span>
                        <span class="event-type" :class="eventCategory(evt.event)"
                              x-text="evt.event" style="font-size:10px;min-width:80px"></span>
                        <span class="waterfall-evt-desc" x-text="richEventSummary(evt)"></span>
                        <span class="waterfall-evt-dur" x-text="eventDuration(evt)"></span>
                      </div>
                    </template>
                  </div>
                </template>
              </div>
            </div>
          </template>
          <template x-if="groupedEvents()._ungrouped && groupedEvents()._ungrouped.length > 0">
            <div class="waterfall-ungrouped">
              <div class="waterfall-ungrouped-title">// ungrouped events</div>
              <template x-for="evt in groupedEvents()._ungrouped" :key="evt._idx">
                <div class="event-row">
                  <span class="event-time" x-text="formatTime(evt.created_at)"></span>
                  <span class="event-type" :class="eventCategory(evt.event)" x-text="evt.event"></span>
                  <span class="event-detail" x-text="richEventSummary(evt)"></span>
                </div>
              </template>
            </div>
          </template>
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

  <div class="compare-bar">
    <div class="compare-field">
      <label>Run A</label>
      <select x-model="compareA">
        <option value="">-- select run --</option>
        <template x-for="r in allRunIds" :key="r.run_id">
          <option :value="r.run_id"
                  x-text="r.run_id.substring(0,12) + '... [' + r.status + '] ' + r.tokens + ' tok'">
          </option>
        </template>
      </select>
    </div>
    <span class="compare-vs">vs</span>
    <div class="compare-field">
      <label>Run B</label>
      <select x-model="compareB">
        <option value="">-- select run --</option>
        <template x-for="r in allRunIds" :key="r.run_id">
          <option :value="r.run_id"
                  x-text="r.run_id.substring(0,12) + '... [' + r.status + '] ' + r.tokens + ' tok'">
          </option>
        </template>
      </select>
    </div>
    <button class="compare-btn" @click="loadComparison()"
            :disabled="!compareA || !compareB || compareA === compareB || loadingCompare">
      Compare
    </button>
  </div>

  <template x-if="compareA && compareB && compareA === compareB">
    <div style="color:var(--warning);font-size:12px;margin-bottom:16px">
      select two different runs to compare
    </div>
  </template>

  <template x-if="loadingCompare">
    <div class="loading"><div class="spinner"></div> comparing runs...</div>
  </template>

  <template x-if="!loadingCompare && compareResult && compareResult._error">
    <div class="panel" style="color:var(--error);padding:16px;font-size:12px">
      <span style="font-weight:600">error:</span> <span x-text="compareResult._error"></span>
    </div>
  </template>

  <template x-if="!loadingCompare && compareResult && !compareResult._error">
    <div>
      <div class="diff-meta">
        <div class="diff-meta-card">
          <div class="diff-meta-label">token delta</div>
          <div class="diff-meta-value"
               :class="compareResult.token_diff > 0 ? 'positive' : (compareResult.token_diff < 0 ? 'negative' : '')"
               x-text="(compareResult.token_diff > 0 ? '+' : '') + formatNumber(compareResult.token_diff || 0)">
          </div>
        </div>
        <div class="diff-meta-card">
          <div class="diff-meta-label">cost delta</div>
          <div class="diff-meta-value"
               :class="compareResult.cost_diff > 0 ? 'positive' : (compareResult.cost_diff < 0 ? 'negative' : '')"
               x-text="(compareResult.cost_diff > 0 ? '+$' : '-$') + formatCost(Math.abs(compareResult.cost_diff || 0))">
          </div>
        </div>
        <div class="diff-meta-card">
          <div class="diff-meta-label">duration delta</div>
          <div class="diff-meta-value"
               :class="compareResult.duration_diff > 0 ? 'positive' : (compareResult.duration_diff < 0 ? 'negative' : '')"
               x-text="(compareResult.duration_diff > 0 ? '+' : '') + formatDuration(compareResult.duration_diff)">
          </div>
        </div>
      </div>

      <!-- Content diff -->
      <div class="diff-section">
        <div class="diff-section-title" style="color:var(--accent)">// output_diff</div>
        <template x-if="compareResult.content_diff">
          <div class="diff-block"><pre x-html="renderDiff(compareResult.content_diff)"></pre></div>
        </template>
        <template x-if="!compareResult.content_diff">
          <div class="diff-block" style="text-align:center;color:var(--text-muted)">
            outputs are identical
          </div>
        </template>
      </div>

      <!-- Tool calls diff -->
      <template x-if="compareResult.tool_calls_diff">
        <div class="diff-section">
          <div class="diff-section-title" style="color:var(--accent)">// tool_calls_diff</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">

            <!-- Added tools -->
            <div>
              <div style="color:var(--success);font-weight:600;font-size:12px;margin-bottom:8px">
                + added in Run B
                (<span x-text="(compareResult.tool_calls_diff.added||[]).length"></span>)
              </div>
              <template x-for="(tc, i) in compareResult.tool_calls_diff.added||[]" :key="'add-'+i">
                <div class="diff-tool-card" style="border-left:3px solid var(--success)">
                  <div class="diff-tool-name" style="color:var(--success)"
                       x-text="'+ ' + tc.tool_name"></div>
                  <div class="diff-tool-detail">
                    <div style="color:var(--text-muted);font-size:10px;margin-bottom:2px">arguments</div>
                    <pre x-text="JSON.stringify(tc.tool_args || {}, null, 2)"></pre>
                    <div style="color:var(--text-muted);font-size:10px;margin-bottom:2px;margin-top:6px">result</div>
                    <pre style="color:var(--success)" x-text="String(tc.result || 'null')"></pre>
                  </div>
                </div>
              </template>
              <template x-if="!(compareResult.tool_calls_diff.added||[]).length">
                <div style="color:var(--text-muted);font-size:12px;font-style:italic">none</div>
              </template>
            </div>

            <!-- Removed tools -->
            <div>
              <div style="color:var(--error);font-weight:600;font-size:12px;margin-bottom:8px">
                - removed from Run A
                (<span x-text="(compareResult.tool_calls_diff.removed||[]).length"></span>)
              </div>
              <template x-for="(tc, i) in compareResult.tool_calls_diff.removed||[]" :key="'rem-'+i">
                <div class="diff-tool-card" style="border-left:3px solid var(--error)">
                  <div class="diff-tool-name" style="color:var(--error)"
                       x-text="'- ' + tc.tool_name"></div>
                  <div class="diff-tool-detail">
                    <div style="color:var(--text-muted);font-size:10px;margin-bottom:2px">arguments</div>
                    <pre x-text="JSON.stringify(tc.tool_args || {}, null, 2)"></pre>
                    <div style="color:var(--text-muted);font-size:10px;margin-bottom:2px;margin-top:6px">result</div>
                    <pre style="color:var(--error)" x-text="String(tc.result || 'null')"></pre>
                  </div>
                </div>
              </template>
              <template x-if="!(compareResult.tool_calls_diff.removed||[]).length">
                <div style="color:var(--text-muted);font-size:12px;font-style:italic">none</div>
              </template>
            </div>

          </div>

          <!-- Common tools -->
          <div style="margin-top:12px;color:var(--text-muted);font-size:12px">
            <span style="font-weight:600">common tool calls:</span>
            <span x-text="compareResult.tool_calls_diff.common || 0"></span>
          </div>
        </div>
      </template>
    </div>
  </template>

  <template x-if="!loadingCompare && !compareResult && !(compareA && compareB)">
    <div class="empty-state">
      <div class="empty-state-icon">&gt;_</div>
      <div class="empty-state-text">select two runs and click Compare</div>
    </div>
  </template>
</div>"""


def _tools_page() -> str:
  return """<div x-show="page==='tools'" x-cloak>
  <div class="page-title"><span>&gt;</span> TOOLS<span>|</span></div>
  <div class="page-subtitle"># tool call analytics — count, errors, latency</div>

  <template x-if="!metrics.tool_stats || !Array.isArray(metrics.tool_stats) || metrics.tool_stats.length === 0">
    <div class="empty-state">
      <div class="empty-state-icon">&gt;_</div>
      <div class="empty-state-text">no tool call data yet</div>
    </div>
  </template>
  <template x-if="Array.isArray(metrics.tool_stats) && metrics.tool_stats.length > 0">
    <div class="panel">
      <table class="data-table">
        <thead><tr>
          <th>tool_name</th><th>calls</th><th>errors</th><th>avg latency</th><th>error rate</th>
        </tr></thead>
        <tbody>
          <template x-for="stats in metrics.tool_stats" :key="stats.tool_name">
            <tr>
              <td><strong x-text="stats.tool_name"></strong></td>
              <td x-text="stats.call_count || 0"></td>
              <td :style="(stats.error_count || 0) > 0 ? 'color:var(--error)' : ''" x-text="stats.error_count || 0"></td>
              <td x-text="formatMs(stats.avg_duration_ms)"></td>
              <td :style="(stats.error_count || 0) > 0 ? 'color:var(--error)' : ''"
                  x-text="stats.call_count ? formatPercent((stats.error_count || 0) / stats.call_count * 100) : '0%'"></td>
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

  <template x-if="!metrics.model_stats || !Array.isArray(metrics.model_stats) || metrics.model_stats.length === 0">
    <div class="empty-state">
      <div class="empty-state-icon">&gt;_</div>
      <div class="empty-state-text">no model call data yet</div>
    </div>
  </template>
  <template x-if="Array.isArray(metrics.model_stats) && metrics.model_stats.length > 0">
    <div class="panel">
      <table class="data-table">
        <thead><tr>
          <th>model_id</th><th>calls</th><th>avg tokens</th><th>total cost</th>
        </tr></thead>
        <tbody>
          <template x-for="stats in metrics.model_stats" :key="stats.model_id">
            <tr>
              <td><strong x-text="stats.model_id"></strong></td>
              <td x-text="stats.call_count || 0"></td>
              <td x-text="formatNumber(stats.avg_tokens || 0)"></td>
              <td x-text="'$' + formatCost(stats.total_cost || 0)"></td>
            </tr>
          </template>
        </tbody>
      </table>
    </div>
  </template>
</div>"""


def _chat_widget() -> str:
  return """<!-- Chat FAB (floating action button) -->
<button class="chat-fab" x-show="!chatOpen" @click="chatOpen = true" title="Chat with agent">
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
  </svg>
</button>

<!-- Chat panel -->
<div class="chat-panel" x-show="chatOpen" x-cloak x-transition.opacity.duration.150ms>
  <div class="chat-header">
    <div class="chat-header-title">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
      </svg>
      <span x-text="agentName + ' chat'"></span>
      <span style="font-size:9px;color:var(--text-muted)" x-text="chatMessages.length ? chatMessages.length + ' msgs' : ''"></span>
    </div>
    <div class="chat-header-actions">
      <button class="chat-header-btn" @click="resetChat()" title="Clear history">clear</button>
      <button class="chat-close" @click="chatOpen = false" title="Close">&times;</button>
    </div>
  </div>

  <div class="chat-messages" x-ref="chatMessages">
    <!-- Empty state -->
    <template x-if="chatMessages.length === 0 && !chatSending">
      <div class="chat-empty">
        <div class="chat-empty-icon">&gt;_</div>
        <div>Send a message to test your agent.</div>
        <div style="margin-top:4px;font-size:10px;color:var(--text-muted)">
          Runs will appear in Live Events in real time.
        </div>
      </div>
    </template>

    <!-- Messages -->
    <template x-for="(msg, i) in chatMessages" :key="i">
      <div>
        <div class="chat-msg" :class="msg.role" x-html="msg.role === 'assistant' ? renderMarkdown(msg.content) : escapeHtml(msg.content)"></div>
        <div class="chat-msg-meta" :style="msg.role === 'user' ? 'text-align:right' : ''"
          x-text="msg.tokens ? msg.tokens + ' tok' + (msg.cost ? ' &middot; $' + msg.cost.toFixed(4) : '') : ''">
        </div>
      </div>
    </template>

    <!-- Typing indicator -->
    <div class="chat-typing" x-show="chatSending">
      <span class="dot">.</span><span class="dot">.</span><span class="dot">.</span> thinking
    </div>
  </div>

  <div class="chat-input-bar">
    <textarea class="chat-input" x-model="chatInput"
      @keydown.enter.prevent="if (!$event.shiftKey) sendChat()"
      placeholder="Message the agent..."
      :disabled="chatSending"
      rows="1"
      x-ref="chatInput"></textarea>
    <button class="chat-send" @click="sendChat()" :disabled="!chatInput.trim() || chatSending">
      send
    </button>
  </div>
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

    // Chart
    chartRange: 'auto',
    chartBucketSize: 3600,
    chartResolvedBucket: 'hour',
    chartTooltipVisible: false,
    chartTooltipX: 0,
    chartTooltipY: 0,
    chartTooltipHtml: '',

    // Chat
    chatOpen: false,
    chatMessages: [],
    chatInput: '',
    chatSending: false,
    chatSessionId: 'obs-chat-' + Date.now(),

    // Init
    async init() {
      document.documentElement.setAttribute('data-theme', this.theme);
      // Bridge for SVG inline event handlers → Alpine tooltip
      const self = this;
      document.__obsTooltip = (evt, bucket) => self.showChartTooltip(evt, bucket);
      document.__obsTooltipHide = () => { self.chartTooltipVisible = false; };
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
        this.recentRuns = data.recent_runs || [];
      }
      await this.fetchTimeline();
    },

    async fetchTimeline() {
      const tl = await this.fetchJSON(
        '/obs/api/metrics/timeline?bucket=auto&range=' + encodeURIComponent(this.chartRange)
      );
      if (tl && tl.data) {
        this.chartBucketSize = tl.bucket_seconds || 3600;
        this.chartResolvedBucket = tl.bucket || 'hour';
        this.metrics.timeline = tl.data.map(b => ({
          ts: b.timestamp,
          count: b.run_count || 0,
          errors: b.error_count || 0,
          tokens: b.tokens || 0,
          cost: b.cost || 0,
        }));
      }
    },

    // Sessions
    async fetchSessions() {
      this.loadingSessions = true;
      const data = await this.fetchJSON('/obs/api/sessions');
      this.sessions = (data && data.sessions) || [];
      this.loadingSessions = false;
    },

    async selectSession(sid) {
      this.selectedSession = sid;
      this.loadingRuns = true;
      const data = await this.fetchJSON('/obs/api/sessions/' + encodeURIComponent(sid));
      this.sessionRuns = (data && data.runs) || [];
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
      // Primary source: recentRuns from metrics (includes live buffer runs)
      const seen = new Set();
      const runs = [];
      for (const r of (this.recentRuns || [])) {
        if (!seen.has(r.run_id)) {
          seen.add(r.run_id);
          runs.push(r);
        }
      }
      // Secondary source: sessions API (persisted trace files)
      const data = await this.fetchJSON('/obs/api/sessions');
      if (data && data.sessions) {
        for (const s of data.sessions) {
          const resp = await this.fetchJSON(
            '/obs/api/sessions/' + encodeURIComponent(s.session_id)
          );
          if (resp && resp.runs) {
            for (const r of resp.runs) {
              if (!seen.has(r.run_id)) {
                seen.add(r.run_id);
                runs.push({
                  run_id: r.run_id,
                  session_id: s.session_id,
                  status: r.status || 'COMPLETED',
                  tokens: r.tokens?.total_tokens || 0,
                  cost: r.cost || 0,
                  duration: r.duration || 0,
                });
              }
            }
          }
        }
      }
      this.allRunIds = runs;
    },

    async loadComparison() {
      if (!this.compareA || !this.compareB || this.compareA === this.compareB) {
        this.compareResult = null;
        return;
      }
      this.loadingCompare = true;
      this.compareResult = null;
      const url = '/obs/api/compare?a=' + encodeURIComponent(this.compareA)
        + '&b=' + encodeURIComponent(this.compareB);
      try {
        const resp = await fetch(url);
        const data = await resp.json();
        if (!resp.ok) {
          this.compareResult = { _error: data.error || 'Compare failed (HTTP ' + resp.status + ')' };
        } else {
          this.compareResult = data;
        }
      } catch (e) {
        this.compareResult = { _error: 'Connection error: ' + e.message };
      }
      this.loadingCompare = false;
    },

    // SSE
    _sseRetries: 0,
    _sseReconnectTimer: null,

    connectSSE() {
      if (this._sseReconnectTimer) { clearTimeout(this._sseReconnectTimer); this._sseReconnectTimer = null; }
      if (this.sseSource) { this.sseSource.close(); this.sseSource = null; }
      try {
        this.sseSource = new EventSource('/obs/api/events');
        this.sseSource.onopen = () => { this.connected = true; this._sseRetries = 0; };
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
        this.sseSource.onerror = () => {
          this.connected = false;
          if (this.sseSource) { this.sseSource.close(); this.sseSource = null; }
          const delay = Math.min(1000 * Math.pow(2, this._sseRetries), 30000);
          this._sseRetries++;
          this._sseReconnectTimer = setTimeout(() => this.connectSSE(), delay);
        };
        window.addEventListener('beforeunload', () => {
          if (this._sseReconnectTimer) clearTimeout(this._sseReconnectTimer);
          if (this.sseSource) { this.sseSource.close(); this.sseSource = null; }
        });
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

    isToolEvent(evt) {
      return evt.event === 'ToolCallStarted' || evt.event === 'ToolCallCompleted';
    },

    isModelEvent(evt) {
      return evt.event === 'ModelCallStarted' || evt.event === 'ModelCallCompleted';
    },

    phaseDescription(name) {
      const desc = {
        'prepare': 'Initialize run context — resolve model, register tools, set session ID',
        'recall': 'Recall session memory — load conversation history, inject summaries',
        'think': 'Reasoning step — internal chain-of-thought before model call',
        'guard_input': 'Input guardrails — validate user input before processing',
        'compose': 'Assemble messages — build system prompt, tool definitions, context',
        'invoke_loop': 'Inference loop — model calls, tool execution, multi-turn reasoning',
        'guard_output': 'Output guardrails — validate model response before returning',
        'store': 'Persist results — store to memory, emit completion events',
      };
      return desc[name] || name;
    },

    eventSummary(evt) {
      if (evt.content) return typeof evt.content === 'string' ? evt.content.substring(0, 120) : JSON.stringify(evt.content).substring(0, 120);
      if (evt.tool) return (evt.tool.tool_name || '') + '(' + JSON.stringify(evt.tool.tool_args || {}).substring(0, 60) + ')';
      if (evt.agent_name) return evt.agent_name;
      if (evt.phase_name) return 'phase: ' + evt.phase_name;
      return '';
    },

    groupedEvents() {
      const filtered = this.filteredEvents().slice(-500);
      const groups = {};
      const ungrouped = [];
      const order = [];
      for (const evt of filtered) {
        const rid = evt.run_id;
        if (!rid) { ungrouped.push(evt); continue; }
        if (!groups[rid]) {
          groups[rid] = { run_id: rid, agent_name: '', model: '', status: 'running', tokens: 0, duration: null, events: [] };
          order.push(rid);
        }
        const g = groups[rid];
        g.events.push(evt);
        if (evt.agent_name) g.agent_name = evt.agent_name;
        if (evt.model_id) g.model = evt.model_id;
        if (evt.event === 'RunCompleted') {
          g.status = 'completed';
          const m = evt.metrics;
          if (m) { g.tokens = m.total_tokens || 0; g.duration = m.duration || null; }
        } else if (evt.event === 'RunError') { g.status = 'error'; }
        else if (evt.event === 'RunCancelled') { g.status = 'cancelled'; }
        else if (evt.event === 'RunPaused') { g.status = 'paused'; }
      }
      const result = order.map(rid => groups[rid]);
      result._ungrouped = ungrouped;
      return result;
    },

    richEventSummary(evt) {
      const e = evt.event;
      if (!e) return '';
      if (e === 'RunStarted') return 'Agent run started' + (evt.agent_name ? ' (' + evt.agent_name + ')' : '');
      if (e === 'RunCompleted') {
        const m = evt.metrics;
        if (m) return 'Completed — ' + (m.total_tokens || 0) + ' tokens, $' + ((m.cost || 0).toFixed(4));
        return 'Run completed';
      }
      if (e === 'RunError') return 'Error: ' + (evt.error || 'unknown');
      if (e === 'RunCancelled') return 'Run cancelled';
      if (e === 'RunPaused') return 'Run paused — awaiting input';
      if (e === 'ToolCallStarted') {
        const t = evt.tool;
        return t ? 'Calling ' + (t.tool_name || '?')
          + '(' + JSON.stringify(t.tool_args || {}).substring(0, 60) + ')'
          : 'Tool call started';
      }
      if (e === 'ToolCallCompleted') {
        const t = evt.tool;
        return t ? (t.tool_name || '?') + ' returned'
          + (t.tool_call_error ? ' [ERROR]' : '')
          : 'Tool call completed';
      }
      if (e === 'ModelCallStarted') {
        return 'Model call' + (evt.model_id ? ' (' + evt.model_id + ')' : '')
          + ' \u2014 ' + (evt.messages_count || '?') + ' messages';
      }
      if (e === 'ModelCallCompleted') {
        const m = evt.metrics;
        if (m) return 'Model responded — ' + (m.total_tokens || 0) + ' tokens' + (m.cost ? ', $' + m.cost.toFixed(4) : '');
        return 'Model call completed';
      }
      if (e === 'GuardrailTriggered') return 'Guardrail: ' + (evt.guardrail_name || '?') + ' — ' + (evt.action || 'blocked');
      if (e === 'GuardrailPassed') return 'Guardrail passed: ' + (evt.guardrail_name || '?');
      if (e === 'PhaseStarted') return 'Phase: ' + (evt.phase_name || '?');
      if (e === 'PhaseCompleted') return 'Phase done: ' + (evt.phase_name || '?');
      if (e === 'KnowledgeRetrievalCompleted') {
        return 'Retrieved ' + (evt.documents_found || 0) + ' documents'
          + (evt.query ? ' for "' + evt.query.substring(0, 40) + '"' : '');
      }
      if (e === 'MemoryRecallCompleted') return 'Memory recall — ' + (evt.chunks_included || 0) + ' chunks';
      if (e === 'SessionSummaryCreated') return 'Session summary generated';
      if (e === 'SubAgentSpawned') return 'Spawned sub-agent: ' + (evt.child_agent_name || '?');
      if (e === 'SubAgentCompleted') return 'Sub-agent completed: ' + (evt.child_agent_name || '?');
      if (e === 'SubAgentFailed') return 'Sub-agent failed: ' + (evt.child_agent_name || '?');
      if (evt.content) return typeof evt.content === 'string' ? evt.content.substring(0, 120) : JSON.stringify(evt.content).substring(0, 120);
      if (evt.phase_name) return 'phase: ' + evt.phase_name;
      return e;
    },

    eventDuration(evt) {
      const e = evt.event;
      if (e === 'ToolCallCompleted' && evt.tool && evt.tool.duration_ms != null) return this.formatMs(evt.tool.duration_ms);
      if (e === 'ModelCallCompleted' && evt.metrics && evt.metrics.duration != null) return this.formatDuration(evt.metrics.duration);
      if (e === 'RunCompleted' && evt.metrics && evt.metrics.duration != null) return this.formatDuration(evt.metrics.duration);
      if (e === 'PhaseCompleted' && evt.duration_ms != null) return this.formatMs(evt.duration_ms);
      if (e === 'KnowledgeRetrievalCompleted' && evt.duration_ms != null) return this.formatMs(evt.duration_ms);
      if (e === 'MemoryRecallCompleted' && evt.duration_ms != null) return this.formatMs(evt.duration_ms);
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

    // Chart bucket label for header
    chartBucketLabel() {
      const bk = this.chartResolvedBucket || 'hour';
      const rng = this.chartRange || 'auto';
      const labels = {'5min':'5 min','30min':'30 min','hour':'1 hr','day':'1 day'};
      return (labels[bk] || bk) + ' buckets' + (rng !== 'auto' ? ' · ' + rng + ' range' : '');
    },

    // Tooltip handlers — called from SVG bar onmouseenter
    showChartTooltip(evt, bucket) {
      const rect = evt.target.closest('.chart-container').getBoundingClientRect();
      this.chartTooltipX = evt.clientX - rect.left + 12;
      this.chartTooltipY = evt.clientY - rect.top - 10;
      // Flip if near right edge
      if (this.chartTooltipX > rect.width - 160) this.chartTooltipX -= 180;
      const d = new Date(bucket.ts * 1000);
      const end = new Date((bucket.ts + this.chartBucketSize) * 1000);
      const fmt = (dt) => dt.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
      const dateFmt = d.toLocaleDateString([], {month:'short', day:'numeric'});
      let html = '<div class="chart-tooltip-time">' + dateFmt + ' ' + fmt(d) + ' – ' + fmt(end) + '</div>';
      html += '<div class="chart-tooltip-row"><span>Runs</span><span>' + (bucket.count || 0) + '</span></div>';
      if (bucket.errors > 0) html += '<div class="chart-tooltip-row"><span>Errors</span>'
        + '<span style="color:var(--error)">' + bucket.errors + '</span></div>';
      html += '<div class="chart-tooltip-row"><span>Tokens</span><span>' + this.formatNumber(bucket.tokens || 0) + '</span></div>';
      html += '<div class="chart-tooltip-row"><span>Cost</span><span>$' + (bucket.cost || 0).toFixed(4) + '</span></div>';
      this.chartTooltipHtml = html;
      this.chartTooltipVisible = true;
    },

    // Bar chart renderer (inline SVG with local time labels + interactive tooltip)
    renderBarChart(timeline) {
      if (!timeline || timeline.length === 0) {
        return '<svg viewBox="0 0 480 160"><text x="240" y="80" fill="var(--text-muted)" '
          + 'text-anchor="middle" font-family="var(--font)" font-size="12">no data</text></svg>';
      }
      const W = 480, H = 160, pad = 30, top = 10;
      const chartH = H - pad - top;
      const n = timeline.length;
      const gap = Math.max(1, Math.min(3, Math.floor(200 / n)));
      const barW = Math.max(2, (W - 20) / n - gap);
      const maxVal = Math.max(...timeline.map(t => t.count || 0), 1);
      let bars = '';
      // Determine label step — show ~10-15 labels max
      const labelStep = Math.max(1, Math.ceil(n / 12));
      timeline.forEach((t, i) => {
        const x = 10 + i * (barW + gap);
        const val = t.count || 0;
        const errVal = t.errors || 0;
        const h = val > 0 ? Math.max(4, (val / maxVal) * chartH) : 1;
        const y = top + chartH - h;
        // JSON-encode bucket data for tooltip
        const bdata = JSON.stringify({ts:t.ts,count:val,errors:errVal,tokens:t.tokens||0,cost:t.cost||0}).replace(/"/g, '&quot;');
        if (val > 0) {
          // Main bar
          bars += '<rect class="bar" x="' + x + '" y="' + y + '" width="' + barW
            + '" height="' + h + '" rx="1.5" fill="var(--accent)" opacity="0.8"'
            + ' onmouseenter="document.__obsTooltip(event,' + bdata + ')"'
            + ' onmouseleave="document.__obsTooltipHide()" />';
          // Error overlay (red portion at bottom of bar)
          if (errVal > 0) {
            const errH = Math.max(3, (errVal / maxVal) * chartH);
            bars += '<rect x="' + x + '" y="' + (top + chartH - errH)
              + '" width="' + barW + '" height="' + errH
              + '" rx="1.5" fill="var(--error)" opacity="0.7" pointer-events="none" />';
          }
        } else {
          // Empty bucket — thin tick
          bars += '<rect x="' + x + '" y="' + (top + chartH - 1)
            + '" width="' + barW + '" height="1" fill="var(--border)" opacity="0.3" />';
        }
        // Local time labels
        if (i % labelStep === 0) {
          const d = new Date(t.ts * 1000);
          const lbl = d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
          const lblColor = val > 0 ? 'var(--text)' : 'var(--text-muted)';
          bars += '<text x="' + (x + barW/2) + '" y="' + (H - 4) + '" fill="' + lblColor + '"'
            + ' text-anchor="middle" font-family="var(--font)" font-size="8">' + lbl + '</text>';
        }
      });
      // Baseline axis
      bars += '<line x1="10" y1="' + (top + chartH) + '" x2="' + (W - 10) + '" y2="' + (top + chartH)
        + '" stroke="var(--border)" stroke-width="0.5" />';
      return '<svg viewBox="0 0 ' + W + ' ' + H + '">' + bars + '</svg>';
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

    // Chat methods
    async sendChat() {
      const text = this.chatInput.trim();
      if (!text || this.chatSending) return;
      this.chatMessages.push({ role: 'user', content: text, tokens: null, cost: null });
      this.chatInput = '';
      this.chatSending = true;
      this.$nextTick(() => this.scrollChatToBottom());
      try {
        const resp = await fetch('/obs/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, session_id: this.chatSessionId }),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({ error: 'Request failed' }));
          this.chatMessages.push({ role: 'error', content: err.error || 'Error ' + resp.status, tokens: null, cost: null });
        } else {
          const data = await resp.json();
          this.chatMessages.push({
            role: 'assistant',
            content: data.content || '',
            tokens: data.tokens ? data.tokens.total_tokens : null,
            cost: data.cost || null,
          });
          // Refresh metrics so the new run shows in overview
          this.fetchMetrics();
        }
      } catch (e) {
        this.chatMessages.push({ role: 'error', content: 'Connection error: ' + e.message, tokens: null, cost: null });
      }
      this.chatSending = false;
      this.$nextTick(() => this.scrollChatToBottom());
    },

    async resetChat() {
      try {
        await fetch('/obs/api/chat/reset', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: this.chatSessionId }),
        });
      } catch {}
      this.chatMessages = [];
      this.chatSessionId = 'obs-chat-' + Date.now();
    },

    scrollChatToBottom() {
      const el = this.$refs.chatMessages;
      if (el) el.scrollTop = el.scrollHeight;
    },

    renderMarkdown(text) {
      if (!text) return '';
      let html = this.escapeHtml(text);
      // Code blocks
      html = html.replace(/```([\\s\\S]*?)```/g, '<pre>$1</pre>');
      // Inline code
      html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
      // Bold
      html = html.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
      // Line breaks
      html = html.replace(/\\n/g, '<br>');
      return html;
    },
  };
}
</script>"""
