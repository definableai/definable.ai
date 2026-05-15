// Real REST + SSE client. Exposes window.DEFINABLE_API.
// All endpoints live under /api/ and are served by definable/observability/server.py.
(function () {
  'use strict';

  function _check(resp) {
    if (!resp.ok) {
      throw new Error('HTTP ' + resp.status + ': ' + resp.statusText);
    }
    return resp;
  }

  function _query(params) {
    if (!params) return '';
    const usable = {};
    Object.keys(params).forEach(function (k) {
      const v = params[k];
      if (v !== undefined && v !== null && v !== '') usable[k] = v;
    });
    const keys = Object.keys(usable);
    if (!keys.length) return '';
    return '?' + new URLSearchParams(usable).toString();
  }

  async function listAgents() {
    const r = await fetch('/api/agents');
    _check(r);
    return r.json();
  }

  async function listRuns(params) {
    const r = await fetch('/api/runs' + _query(params || {}));
    _check(r);
    return r.json();
  }

  async function getRun(runId) {
    const r = await fetch('/api/runs/' + encodeURIComponent(runId));
    _check(r);
    return r.json();
  }

  async function getRunEvents(runId, params) {
    const r = await fetch('/api/runs/' + encodeURIComponent(runId) + '/events' + _query(params || {}));
    _check(r);
    return r.json();
  }

  async function metrics(range) {
    const r = await fetch('/api/metrics' + _query({ range: range || '1h' }));
    _check(r);
    return r.json();
  }

  // GET SSE — long-lived browser-managed EventSource with auto-reconnect.
  // Returns an unsubscribe callable.
  function stream(onEvent, agentFilter) {
    const url = '/api/stream' + _query({ agent: agentFilter });
    const es = new EventSource(url);
    es.onmessage = function (e) {
      try {
        onEvent(JSON.parse(e.data));
      } catch (err) {
        console.error('stream parse error', err, e.data);
      }
    };
    es.onerror = function () {
      // EventSource auto-reconnects — no-op here.
    };
    return function close() {
      es.close();
    };
  }

  // POST → SSE. fetch+ReadableStream because EventSource is GET-only.
  // Calls onEvent(parsed) for every data: line. Resolves when the stream closes.
  async function playgroundRun(agent, input, onEvent) {
    const r = await fetch('/api/playground/run', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ agent: agent, input: input }),
    });
    if (!r.ok) {
      const text = await r.text();
      throw new Error('playground HTTP ' + r.status + ': ' + text);
    }
    const reader = r.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buf = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      // SSE messages end with "\n\n". Split, keep tail.
      let idx;
      while ((idx = buf.indexOf('\n\n')) !== -1) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const lines = block.split('\n');
        const dataLines = [];
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            dataLines.push(line.slice(6));
          }
        }
        if (dataLines.length) {
          const payload = dataLines.join('\n');
          try {
            onEvent(JSON.parse(payload));
          } catch (err) {
            console.error('playground parse error', err, payload);
          }
        }
      }
    }
  }

  window.DEFINABLE_API = {
    listAgents,
    listRuns,
    getRun,
    getRunEvents,
    metrics,
    stream,
    playgroundRun,
  };
})();
