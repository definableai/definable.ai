// Endpoints + React hooks. window.API and window.{useAgents, useRuns, ...}.
// All routes live under /api/ and are served by definable/observability/server.py.
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
    if (!runId) return null;
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

  window.API = {
    listAgents,
    listRuns,
    getRun,
    getRunEvents,
    metrics,
    stream,
    playgroundRun,
  };

  // Hooks — built on top of one generic useApi(loader, deps). Views never call
  // fetch directly. When mutation endpoints land, each hook returns
  // { data, refresh, mutate } — view code unchanged.
  //
  // React is loaded by index.html before this file runs, so we can attach
  // hook helpers to window once everything else is in place.
  function defineHooks() {
    if (!window.React) {
      // Defer until React is parsed.
      setTimeout(defineHooks, 0);
      return;
    }
    const { useState, useEffect, useCallback, useRef } = window.React;

    function useApi(loader, deps) {
      const [data, setData] = useState(null);
      const [loading, setLoading] = useState(true);
      const [error, setError] = useState(null);
      const aliveRef = useRef(true);
      const refresh = useCallback(async () => {
        setLoading(true);
        try {
          const v = await loader();
          if (aliveRef.current) { setData(v); setError(null); }
        } catch (e) {
          if (aliveRef.current) setError(e);
        } finally {
          if (aliveRef.current) setLoading(false);
        }
      // eslint-disable-next-line react-hooks/exhaustive-deps
      }, deps || []);
      useEffect(() => {
        aliveRef.current = true;
        refresh();
        return () => { aliveRef.current = false; };
      // eslint-disable-next-line react-hooks/exhaustive-deps
      }, deps || []);
      return { data, loading, error, refresh };
    }

    function useAgents(pollMs) {
      const h = useApi(() => API.listAgents(), []);
      useEffect(() => {
        if (!pollMs) return;
        const id = setInterval(h.refresh, pollMs);
        return () => clearInterval(id);
      }, [pollMs, h.refresh]);
      return h;
    }

    function useRuns(agentId, opts) {
      return useApi(() => API.listRuns({ agent: agentId, limit: (opts && opts.limit) || 200 }), [agentId, (opts && opts.limit) || 200]);
    }

    function useRun(runId) {
      return useApi(() => runId ? API.getRun(runId) : Promise.resolve(null), [runId]);
    }

    function useMetrics(range) {
      const h = useApi(() => API.metrics(range || '1h'), [range]);
      useEffect(() => {
        const id = setInterval(h.refresh, 10_000);
        return () => clearInterval(id);
      }, [h.refresh]);
      return h;
    }

    function useStream(handler, agentId) {
      useEffect(() => {
        if (!handler) return;
        const unsub = API.stream(handler, agentId);
        return unsub;
      // eslint-disable-next-line react-hooks/exhaustive-deps
      }, [agentId]);
    }

    function useLocalStorage(key, initial) {
      const [v, setV] = useState(() => {
        try {
          const raw = localStorage.getItem(key);
          if (raw == null) return initial;
          return JSON.parse(raw);
        } catch { return initial; }
      });
      const set = useCallback((next) => {
        setV(next);
        try { localStorage.setItem(key, JSON.stringify(next)); } catch {}
      }, [key]);
      return [v, set];
    }

    window.useApi = useApi;
    window.useAgents = useAgents;
    window.useRuns = useRuns;
    window.useRun = useRun;
    window.useMetrics = useMetrics;
    window.useStream = useStream;
    window.useLocalStorage = useLocalStorage;
  }
  defineHooks();
})();
