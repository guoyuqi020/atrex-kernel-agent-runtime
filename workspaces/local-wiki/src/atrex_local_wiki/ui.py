"""Self-contained browser client for exercising the local Wiki test double."""

from __future__ import annotations

# The HTML is deliberately kept as one dependency-free, packageable asset.
# ruff: noqa: E501
BROWSER_UI = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Atrex Local GPU Wiki</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0a0d14;
      --panel: #121824;
      --panel-2: #182131;
      --line: #29364a;
      --text: #eef3fb;
      --muted: #9cabc0;
      --accent: #5ee0b5;
      --accent-2: #72a7ff;
      --danger: #ff8a8a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 10% 0%, rgba(94, 224, 181, .12), transparent 28rem),
        radial-gradient(circle at 100% 10%, rgba(114, 167, 255, .12), transparent 34rem),
        var(--bg);
    }
    main { width: min(1120px, calc(100% - 32px)); margin: 0 auto; padding: 52px 0 72px; }
    header { display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; margin-bottom: 28px; }
    h1 { font-size: clamp(30px, 5vw, 54px); line-height: 1.02; letter-spacing: -.04em; margin: 0 0 12px; }
    h2 { font-size: 17px; margin: 0; }
    p { margin: 0; }
    .subtitle { color: var(--muted); max-width: 700px; font-size: 16px; }
    .badge { border: 1px solid var(--line); background: rgba(18, 24, 36, .8); border-radius: 999px; padding: 7px 12px; color: var(--accent); white-space: nowrap; }
    .grid { display: grid; grid-template-columns: minmax(280px, 360px) minmax(0, 1fr); gap: 18px; }
    .panel { background: rgba(18, 24, 36, .9); border: 1px solid var(--line); border-radius: 18px; padding: 22px; box-shadow: 0 20px 60px rgba(0, 0, 0, .2); }
    form { display: grid; gap: 14px; }
    label { display: grid; gap: 7px; color: var(--muted); font-size: 13px; font-weight: 650; }
    input, select, textarea, button { font: inherit; }
    input, select, textarea {
      width: 100%; color: var(--text); background: var(--panel-2); border: 1px solid var(--line);
      border-radius: 10px; padding: 10px 12px; outline: none;
    }
    textarea { min-height: 124px; resize: vertical; }
    input:focus, select:focus, textarea:focus { border-color: var(--accent-2); box-shadow: 0 0 0 3px rgba(114, 167, 255, .12); }
    button { border: 0; border-radius: 10px; padding: 11px 16px; font-weight: 750; color: #06130f; background: var(--accent); cursor: pointer; }
    button:disabled { opacity: .55; cursor: wait; }
    .results-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 14px; }
    #status { color: var(--muted); font-size: 13px; }
    #status.error { color: var(--danger); }
    #results { display: grid; gap: 12px; }
    .empty { min-height: 260px; display: grid; place-items: center; text-align: center; color: var(--muted); border: 1px dashed var(--line); border-radius: 13px; padding: 28px; }
    article { background: var(--panel-2); border: 1px solid var(--line); border-radius: 13px; padding: 16px; }
    article h3 { margin: 0 0 4px; font-size: 16px; }
    .path { color: var(--accent-2); overflow-wrap: anywhere; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
    .excerpt { margin-top: 12px; color: #cbd5e4; white-space: pre-wrap; overflow-wrap: anywhere; }
    .meta { display: flex; gap: 12px; margin-top: 12px; color: var(--muted); font-size: 12px; }
    .snapshot { margin-top: 14px; color: var(--muted); font: 11px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap: anywhere; }
    .note { margin-top: 18px; color: var(--muted); font-size: 12px; }
    code { color: var(--accent); }
    @media (max-width: 760px) { header { display: block; } .badge { display: inline-block; margin-top: 18px; } .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Local GPU Wiki</h1>
        <p class="subtitle">Query the wire-compatible local test service against the pinned GPU Wiki corpus. This page is a development client; Runtime and Agents use the versioned JSON API directly.</p>
      </div>
      <span class="badge">local test double</span>
    </header>
    <div class="grid">
      <section class="panel">
        <form id="query-form">
          <label>Operator<input id="operator" value="reduction" maxlength="500" required></label>
          <label>DSL<select id="dsl"><option value="triton">Triton</option><option value="cuda">CUDA</option><option value="cutedsl">CuteDSL</option></select></label>
          <label>Hardware target<input id="hardware" value="nvidia-h100" maxlength="500" required></label>
          <label>Question<textarea id="query" maxlength="65536" required>How should I tile and schedule this reduction kernel?</textarea></label>
          <label>Bearer token (optional)<input id="token" type="password" autocomplete="off" placeholder="Only if enabled in local config"></label>
          <button id="submit" type="submit">Query Wiki</button>
        </form>
        <p class="note">Sends <code>POST /v1/knowledge/query</code> with synthetic valid Attempt identity fields. Production identity is always supplied by Runtime.</p>
      </section>
      <section class="panel">
        <div class="results-head"><h2>Knowledge matches</h2><span id="status">Ready</span></div>
        <div id="results"><div class="empty">Run a query to inspect structured GPU Wiki records.</div></div>
        <div id="snapshot" class="snapshot"></div>
      </section>
    </div>
  </main>
  <script>
    const form = document.querySelector('#query-form');
    const submit = document.querySelector('#submit');
    const statusNode = document.querySelector('#status');
    const results = document.querySelector('#results');
    const snapshot = document.querySelector('#snapshot');
    const zeros = '0'.repeat(64);
    const suffix = '0'.repeat(32);

    function textElement(name, className, text) {
      const node = document.createElement(name);
      if (className) node.className = className;
      node.textContent = text;
      return node;
    }

    function testContext() {
      return {
        schema_version: 1,
        service_api_version: 1,
        campaign_id: `campaign_${suffix}`,
        lineage_id: `lineage_${suffix}`,
        epoch_id: `epoch_${suffix}`,
        epoch_number: 1,
        attempt_id: `attempt_${suffix}`,
        branch: 'active',
        attempt_ordinal: 1,
        kernel_agent_revision_id: `agentrev_${suffix}`,
        operator: document.querySelector('#operator').value,
        dsl: document.querySelector('#dsl').value,
        hardware_target: document.querySelector('#hardware').value,
        evaluation_contract_digest: `sha256:${zeros}`,
        epoch_evidence_checkpoint_digest: `sha256:${zeros}`,
        attempt_evidence_digest: `sha256:${zeros}`
      };
    }

    function renderMatches(payload) {
      results.replaceChildren();
      const content = payload.content || {};
      const records = content.records && typeof content.records === 'object' ? content.records : {};
      const entries = Object.entries(records);
      if (!entries.length) {
        results.append(textElement('div', 'empty', 'No scoped matches found. Try a more specific operator or question.'));
      }
      for (const [recordId, record] of entries) {
        const card = document.createElement('article');
        card.append(textElement('h3', '', String(record.type || 'GPU Wiki record')));
        card.append(textElement('div', 'path', recordId));
        card.append(textElement('pre', 'excerpt', JSON.stringify(record.payload || record, null, 2)));
        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.append(textElement('span', '', 'Complete safe Record; preserve this stable ID if used'));
        card.append(meta);
        results.append(card);
      }
      snapshot.textContent = `snapshot: ${payload.snapshot_id || '-'}\ncontent: ${payload.content_digest || '-'}`;
      statusNode.textContent = `${entries.length} record${entries.length === 1 ? '' : 's'}`;
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      submit.disabled = true;
      statusNode.className = '';
      statusNode.textContent = 'Querying…';
      snapshot.textContent = '';
      const body = {
        ...testContext(),
        query: document.querySelector('#query').value
      };
      const token = document.querySelector('#token').value;
      const headers = {'content-type': 'application/json'};
      if (token) headers.authorization = `Bearer ${token}`;
      try {
        const response = await fetch('/v1/knowledge/query', {method: 'POST', headers, body: JSON.stringify(body)});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
        renderMatches(payload);
      } catch (error) {
        results.replaceChildren(textElement('div', 'empty', `Query failed: ${error.message || error}`));
        statusNode.className = 'error';
        statusNode.textContent = 'Failed';
      } finally {
        submit.disabled = false;
      }
    });
  </script>
</body>
</html>
""".encode()
