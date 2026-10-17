"""A local web UI for viewing and editing the config file.

Built on ``http.server`` (standard library only -- no extra dependencies). The
UI presents the config as a structured form (a providers list plus the agent
section) and writes it back to ``config.yaml`` as YAML. Values are stored
verbatim, so ``$NAME`` / ``${NAME}`` references round-trip intact (they are not
resolved here). The UI only touches the file on disk: a running ``agent chat``
session keeps the config it loaded at start, so edits apply to the *next*
session, not the current one.

A persona's prompt is the one value that does not live in the config file: the
config's ``SOUL.md`` entry only *seeds* ``personas/<name>/SOUL.md``, and the
stored file is what the agent reads. So the persona field shows the stored file
and saving it writes that file too (see :func:`light_harness.config.ensure_persona`).

Endpoints:
- ``GET  /``            -> the HTML page
- ``GET  /api/config``  -> ``{"exists", "path", "text", "data"}`` (``data`` shows
                           each persona's stored ``SOUL.md``; ``text`` is the
                           config file verbatim; a file that cannot be read or
                           is not UTF-8 gets a 500 ``{"error"}`` instead)
- ``GET  /api/tools``   -> ``{"tools": [...], "forgejo_tools": [...]}`` (core and Forgejo
                           tool names)
- ``POST /api/config``  -> JSON body naming any of ``providers``, ``agents``,
                           ``personas``, ``interaction``, ``webhooks``; only the sections the
                           body sends are rewritten (as YAML), the others are kept
                           exactly as they are on disk, and a persona entry that
                           sends ``SOUL.md`` rewrites that persona's file
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import tempfile
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

from .config import (
    SOUL_FILENAME,
    config_file_path,
    has_env_reference,
    load_yaml,
    persona_dir,
    persona_soul_text,
    resolve_env,
    write_persona_soul,
)
from .httputil import (
    BodyError,
    declared_body,
    host_allow_list,
    host_is_allowed,
    read_body,
    secret_matches,
)
from .tools.builtin import CORE_TOOL_NAMES
from .tools.forgejo import FORGEJO_TOOL_NAMES

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

_PAGE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="agent-token" content="__AGENT_TOKEN__">
<title>light-harness</title>
<style>
"""
_PAGE_CSS = """  :root {
    --bg: #f4f5f7; --card: #fff; --fg: #1f2328; --muted: #6b7280;
    --border: #e5e7eb; --accent: #2563eb; --accent-fg: #fff;
    --ok: #16a34a; --err: #dc2626; --side: #1f2430;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg); line-height: 1.5;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  .app { display: flex; min-height: 100vh; }
  .sidebar { width: 220px; flex-shrink: 0; background: var(--side); color: #cdd3df;
    padding: 20px 16px; }
  .brand { font-weight: 700; font-size: 16px; color: #fff; margin-bottom: 24px; }
  .nav a { display: block; padding: 10px 12px; border-radius: 8px; margin-bottom: 4px;
    color: #cdd3df; text-decoration: none; font-size: 14px; }
  .nav a.active { background: var(--accent); color: #fff; font-weight: 600; }
  .nav a:hover { background: rgba(255,255,255,.08); }
  .content { flex: 1; padding: 24px 28px 64px; }
  .topbar { display: flex; align-items: center; gap: 12px; margin-bottom: 20px;
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 12px 14px; }
  .path-field { flex: 1; min-width: 0; }
  .path-field .lbl { display: block; font-size: 12px; color: var(--muted); margin-bottom: 2px; }
  .path-field input { width: 100%; font-family: var(--mono); font-size: 13px; padding: 7px 9px;
    border: 1px solid var(--border); border-radius: 7px; background: #f7f8fa; color: var(--fg); }
  .topbar-actions { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
  button { font: inherit; font-size: 14px; cursor: pointer; padding: 8px 14px;
    border: 1px solid var(--border); border-radius: 8px; background: #fff; color: var(--fg); }
  button.primary { background: var(--accent); border-color: var(--accent);
    color: var(--accent-fg); font-weight: 600; }
  button.icon { padding: 4px 8px; line-height: 1; font-size: 16px; }
  button:disabled { opacity: .5; cursor: default; }
  #status { font-size: 13px; color: var(--muted); }
  #status.ok { color: var(--ok); }
  #status.err { color: var(--err); }
  .section { margin-bottom: 24px; }
  .section-head { display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 12px; }
  .section-head h2 { font-size: 17px; margin: 0; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    box-shadow: 0 1px 2px rgba(0,0,0,.05); padding: 16px; }
  .provider { margin-bottom: 14px; }
  .prov-head { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
  .prov-name { flex: 1; font-weight: 600; font-size: 14px; padding: 8px 10px;
    border: 1px solid var(--border); border-radius: 8px; }
  .grid { display: grid; gap: 10px; margin-bottom: 12px; }
  .grid.three { grid-template-columns: 2fr 2fr 1fr; }
  .grid.two { grid-template-columns: 1fr 1fr; }
  label { display: block; font-size: 13px; color: var(--muted); }
  label span { display: block; margin-bottom: 3px; }
  input, select { width: 100%; font: inherit; font-size: 14px; padding: 8px 10px;
    border: 1px solid var(--border); border-radius: 8px; background: #fff; color: var(--fg); }
  .models { border-top: 1px dashed var(--border); padding-top: 10px; }
  .models-head { display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 8px; }
  .models-head span { font-size: 13px; color: var(--muted); }
  .model-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
  .tools-block { border-top: 1px dashed var(--border); padding-top: 12px; }
  .tools-label { font-size: 13px; color: var(--muted); margin-bottom: 8px; }
  .tools-columns { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 4px 48px; }
  .tools-col { min-width: 0; }
  .tool-check { display: flex; align-items: center; gap: 8px; font-size: 14px;
    color: var(--fg); margin-bottom: 6px; }
  .tool-check input { width: auto; }
  .tool-list { margin: 4px 0 10px 4px; }
  .empty { color: var(--muted); font-size: 14px; font-style: italic; }
  .agent { margin-bottom: 14px; }
  .agent-head { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
  .agent-name { flex: 1; font-weight: 600; font-size: 14px; padding: 8px 10px;
    border: 1px solid var(--border); border-radius: 8px; }
  .persona { margin-bottom: 14px; }
  .persona-head { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
  .persona-name { flex: 1; font-weight: 600; font-size: 14px; padding: 8px 10px;
    border: 1px solid var(--border); border-radius: 8px; }
  textarea { width: 100%; font: inherit; font-size: 14px; padding: 8px 10px;
    border: 1px solid var(--border); border-radius: 8px; background: #fff;
    color: var(--fg); font-family: var(--mono); min-height: 90px; resize: vertical; }
  .tool-all-row { display: flex; align-items: center; gap: 8px; font-size: 13px;
    color: var(--muted); margin-bottom: 8px; }
  .tool-all-row input { width: auto; }
  .server-check { display: flex; align-items: center; gap: 8px; font-size: 14px;
    color: var(--fg); margin: 12px 0 8px; }
  .server-check input { width: auto; }
  .page { display: none; }
  .page.active { display: block; }
  .webhook { margin-bottom: 14px; }
  .webhook-head { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
  .webhook-path { flex: 1; font-weight: 600; font-size: 14px; padding: 8px 10px;
    border: 1px solid var(--border); border-radius: 8px; }
</style>
</head>
"""
_PAGE_BODY = """<body>
<div class="app">
  <aside class="sidebar">
    <div class="brand">light-harness</div>
    <nav class="nav">
      <a href="#providers" data-page="providers">Providers</a>
      <a href="#agents" data-page="agents">Agents</a>
      <a href="#personas" data-page="personas">Personas</a>
      <a href="#chat" data-page="chat">Chat</a>
      <a href="#webhooks" data-page="webhooks">Webhooks</a>
    </nav>
  </aside>
  <main class="content">
    <div class="topbar">
      <div class="path-field">
        <span class="lbl">Config file</span>
        <input id="path" readonly value="">
      </div>
      <div class="topbar-actions">
        <span id="status">Loading...</span>
        <button id="refresh" type="button">Refresh</button>
        <button id="save" class="primary" type="button">Save</button>
      </div>
    </div>
    <section class="section page" id="page-providers">
      <div class="section-head">
        <h2>Providers</h2>
        <button id="add-provider" class="ghost" type="button">+ Add provider</button>
      </div>
      <div id="provider-list"></div>
    </section>
    <section class="section page" id="page-agents">
      <div class="section-head">
        <h2>Agents</h2>
        <button id="add-agent" class="ghost" type="button">+ Add agent</button>
      </div>
      <div id="agent-list"></div>
    </section>
    <section class="section page" id="page-personas">
      <div class="section-head">
        <h2>Personas</h2>
        <button id="add-persona" class="ghost" type="button">+ Add persona</button>
      </div>
      <div id="persona-list"></div>
    </section>
    <section class="section page" id="page-chat">
      <div class="section-head"><h2>Chat</h2></div>
      <div id="interaction-section"></div>
    </section>
    <section class="section page" id="page-webhooks">
      <div class="card server-card" style="margin-bottom:18px">
        <div class="grid three">
          <label><span>Bind host (--host overrides)</span>
            <input id="ws-host" placeholder="127.0.0.1"></label>
          <label><span>Bind port (--port overrides)</span>
            <input id="ws-port" placeholder="8787"></label>
          <label><span>Allowed hosts (comma-separated; --allowed-host overrides)</span>
            <input id="ws-hosts" placeholder="box.local, hooks.example.com"></label>
        </div>
        <label class="server-check"><input type="checkbox" id="ws-output-log">
          Output log: print a line for each webhook call naming its session
          log file (--output-log overrides)</label>
        <div style="font-size:13px;color:var(--muted)">Server settings take effect when
          <code>agent webhooks</code> is (re)started.</div>
      </div>
      <div class="section-head">
        <h2>Webhooks</h2>
        <button id="add-webhook" class="ghost" type="button">+ Add webhook</button>
      </div>
      <div id="webhook-list"></div>
    </section>
  </main>
</div>
<script>
"""
_PAGE_JS = """  var state = { data: null, builtinTools: [], dirty: false, fileDirty: false,
    busy: false, lastText: "" };

  var API_TOKEN = document.querySelector('meta[name="agent-token"]').content;
  function apiFetch(url, opts) {
    opts = opts || {};
    opts.headers = Object.assign({}, opts.headers, { "X-Agent-Token": API_TOKEN });
    return fetch(url, opts);
  }
  var pathEl = document.getElementById("path");
  var statusEl = document.getElementById("status");
  var providerList = document.getElementById("provider-list");
  var agentList = document.getElementById("agent-list");
  var personaList = document.getElementById("persona-list");
  var interactionSection = document.getElementById("interaction-section");
  var saveBtn = document.getElementById("save");
  var refreshBtn = document.getElementById("refresh");
  var addProviderBtn = document.getElementById("add-provider");
  var addAgentBtn = document.getElementById("add-agent");
  var addPersonaBtn = document.getElementById("add-persona");
  var wsHost = document.getElementById("ws-host");
  var wsPort = document.getElementById("ws-port");
  var wsHosts = document.getElementById("ws-hosts");
  var wsOutputLog = document.getElementById("ws-output-log");
  var webhookList = document.getElementById("webhook-list");
  var addWebhookBtn = document.getElementById("add-webhook");

  function esc(v) {
    return String(v == null ? "" : v).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function setStatus(msg, kind) { statusEl.textContent = msg; statusEl.className = kind || ""; }

  function deriveStatus() {
    if (state.busy) return setStatus("Saving...");
    if (state.fileDirty) return setStatus("File changed on disk - click Refresh to reload", "err");
    if (state.dirty) return setStatus("Unsaved changes");
    setStatus("Ready");
  }

  function field(label, cls, value, placeholder) {
    return '<label><span>' + label + '</span><input class="' + cls + '" value="' + esc(value) +
      '" placeholder="' + placeholder + '"></label>';
  }

  function modelRow(m) {
    var h = [];
    h.push('<div class="model-row">');
    h.push('<input class="model-name" value="' + esc(m) + '" placeholder="model name">');
    h.push('<button class="icon remove-model" type="button" title="Remove model">&times;</button>');
    h.push('</div>');
    return h.join("");
  }

  function providerCard(name, p) {
    p = p || {};
    var models = (p.models || []).map(modelRow).join("");
    var h = [];
    h.push('<div class="provider card" data-provider>');
    h.push('<div class="prov-head">');
    h.push('<input class="prov-name" value="' + esc(name) + '" placeholder="provider name">');
    h.push('<button class="icon remove-prov" type="button" title="Remove provider"' +
      '>&times;</button>');
    h.push('</div>');
    h.push('<div class="grid three">');
    h.push(field("Base URL", "prov-base_url", p.base_url || "", "http://127.0.0.1:8000/v1"));
    h.push(field("API key", "prov-api_key", p.api_key || "", "$MY_API_KEY"));
    h.push(field("Timeout (s)", "prov-timeout", p.timeout == null ? "" : p.timeout, "60"));
    h.push('</div>');
    h.push('<div class="models">');
    h.push('<div class="models-head"><span>Models</span>' +
      '<button class="ghost add-model" type="button">+ Add model</button></div>');
    h.push('<div class="model-rows">' + models + '</div>');
    h.push('</div>');
    h.push('</div>');
    return h.join("");
  }

  function providerNamesFromDom() {
    return Array.from(providerList.querySelectorAll(".prov-name"))
      .map(function (el) { return el.value.trim(); }).filter(Boolean);
  }

  function personaNamesFromDom() {
    return Array.from(personaList.querySelectorAll(".persona-name"))
      .map(function (el) { return el.value.trim(); }).filter(Boolean);
  }

  function agentNamesFromDom() {
    return Array.from(agentList.querySelectorAll(".agent-name"))
      .map(function (el) { return el.value.trim(); }).filter(Boolean);
  }

  function providerOptions(selected) {
    var names = providerNamesFromDom();
    if (selected && names.indexOf(selected) === -1) names.unshift(selected);
    return names.map(function (n) {
      var sel = n === selected ? " selected" : "";
      return '<option value="' + esc(n) + '"' + sel + ">" + esc(n) + "</option>";
    }).join("");
  }

  function personaOptions(selected) {
    var names = personaNamesFromDom();
    if (selected && names.indexOf(selected) === -1) names.unshift(selected);
    var opts = ['<option value="">(none)</option>'];
    names.forEach(function (n) {
      var sel = n === selected ? " selected" : "";
      opts.push('<option value="' + esc(n) + '"' + sel + ">" + esc(n) + "</option>");
    });
    return opts.join("");
  }

  function toolsBlock(tools) {
    var allChecked = tools == null;
    var selected = Array.isArray(tools) ? tools : [];
    function boxes(names) {
      return names.map(function (t) {
        var checked = allChecked || selected.indexOf(t) !== -1;
        return '<label class="tool-check"><input type="checkbox" class="tool" value="' +
          esc(t) + '"' + (checked ? " checked" : "") + "> " + esc(t) + "</label>";
      }).join("");
    }
    var known = state.builtinTools.concat(state.forgejoTools);
    var other = allChecked ? [] : selected.filter(function (t) {
      return known.indexOf(t) === -1;
    });
    function column(title, allLabel, names) {
      var every = names.length > 0 && (allChecked || names.every(function (t) {
        return selected.indexOf(t) !== -1;
      }));
      return '<div class="tools-col"><div class="tools-label">' + title + "</div>" +
        '<label class="tool-all-row"><input type="checkbox" class="tool-group-all"' +
        (every ? " checked" : "") + "> " + allLabel + "</label>" +
        '<div class="tool-list tool-group">' + boxes(names) + "</div></div>";
    }
    var h = [];
    h.push('<div class="tools-block">');
    h.push('<label class="tool-all-row"><input type="checkbox" class="tool-all"' +
      (allChecked ? " checked" : "") + "> All tools (default)</label>");
    h.push('<div class="tools-columns">');
    h.push(column("Standard tools", "All standard tools", state.builtinTools));
    if (state.forgejoTools.length) {
      h.push(column("Forgejo tools", "All Forgejo tools", state.forgejoTools));
    }
    h.push("</div>");
    h.push('<label><span>Other tools (comma-separated)</span>' +
      '<input class="agent-other-tools" value="' + esc(other.join(", ")) + '"></label>');
    h.push("</div>");
    return h.join("");
  }

  function agentCard(a) {
    a = a || {};
    var h = [];
    h.push('<div class="card agent" data-agent>');
    h.push('<div class="agent-head">');
    h.push('<input class="agent-name" value="' + esc(a.name || "") + '" placeholder="agent name">');
    h.push('<button class="icon remove-agent" type="button" title="Remove agent">&times;</button>');
    h.push('</div>');
    h.push('<div class="grid three">');
    h.push('<label><span>Provider</span><select class="agent-provider">' +
      providerOptions(a.provider || "") + "</select></label>");
    h.push(field("Model", "agent-model", a.model || "", "model name"));
    h.push('<label><span>Persona</span><select class="agent-persona">' +
      personaOptions(a.persona || "") + "</select></label>");
    h.push("</div>");
    h.push(field("Workspace (absolute directory; required for the file tools)",
      "agent-workspace", a.workspace || "", "/path/to/project"));
    h.push(toolsBlock(a.tools));
    h.push("</div>");
    return h.join("");
  }

  function personaCard(name, p) {
    p = p || {};
    var h = [];
    h.push('<div class="card persona" data-persona>');
    h.push('<div class="persona-head">');
    h.push('<input class="persona-name" value="' + esc(name) + '" placeholder="persona name">');
    h.push('<button class="icon remove-persona" type="button" ' +
      'title="Remove persona">&times;</button>');
    h.push('</div>');
    h.push('<label><span>SOUL.md (system prompt; saved to personas/&lt;name&gt;/SOUL.md)</span>' +
      '<textarea class="persona-soul" placeholder="You are ...">' +
      esc(p["SOUL.md"] || "") + "</textarea></label>");
    h.push("</div>");
    return h.join("");
  }

  function agentOptions(selected) {
    var names = agentNamesFromDom();
    if (selected && names.indexOf(selected) === -1) names.unshift(selected);
    return names.map(function (n) {
      var sel = n === selected ? " selected" : "";
      return '<option value="' + esc(n) + '"' + sel + ">" + esc(n) + "</option>";
    }).join("");
  }

  function webhookCard(w) {
    w = w || {};
    var h = [];
    h.push('<div class="card webhook" data-webhook>');
    h.push('<div class="webhook-head">');
    h.push('<input class="webhook-path" value="' + esc(w.path || "") +
      '" placeholder="/hooks/name">');
    h.push('<button class="icon remove-webhook" type="button" ' +
      'title="Remove webhook">&times;</button>');
    h.push('</div>');
    h.push('<div class="grid two">');
    h.push('<label><span>Agent</span><select class="webhook-agent">' +
      agentOptions(w.agent || "") + "</select></label>");
    h.push(field("Secret (bearer token; empty = unauthenticated)", "webhook-secret",
      w.secret || "", "$MY_WEBHOOK_SECRET"));
    h.push('</div>');
    h.push('<label><span>System prompt (optional; overrides the default agent prompt)</span>' +
      '<textarea class="webhook-prompt" placeholder="You are ...">' +
      esc(w.system_prompt || "") + "</textarea></label>");
    h.push("</div>");
    return h.join("");
  }

  function showPage() {
    var page = location.hash.replace("#", "");
    if (!document.getElementById("page-" + page)) page = "providers";
    document.querySelectorAll(".page").forEach(function (el) {
      el.classList.toggle("active", el.id === "page-" + page);
    });
    document.querySelectorAll(".nav a").forEach(function (a) {
      a.classList.toggle("active", a.dataset.page === page);
    });
  }

  function renderInteraction() {
    var sel = interactionSection.querySelector(".interaction-agent");
    var current = sel ? sel.value : ((state.data.interaction && state.data.interaction.chat)
      ? state.data.interaction.chat.agent : "");
    var names = agentNamesFromDom();
    if (current && names.indexOf(current) === -1) names.unshift(current);
    var opts = names.map(function (n) {
      var s = n === current ? " selected" : "";
      return '<option value="' + esc(n) + '"' + s + ">" + esc(n) + "</option>";
    }).join("") || '<option value=""></option>';
    interactionSection.innerHTML =
      '<div class="card"><div class="grid two"><label><span>Chat uses agent</span>' +
      '<select class="interaction-agent">' + opts + "</select></label></div></div>";
  }

  function render(data) {
    state.data = data || {};
    var providers = state.data.providers || {};
    var providerNames = Object.keys(providers);
    providerList.innerHTML = providerNames.length
      ? providerNames.map(function (n) { return providerCard(n, providers[n]); }).join("")
      : '<p class="empty">No providers defined. Add one to configure endpoints.</p>';
    var personas = state.data.personas || {};
    var personaNames = Object.keys(personas);
    personaList.innerHTML = personaNames.length
      ? personaNames.map(function (n) { return personaCard(n, personas[n]); }).join("")
      : '<p class="empty">No personas defined. Add one to customize an agent.</p>';
    var agents = state.data.agents || [];
    agentList.innerHTML = agents.length
      ? agents.map(function (a) { return agentCard(a); }).join("")
      : '<p class="empty">No agents defined. Add one to configure the chat loop.</p>';
    renderInteraction();
    var ws = state.data.webhook_server || {};
    wsHost.value = ws.host || "";
    wsPort.value = ws.port == null ? "" : ws.port;
    wsHosts.value = Array.isArray(ws.allowed_hosts) ? ws.allowed_hosts.join(", ") : "";
    wsOutputLog.checked = ws.output_log === true;
    var webhooks = state.data.webhooks || [];
    webhookList.innerHTML = webhooks.length
      ? webhooks.map(function (w) { return webhookCard(w); }).join("")
      : '<p class="empty">No webhooks defined. Add one to trigger an agent over HTTP.</p>';
    syncOtherToolsDisabled();
  }

  // "Other tools" only applies to an explicit list; "All tools" saves no list at all.
  function syncOtherToolsDisabled() {
    document.querySelectorAll(".agent").forEach(function (card) {
      var all = card.querySelector(".tool-all");
      var other = card.querySelector(".agent-other-tools");
      if (other) other.disabled = all ? all.checked : false;
    });
  }

  function refreshDependentSelects() {
    document.querySelectorAll(".agent").forEach(function (card) {
      var provSel = card.querySelector(".agent-provider");
      if (provSel) provSel.innerHTML = providerOptions(provSel.value);
      var personaSel = card.querySelector(".agent-persona");
      if (personaSel) personaSel.innerHTML = personaOptions(personaSel.value);
    });
    renderInteraction();
    document.querySelectorAll(".webhook-agent").forEach(function (sel) {
      sel.innerHTML = agentOptions(sel.value);
    });
  }

  function collectProviders() {
    return Array.from(providerList.querySelectorAll("[data-provider]")).map(function (card) {
      return {
        name: card.querySelector(".prov-name").value.trim(),
        base_url: card.querySelector(".prov-base_url").value.trim(),
        api_key: card.querySelector(".prov-api_key").value.trim(),
        timeout: card.querySelector(".prov-timeout").value.trim(),
        models: Array.from(card.querySelectorAll(".model-name"))
          .map(function (m) { return m.value.trim(); }).filter(Boolean)
      };
    });
  }

  function collectAgents() {
    return Array.from(agentList.querySelectorAll("[data-agent]")).map(function (card) {
      var tools = null;
      var allEl = card.querySelector(".tool-all");
      if (!allEl || !allEl.checked) {
        tools = Array.from(card.querySelectorAll(".tool-list input:checked"))
          .map(function (c) { return c.value; });
        var otherEl = card.querySelector(".agent-other-tools");
        var other = otherEl ? otherEl.value : "";
        other = other.split(",").map(function (s) { return s.trim(); }).filter(Boolean);
        tools = tools.concat(other);
      }
      return {
        name: (card.querySelector(".agent-name") || {}).value.trim(),
        provider: (card.querySelector(".agent-provider") || {}).value || "",
        model: (card.querySelector(".agent-model") || {}).value.trim(),
        persona: (card.querySelector(".agent-persona") || {}).value || "",
        workspace: ((card.querySelector(".agent-workspace") || {}).value || "").trim(),
        tools: tools
      };
    });
  }

  function collectPersonas() {
    return Array.from(personaList.querySelectorAll("[data-persona]")).map(function (card) {
      var soul = card.querySelector(".persona-soul");
      return {
        name: (card.querySelector(".persona-name") || {}).value.trim(),
        "SOUL.md": soul ? soul.value : ""
      };
    });
  }

  function collectInteraction() {
    var sel = interactionSection.querySelector(".interaction-agent");
    var agent = sel ? sel.value : "";
    return { chat: { agent: agent } };
  }

  function collectWebhookServer() {
    return {
      host: wsHost.value.trim(),
      port: wsPort.value.trim(),
      allowed_hosts: wsHosts.value.split(",").map(function (s) { return s.trim(); })
        .filter(Boolean),
      output_log: wsOutputLog.checked
    };
  }

  function collectWebhooks() {
    return Array.from(webhookList.querySelectorAll("[data-webhook]")).map(function (card) {
      return {
        path: card.querySelector(".webhook-path").value.trim(),
        agent: card.querySelector(".webhook-agent").value || "",
        secret: card.querySelector(".webhook-secret").value.trim(),
        system_prompt: card.querySelector(".webhook-prompt").value
      };
    });
  }

  async function fetchConfig() {
    var res = await apiFetch("/api/config");
    var data = await res.json();
    if (!res.ok) throw new Error(data.error || "HTTP " + res.status);
    pathEl.value = data.path;
    state.lastText = data.text == null ? "" : data.text;
    state.fileDirty = false;
    render(data.data || {});
    return data;
  }

  async function loadConfig() {
    try {
      var data = await fetchConfig();
      if (!data.exists && state.lastText === "") {
        setStatus("No config file yet - edit and Save to create it");
      } else { deriveStatus(); }
    } catch (e) { setStatus("Failed to load: " + e, "err"); }
  }

  async function saveConfig() {
    state.busy = true;
    deriveStatus();
    try {
      var body = JSON.stringify({
        providers: collectProviders(),
        agents: collectAgents(),
        personas: collectPersonas(),
        interaction: collectInteraction(),
        webhooks: collectWebhooks(),
        webhook_server: collectWebhookServer()
      });
      var res = await apiFetch("/api/config", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: body
      });
      var data = await res.json();
      if (res.ok) {
        state.dirty = false;
        state.fileDirty = false;
        await fetchConfig();
        setStatus("Saved to " + data.path, "ok");
      } else { setStatus(data.error || "Save failed", "err"); }
    } catch (e) { setStatus("Save failed: " + e, "err"); }
    finally { state.busy = false; }
  }

  function markDirty() { state.dirty = true; deriveStatus(); }

  providerList.addEventListener("click", function (e) {
    var t = e.target;
    if (t.classList.contains("remove-prov")) {
      t.closest("[data-provider]").remove(); markDirty(); refreshDependentSelects();
    }
    else if (t.classList.contains("remove-model")) {
      t.closest(".model-row").remove(); markDirty();
    }
    else if (t.classList.contains("add-model")) {
      var rows = t.closest(".models").querySelector(".model-rows");
      rows.insertAdjacentHTML("beforeend", modelRow(""));
      markDirty();
    }
  });
  providerList.addEventListener("input", function (e) {
    markDirty();
    if (e.target && e.target.classList.contains("prov-name")) refreshDependentSelects();
  });

  agentList.addEventListener("click", function (e) {
    var t = e.target;
    if (t.classList.contains("remove-agent")) {
      t.closest("[data-agent]").remove(); markDirty(); refreshDependentSelects();
    }
  });
  agentList.addEventListener("input", function (e) {
    markDirty();
    if (e.target && e.target.classList.contains("agent-name")) refreshDependentSelects();
  });
  agentList.addEventListener("change", function (e) {
    var t = e.target;
    markDirty();
    var block = t && t.closest ? t.closest(".tools-block") : null;
    if (!block) return;
    var allEl = block.querySelector(".tool-all");
    if (t.classList.contains("tool-all")) {
      if (t.checked) {
        block.querySelectorAll(".tool-list input, .tool-group-all").forEach(function (el) {
          el.checked = true;
        });
      }
    } else if (t.classList.contains("tool-group-all")) {
      t.closest(".tools-col").querySelectorAll(".tool-group input").forEach(function (el) {
        el.checked = t.checked;
      });
      if (!t.checked) allEl.checked = false;
    } else if (t.classList.contains("tool")) {
      if (!t.checked) allEl.checked = false;
      var col = t.closest(".tools-col");
      var boxes = Array.from(col.querySelectorAll(".tool-group input"));
      col.querySelector(".tool-group-all").checked = boxes.every(function (el) {
        return el.checked;
      });
    }
    syncOtherToolsDisabled();
  });

  personaList.addEventListener("click", function (e) {
    var t = e.target;
    if (t.classList.contains("remove-persona")) {
      t.closest("[data-persona]").remove(); markDirty(); refreshDependentSelects();
    }
  });
  personaList.addEventListener("input", function (e) {
    markDirty();
    if (e.target && e.target.classList.contains("persona-name")) refreshDependentSelects();
  });

  interactionSection.addEventListener("change", markDirty);

  webhookList.addEventListener("click", function (e) {
    if (e.target.classList.contains("remove-webhook")) {
      e.target.closest("[data-webhook]").remove(); markDirty();
    }
  });
  webhookList.addEventListener("input", markDirty);
  [wsHost, wsPort, wsHosts].forEach(function (el) { el.addEventListener("input", markDirty); });
  webhookList.addEventListener("change", markDirty);
  addWebhookBtn.addEventListener("click", function () {
    var empty = webhookList.querySelector(".empty");
    if (empty) empty.remove();
    webhookList.insertAdjacentHTML("beforeend", webhookCard({}));
    markDirty();
  });
  window.addEventListener("hashchange", showPage);
  showPage();

  addProviderBtn.addEventListener("click", function () {
    var empty = providerList.querySelector(".empty");
    if (empty) empty.remove();
    providerList.insertAdjacentHTML("beforeend", providerCard("", {}));
    markDirty(); refreshDependentSelects();
  });
  addAgentBtn.addEventListener("click", function () {
    var empty = agentList.querySelector(".empty");
    if (empty) empty.remove();
    agentList.insertAdjacentHTML("beforeend", agentCard({ name: "" }));
    markDirty(); refreshDependentSelects();
  });
  addPersonaBtn.addEventListener("click", function () {
    var empty = personaList.querySelector(".empty");
    if (empty) empty.remove();
    personaList.insertAdjacentHTML("beforeend", personaCard("", {}));
    markDirty(); refreshDependentSelects();
  });
  saveBtn.addEventListener("click", saveConfig);
  refreshBtn.addEventListener("click", function () {
    state.dirty = false; state.fileDirty = false; loadConfig();
  });

  setInterval(async function () {
    try {
      var res = await apiFetch("/api/config");
      if (!res.ok) return;
      var data = await res.json();
      var text = data.text == null ? "" : data.text;
      if (text === state.lastText) return;
      if (state.dirty) { state.fileDirty = true; deriveStatus(); }
      else { state.lastText = text; render(data.data || {}); setStatus("Updated from file", "ok"); }
    } catch (e) { /* transient */ }
  }, 3000);

  (async function init() {
    try {
      var res = await apiFetch("/api/tools");
      var tools = await res.json();
      state.builtinTools = tools.tools || [];
      state.forgejoTools = tools.forgejo_tools || [];
    } catch (e) { state.builtinTools = []; state.forgejoTools = []; }
    loadConfig();
  })();
</script>
</body>
</html>
"""
_PAGE = _PAGE_HEAD + _PAGE_CSS + _PAGE_BODY + _PAGE_JS


def _yaml_number(text: str) -> int | float | str | None:
    """Convert a form string to a YAML number where possible.

    ``"60"`` -> ``60``, ``"60.5"`` -> ``60.5``; an empty string -> ``None``;
    anything non-numeric is returned as-is (e.g. a ``$TIMEOUT`` env ref).
    """
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return text
    return int(value) if value.is_integer() else value


_MANAGED_SECTIONS: tuple[str, ...] = (
    "providers",
    "agents",
    "personas",
    "interaction",
    "webhooks",
    "webhook_server",
)


def _section_sent(payload: dict[str, Any], section: str) -> bool:
    """True when the body actually *sends* ``section``.

    An absent key -- or one sent as ``null`` -- means "leave this section as it
    is on disk". Reading it as "clear this section" is what let a body that named
    nothing wipe every managed section.
    """
    return section in payload and payload[section] is not None


def _form_entries(raw: Any, section: str) -> list[Any]:
    """The form's shape for a list-shaped section: a list of entry objects.

    ``providers`` and ``personas`` are mappings in the *file*; sending that shape
    here is a client guessing wrong, so it is rejected instead of being parsed
    into an empty section.
    """
    if not isinstance(raw, list):
        raise ValueError(
            f"{section}: expected a list of entries (the file's own mapping shape "
            "is not what this API takes); omit the key to leave the section unchanged."
        )
    return raw


def _form_entry(entry: Any, section: str, index: int) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"{section}[{index}]: expected an object.")
    return entry


def _form_name(entry: dict[str, Any], section: str, index: int) -> str:
    """A trimmed, non-empty ``name``, or an error naming the entry that lacks one."""
    raw_name = entry.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError(f"{section}[{index}]: each entry requires a non-empty name.")
    return raw_name.strip()


def _form_text(entry: dict[str, Any], key: str, where: str) -> str:
    """An optional trimmed string field (absent or ``null`` -> empty)."""
    value = entry.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{where}.{key}: expected a string.")
    return value.strip()


def _form_names(value: Any, where: str) -> list[str]:
    """A list of non-empty strings, as ``models`` and ``tools`` are."""
    if not isinstance(value, list):
        raise ValueError(f"{where}: expected a list.")
    names: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{where}: every entry must be a string.")
        if item.strip():
            names.append(item.strip())
    return names


def _providers_from_form(raw: Any) -> dict[str, Any]:
    """Build the file's ``providers:`` mapping from the form's provider list."""
    providers: dict[str, Any] = {}
    for index, any_entry in enumerate(_form_entries(raw, "providers")):
        entry = _form_entry(any_entry, "providers", index)
        name = _form_name(entry, "providers", index)
        spec: dict[str, Any] = {}
        for key in ("base_url", "api_key"):
            text = _form_text(entry, key, f"providers.{name}")
            if text:
                spec[key] = text
        timeout = entry.get("timeout")
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (str, int, float)):
                raise ValueError(
                    f"providers.{name}.timeout: expected a number or a $NAME reference."
                )
            number = _yaml_number(str(timeout).strip())
            if number is not None:
                spec["timeout"] = number
        if entry.get("models") is not None:
            models = _form_names(entry["models"], f"providers.{name}.models")
            if models:
                spec["models"] = models
        providers[name] = spec
    return providers


def _agents_from_form(raw: Any) -> list[dict[str, Any]]:
    """Build the file's ``agents:`` list from the form's agent list."""
    agents: list[dict[str, Any]] = []
    for index, any_entry in enumerate(_form_entries(raw, "agents")):
        entry = _form_entry(any_entry, "agents", index)
        name = _form_name(entry, "agents", index)
        agent: dict[str, Any] = {"name": name}
        for key in ("provider", "model", "persona", "workspace"):
            text = _form_text(entry, key, f"agents.{name}")
            if text:
                agent[key] = text
        if entry.get("tools") is not None:
            # ``[]`` means "no tools"; an absent key means the built-in default.
            agent["tools"] = _form_names(entry["tools"], f"agents.{name}.tools")
        agents.append(agent)
    return agents


def _personas_from_form(raw: Any, current: Any) -> dict[str, Any]:
    """Build the file's ``personas:`` mapping from the form's persona list.

    The form has a field for ``SOUL.md`` only, so a persona's other files (the
    section is a name -> {file: content} mapping) are carried over from the file
    rather than dropped by the save. A persona name becomes a directory under
    ``LIGHT_HARNESS_HOME`` — the save also writes that persona's ``SOUL.md`` — so a
    name that is not a single path segment is refused here instead of reaching disk.
    """
    existing = current if isinstance(current, dict) else {}
    personas: dict[str, Any] = {}
    for index, any_entry in enumerate(_form_entries(raw, "personas")):
        entry = _form_entry(any_entry, "personas", index)
        name = _form_name(entry, "personas", index)
        try:
            persona_dir(name)  # the name becomes a directory under the home
        except ValueError as exc:
            raise ValueError(f"personas[{index}]: {exc}") from exc
        files: dict[str, str] = {}
        carried = existing.get(name)
        if isinstance(carried, dict):
            for key, value in carried.items():
                if key != SOUL_FILENAME and isinstance(key, str) and isinstance(value, str):
                    files[key] = value
        for key, value in entry.items():
            if key == "name" or value is None:
                continue
            if not isinstance(value, str):
                raise ValueError(f"personas.{name}.{key}: expected a string.")
            if value.strip():
                files[key] = value.strip()
        if files:
            personas[name] = files
    return personas


def _persona_souls_from_form(raw: Any) -> dict[str, str]:
    """The persona prompts the body sends, keyed by persona name.

    These are what get written to ``personas/<name>/SOUL.md`` — the file the agent
    actually reads, where the config value is only its first-run seed. An entry that
    omits ``SOUL.md`` (or sends it as ``null``) is left out: "not sent" means "leave
    the stored prompt alone", while ``""`` means "clear it".
    """
    souls: dict[str, str] = {}
    for index, any_entry in enumerate(_form_entries(raw, "personas")):
        entry = _form_entry(any_entry, "personas", index)
        name = _form_name(entry, "personas", index)
        value = entry.get(SOUL_FILENAME)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"personas.{name}.{SOUL_FILENAME}: expected a string.")
        souls[name] = value
    return souls


def _effective_persona_souls(data: dict[str, Any]) -> None:
    """Point each persona's ``SOUL.md`` in *data* at that persona's stored file.

    :meth:`light_harness.config.Config.system_prompt_for` reads
    ``personas/<name>/SOUL.md``; the config value only seeds it on first creation.
    Showing the config value is what made a persona edit look saved — 200, written to
    ``config.yaml`` — while the prompt the agent uses never changed.

    A config value that is still a ``$NAME`` reference is left as it is: the stored
    file holds the *resolved* text, and showing it would turn the next save into a
    literal rewrite of the reference."""
    personas = data.get("personas")
    if not isinstance(personas, dict):
        return
    for name, entry in personas.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            continue
        configured = entry.get(SOUL_FILENAME)
        if isinstance(configured, str) and has_env_reference(configured):
            continue
        try:
            stored = persona_soul_text(name)
        except (OSError, ValueError):
            continue  # unreadable file, or a name that is not a safe directory name
        if stored is not None:
            entry[SOUL_FILENAME] = stored


def _webhook_server_from_form(raw: Any) -> dict[str, Any]:
    """Build the file's ``webhook_server:`` mapping; empty fields are omitted."""
    if not isinstance(raw, dict):
        raise ValueError("webhook_server: expected an object.")
    server: dict[str, Any] = {}
    host = _form_text(raw, "host", "webhook_server")
    if host:
        server["host"] = host
    port = raw.get("port")
    if isinstance(port, str):
        port = port.strip()
    if port not in (None, ""):
        try:
            number = int(port)
        except (TypeError, ValueError):
            raise ValueError("webhook_server.port: expected an integer.") from None
        if isinstance(port, bool) or not 1 <= number <= 65535:
            raise ValueError("webhook_server.port: must be between 1 and 65535.")
        server["port"] = number
    hosts = raw.get("allowed_hosts")
    if hosts is not None:
        names = _form_names(hosts, "webhook_server.allowed_hosts")
        if names:
            server["allowed_hosts"] = names
    output_log = raw.get("output_log")
    if output_log is not None:
        if not isinstance(output_log, bool):
            raise ValueError("webhook_server.output_log: expected a boolean.")
        if output_log:
            server["output_log"] = True
    return server


def _interaction_from_form(raw: Any) -> dict[str, Any]:
    """Build the file's ``interaction:`` mapping from the form's chat selection."""
    if not isinstance(raw, dict):
        raise ValueError('interaction: expected an object, e.g. {"chat": {"agent": "name"}}.')
    interaction: dict[str, Any] = {}
    chat = raw.get("chat")
    if chat is not None:
        if not isinstance(chat, dict):
            raise ValueError("interaction.chat: expected an object.")
        agent = _form_text(chat, "agent", "interaction.chat")
        if agent:
            interaction["chat"] = {"agent": agent}
    return interaction


def _webhooks_from_form(raw: Any) -> list[dict[str, Any]]:
    """Build the file's ``webhooks:`` list from the form's webhook entries.

    Each entry needs a ``path`` starting with ``/`` (unique); ``agent``,
    ``secret`` and ``system_prompt`` are written only when non-empty.
    """
    webhooks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, any_entry in enumerate(_form_entries(raw, "webhooks")):
        entry = _form_entry(any_entry, "webhooks", index)
        path = _form_text(entry, "path", f"webhooks[{index}]")
        if not path.startswith("/"):
            raise ValueError(f"webhooks[{index}]: path must start with '/'.")
        if path in seen:
            raise ValueError(f"webhooks[{index}]: duplicate path {path!r}.")
        seen.add(path)
        item: dict[str, Any] = {"path": path}
        for key in ("agent", "secret", "system_prompt"):
            text = _form_text(entry, key, f"webhooks[{index}]")
            if text:
                item[key] = text
        webhooks.append(item)
    return webhooks


def _build_section(section: str, raw: Any, current: Any) -> Any:
    """Build one managed section from the body's value for it."""
    if section == "providers":
        return _providers_from_form(raw)
    if section == "agents":
        return _agents_from_form(raw)
    if section == "personas":
        return _personas_from_form(raw, current)
    if section == "interaction":
        return _interaction_from_form(raw)
    if section == "webhook_server":
        return _webhook_server_from_form(raw)
    return _webhooks_from_form(raw)


def _build_config_dict(payload: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Build the config mapping from the form JSON.

    The form manages ``providers``, ``agents``, ``personas``, ``interaction`` and
    ``webhooks``.
    A managed section is rewritten only when the body *sends* it: a section that
    is absent (or ``null``) is carried over exactly as it is on disk, so a body
    that touches one section can never wipe the others, and a body that sends no
    managed section at all is refused. A section the body does send replaces its
    previous value -- an empty list clears it. Top-level keys the form never
    manages are always carried over. A section whose shape is
    wrong is rejected with an error instead of being silently dropped.
    """
    if not any(_section_sent(payload, section) for section in _MANAGED_SECTIONS):
        raise ValueError(
            "Nothing to save: the body must send at least one of providers, "
            "agents, personas, interaction or webhooks."
        )

    data: dict[str, Any] = {}
    for section in _MANAGED_SECTIONS:
        current_value = current.get(section)
        value = (
            _build_section(section, payload[section], current_value)
            if _section_sent(payload, section)
            else current_value
        )
        if value:  # a section emptied through the form is written as no section
            data[section] = value

    # Carry over top-level keys the form doesn't manage so a save via the UI
    # never drops them.
    for key, value in current.items():
        if key not in _MANAGED_SECTIONS:
            data[key] = value

    return data


def _write_config_text(text: str) -> Path:
    """Atomically write ``text`` to the config file and return its path."""
    path = config_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".config-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return path


class ConfigHandler(BaseHTTPRequestHandler):
    """Serve the config page and a tiny JSON API for reading/writing the file."""

    server_version = "LightHarnessWeb/1.0"

    # Bound per server by create_server().
    api_token: str
    cors_allow_origin: str | None
    allowed_hosts: frozenset[str]

    def _host_allowed(self) -> bool:
        """True when the request's ``Host`` header names this server."""
        return host_is_allowed(self, self.allowed_hosts)

    def _guard(self, api: bool) -> bool:
        """Apply the Host check and, for ``/api/*``, the token check; send the
        error response and return False when the request must not proceed."""
        if not self._host_allowed():
            self._reject(403, {"error": "forbidden host"})
            return False
        supplied = self.headers.get("X-Agent-Token") or ""
        if api and not secret_matches(supplied, self.api_token):
            self._reject(401, {"error": "missing or invalid X-Agent-Token"})
            return False
        return True

    def do_GET(self) -> None:
        # A GET body is never consumed, so a request that declares one cannot be
        # answered on a connection that stays open.
        if declared_body(self):
            self.close_connection = True
        path = self.path.split("?", 1)[0]
        if not self._guard(api=path.startswith("/api/")):
            return
        if path in ("", "/"):
            self._send_html(200, _PAGE)
        elif path == "/api/config":
            self._handle_get_config()
        elif path == "/api/tools":
            self._handle_get_tools()
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self._reject(404, {"error": "not found"})

    def do_OPTIONS(self) -> None:
        """CORS preflight for ``/api/*``. Needs no token (browsers don't send one)."""
        if not self._host_allowed():
            self._send_json(403, {"error": "forbidden host"})
            return
        self.send_response(204)
        self._send_cors_headers()
        if self.cors_allow_origin is not None:
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Agent-Token")
            self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        if not self._guard(api=True):
            return
        if self.path.split("?", 1)[0] != "/api/config":
            self._reject(404, {"error": "not found"})
            return
        try:
            raw = read_body(self)
        except BodyError as exc:
            self._send_json(exc.status, {"ok": False, "error": exc.message})
            return
        try:
            # An empty body is not a JSON object: it must not stand in for "{}",
            # which used to mean "write the config with no managed sections".
            payload = json.loads(raw)
        except ValueError as exc:  # JSONDecodeError or invalid UTF-8
            self._send_json(400, {"ok": False, "error": f"Invalid JSON: {exc}"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"ok": False, "error": "Body must be a JSON object."})
            return
        try:
            current = load_yaml(config_file_path())
            data = _build_config_dict(payload, current)
            souls = (
                _persona_souls_from_form(payload["personas"])
                if _section_sent(payload, "personas")
                else {}
            )
            text = yaml.safe_dump(
                data, sort_keys=False, default_flow_style=False, allow_unicode=True
            )
            # The persona's prompt is the file the agent reads, so saving the form's
            # persona field rewrites it. A ``$NAME`` in it resolves exactly as it
            # does when the config seeds that file on first creation.
            for persona_name, soul in souls.items():
                write_persona_soul(persona_name, resolve_env(soul))
            path = _write_config_text(text)
        except ValueError as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
            return
        except OSError as exc:
            self._send_json(500, {"ok": False, "error": f"Could not write file: {exc}"})
            return
        self._send_json(200, {"ok": True, "path": str(path)})

    def _handle_get_config(self) -> None:
        path = config_file_path()
        if path.exists():
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, ValueError) as exc:  # unreadable, or not UTF-8 text
                self._send_json(500, {"error": f"Cannot read {path}: {exc}"})
                return
            data: dict[str, Any] = {}
            try:
                loaded = yaml.safe_load(text)
            except yaml.YAMLError:
                loaded = None
            if isinstance(loaded, dict):
                data = loaded
            # The form has to show the prompt the agent actually uses: the persona's
            # stored SOUL.md, not the config value that only seeds it.
            _effective_persona_souls(data)
            self._send_json(
                200, {"exists": True, "path": str(path), "text": text, "data": data}
            )
        else:
            self._send_json(200, {"exists": False, "path": str(path), "text": None, "data": {}})

    def _handle_get_tools(self) -> None:
        self._send_json(
            200, {"tools": list(CORE_TOOL_NAMES), "forgejo_tools": list(FORGEJO_TOOL_NAMES)}
        )

    def _reject(self, status: int, payload: dict[str, Any]) -> None:
        """Answer a request whose body is not going to be read: the connection has
        to close, otherwise its leftover bytes are parsed as the next request."""
        if declared_body(self):
            self.close_connection = True
        self._send_json(status, payload)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # /api/* answers carry the config's API keys and webhook secrets, so none
        # of them -- not even an error -- may be cached by anything in between.
        self.send_header("Cache-Control", "no-store")
        if self.close_connection:
            self.send_header("Connection", "close")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        if self.cors_allow_origin is not None:
            self.send_header("Access-Control-Allow-Origin", self.cors_allow_origin)
            if self.cors_allow_origin != "*":
                self.send_header("Vary", "Origin")

    def _send_html(self, status: int, html: str) -> None:
        body = html.replace("__AGENT_TOKEN__", self.api_token).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # The page embeds the API token: never cache it or let it be framed.
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Options", "DENY")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the console clean while the server runs."""


class ConfigServer(ThreadingHTTPServer):
    """The config web server; ``token`` is the per-launch API token."""

    token: str


_CORS_ORIGIN_RE = re.compile(
    r"https?://"
    r"(?:\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"(?::[0-9]{1,5})?"
)


def _validated_cors_origin(origin: str | None) -> str | None:
    """The value to send as ``Access-Control-Allow-Origin``, or ``None`` when CORS
    is disabled (``None`` or empty).

    ``*`` is accepted as it is; anything else must be a bare origin -- ``http(s)``
    plus a host and an optional port. The value is echoed into a response header
    verbatim, so a value carrying ``CR``/``LF`` would append a header of the
    caller's choosing (``Access-Control-Allow-Credentials: true``, say) to every
    ``/api/*`` response -- an API that returns API keys and webhook secrets. A path,
    query, fragment or ``user:pass@`` is not an origin either and never matched a
    browser's origin comparison anyway, so rejecting them at startup turns a typo
    into a message instead of a silently wider CORS.
    """
    if origin is None or not origin.strip():
        return None
    if origin == "*":
        return "*"
    if not _CORS_ORIGIN_RE.fullmatch(origin):
        raise ValueError(
            f"Invalid CORS allow-origin {origin!r}: use '*' or a bare origin such as "
            "https://ui.example or https://ui.example:8443 (no path, credentials, "
            "or whitespace)."
        )
    return origin


def create_server(
    host: str,
    port: int,
    cors_origin: str | None = "*",
    allowed_hosts: Sequence[str] = (),
) -> ConfigServer:
    """Build (but do not start) the config web server.

    ``cors_origin`` is sent as ``Access-Control-Allow-Origin`` on ``/api/*``
    responses (default ``*``; empty/``None`` disables CORS) and must be ``*`` or a
    bare origin -- anything else raises ``ValueError`` (see
    :func:`_validated_cors_origin`). Every ``/api/*`` request must also carry the
    per-launch token (``server.token``) in an ``X-Agent-Token`` header, so permissive
    CORS does not let other sites use the API.

    The token is only embedded in the HTML page, which is sent with no CORS headers,
    with ``Cache-Control: no-store``, and only for a request whose ``Host`` header
    names this server: loopback names, an IP address, the bind host, or one of
    ``allowed_hosts`` (the names this server is actually reached by -- a mDNS name,
    the name a reverse proxy forwards). Binding to a wildcard address does not relax
    that: it is the bind a browser can be talked into reaching under an attacker's
    hostname.
    """
    token = secrets.token_urlsafe(32)
    origin = _validated_cors_origin(cors_origin)
    hosts = host_allow_list(host, allowed_hosts)

    class BoundHandler(ConfigHandler):
        api_token = token
        cors_allow_origin = origin
        allowed_hosts = hosts

    server = ConfigServer((host, port), BoundHandler)
    server.token = token
    return server


def start_config_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    cors_origin: str | None = "*",
    allowed_hosts: Sequence[str] = (),
) -> ConfigServer:
    """Bind the config web server and serve it from a daemon thread.

    Binding happens in the caller, so a bad origin or busy port raises here.
    The caller owns the returned server: ``shutdown()`` then ``server_close()``.
    """
    import threading

    server = create_server(host, port, cors_origin, allowed_hosts)
    print(f"Serving config UI at http://{host}:{port}/")
    print(f"Config file: {config_file_path()}")
    print(f"API token (send as 'X-Agent-Token' to call /api/* directly): {server.token}")
    threading.Thread(target=server.serve_forever, daemon=True, name="config-webui").start()
    return server
