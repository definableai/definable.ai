#!/usr/bin/env node
/**
 * Definable WhatsApp Bridge — Baileys sidecar.
 *
 * Wraps @whiskeysockets/baileys and exposes a WebSocket JSON API
 * for the Python BaileysProvider to communicate with.
 *
 * Usage:
 *   node index.js --port=PORT --auth-dir=./auth [--heartbeat=60] [--verbose]
 */

import { makeWASocket, useMultiFileAuthState, makeCacheableSignalKeyStore, DisconnectReason, fetchLatestBaileysVersion } from "@whiskeysockets/baileys";
import { WebSocketServer } from "ws";
import { writeFileSync, copyFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import pino from "pino";

import { DEFAULT_RECONNECT_POLICY, computeBackoff, sleep } from "./lib/reconnect.js";
import { downloadAndEncode, hasMedia } from "./lib/media.js";
import { normalizeE164, phoneFromJid, redactPhone, isGroupJid } from "./lib/normalize.js";
import qrcode from "qrcode-terminal";

// --------------------------------------------------------------------------- //
// CLI args                                                                     //
// --------------------------------------------------------------------------- //

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const [k, v] = a.replace(/^--/, "").split("=");
    return [k, v ?? "true"];
  })
);

const PORT = parseInt(args.port || "0", 10);
const AUTH_DIR = args["auth-dir"] || "./whatsapp-auth";
const HEARTBEAT_SEC = parseInt(args.heartbeat || "60", 10);
const VERBOSE = args.verbose === "true";
const MESSAGE_TIMEOUT_MS = 30 * 60 * 1000; // 30 min watchdog

const logger = pino({ level: VERBOSE ? "debug" : "silent" });

// --------------------------------------------------------------------------- //
// State                                                                        //
// --------------------------------------------------------------------------- //

/** @type {import("ws").WebSocket | null} */
let wsClient = null;
let sock = null;
let reconnectAttempts = 0;
let lastConnectedAt = null;
let lastMessageAt = null;
let lastError = null;
let messagesHandled = 0;
let isShuttingDown = false;
let selfPhone = null;
let selfJid = null;
let heartbeatTimer = null;
let watchdogTimer = null;

// Credential save queue — ensures sequential saves, prevents corruption
let credSavePromise = Promise.resolve();

// Baileys message retry counter (LRU-bounded)
const msgRetryCounterCache = new Map();
const MSG_RETRY_CACHE_MAX = 1000;

/** @type {Map<string, { resolve: Function, reject: Function }>} */
const pendingCommands = new Map();

// --------------------------------------------------------------------------- //
// WebSocket server                                                             //
// --------------------------------------------------------------------------- //

function send(data) {
  if (wsClient && wsClient.readyState === 1) {
    try {
      const payload = JSON.stringify(data);
      wsClient.send(payload);
    } catch (err) {
      console.error(`[bridge] Failed to send WebSocket frame: ${err.message}`);
    }
  }
}

function startWsServer() {
  return new Promise((resolve) => {
    const wss = new WebSocketServer({ port: PORT, host: "127.0.0.1", maxPayload: 50 * 1024 * 1024 }, () => {
      const addr = wss.address();
      const actualPort = typeof addr === "object" ? addr.port : PORT;
      // Print port on stdout so Python can read it
      process.stdout.write(`PORT:${actualPort}\n`);
      if (VERBOSE) console.error(`[bridge] WebSocket server on 127.0.0.1:${actualPort}`);
      resolve(actualPort);
    });

    wss.on("connection", (ws) => {
      if (wsClient) {
        ws.close(4000, "Only one client allowed");
        return;
      }
      wsClient = ws;
      if (VERBOSE) console.error("[bridge] Python client connected");

      // Send initial ready state
      send({
        type: "ready",
        connected: sock !== null && selfJid !== null,
        self_phone: selfPhone,
        self_jid: selfJid,
        auth_exists: existsSync(join(AUTH_DIR, "creds.json")),
      });

      ws.on("message", (raw) => {
        try {
          const msg = JSON.parse(raw.toString());
          handleCommand(msg);
        } catch (err) {
          console.error(`[bridge] Failed to parse command: ${err.message}`);
        }
      });

      ws.on("close", () => {
        wsClient = null;
        if (VERBOSE) console.error("[bridge] Python client disconnected");
      });
    });

    wss.on("error", (err) => {
      console.error(`[bridge] WebSocket server error: ${err.message}`);
    });
  });
}

// --------------------------------------------------------------------------- //
// Command handler                                                              //
// --------------------------------------------------------------------------- //

async function handleCommand(msg) {
  const { type, id } = msg;

  try {
    switch (type) {
      case "send": {
        const result = await handleSend(msg);
        send({ type: "send_result", id, ...result });
        break;
      }
      case "send_poll": {
        const result = await handleSendPoll(msg);
        send({ type: "send_poll_result", id, ...result });
        break;
      }
      case "send_reaction": {
        const result = await handleSendReaction(msg);
        send({ type: "send_reaction_result", id, ...result });
        break;
      }
      case "send_composing": {
        if (sock && msg.to) {
          await sock.sendPresenceUpdate("composing", msg.to);
        }
        break;
      }
      case "login_qr_start": {
        // QR is auto-generated on connect when no creds exist.
        // If force, delete creds and reconnect.
        if (msg.force && existsSync(join(AUTH_DIR, "creds.json"))) {
          const { unlinkSync } = await import("node:fs");
          unlinkSync(join(AUTH_DIR, "creds.json"));
        }
        if (!sock) {
          startBaileys();
        }
        send({ type: "login_qr_start_result", id, message: "QR login initiated. Watch for 'qr' events." });
        break;
      }
      case "login_qr_wait": {
        // The Python side waits for a "connected" event.
        // We just acknowledge.
        send({ type: "login_qr_wait_result", id, message: "Waiting for QR scan..." });
        break;
      }
      case "get_status": {
        send({
          type: "get_status_result",
          id,
          connected: sock !== null && selfJid !== null,
          running: !isShuttingDown,
          reconnect_attempts: reconnectAttempts,
          last_connected_at: lastConnectedAt,
          last_message_at: lastMessageAt,
          last_error: lastError,
          linked: existsSync(join(AUTH_DIR, "creds.json")),
          self_phone: selfPhone,
          self_jid: selfJid,
          messages_handled: messagesHandled,
        });
        break;
      }
      case "logout": {
        if (sock) {
          await sock.logout();
        }
        send({ type: "logout_result", id, success: true });
        break;
      }
      case "check_on_whatsapp": {
        const result = await handleCheckOnWhatsApp(msg);
        send({ type: "check_on_whatsapp_result", id, ...result });
        break;
      }
      case "save_contact": {
        const result = await handleSaveContact(msg);
        send({ type: "save_contact_result", id, ...result });
        break;
      }
      case "shutdown": {
        isShuttingDown = true;
        if (sock) {
          sock.end(undefined);
        }
        clearInterval(heartbeatTimer);
        clearInterval(watchdogTimer);
        send({ type: "shutdown_ack" });
        setTimeout(() => process.exit(0), 500);
        break;
      }
      default:
        send({ type: "error", code: "UNKNOWN_COMMAND", message: `Unknown command: ${type}` });
    }
  } catch (err) {
    send({ type: `${type}_result`, id, success: false, error: err.message });
  }
}

// --------------------------------------------------------------------------- //
// Send handlers                                                                //
// --------------------------------------------------------------------------- //

async function handleSend(msg) {
  if (!sock) return { success: false, error: "Not connected" };

  const to = msg.to;
  const content = {};

  // Text
  if (msg.body) {
    content.text = msg.body;
  }

  // Media
  if (msg.media) {
    const buffer = Buffer.from(msg.media.base64 || msg.media.content_base64 || "", "base64");
    const mediaType = msg.media.type || "image";
    const mimeType = msg.media.mime_type || msg.media.mimeType || "application/octet-stream";

    if (mediaType === "image") {
      content.image = buffer;
      content.mimetype = mimeType;
      if (msg.body) content.caption = msg.body;
      delete content.text;
    } else if (mediaType === "audio") {
      content.audio = buffer;
      content.mimetype = mimeType;
      content.ptt = mimeType.includes("ogg");
      delete content.text;
    } else if (mediaType === "video") {
      content.video = buffer;
      content.mimetype = mimeType;
      if (msg.body) content.caption = msg.body;
      delete content.text;
    } else {
      content.document = buffer;
      content.mimetype = mimeType;
      content.fileName = msg.media.filename || "file";
      if (msg.body) content.caption = msg.body;
      delete content.text;
    }
  }

  // Reply context
  const opts = {};
  if (msg.reply_to_id) {
    opts.quoted = { key: { id: msg.reply_to_id, remoteJid: to } };
  }

  try {
    const result = await sock.sendMessage(to, content, opts);
    const messageId = result?.key?.id || null;
    if (VERBOSE) console.error(`[bridge] Sent to ${redactPhone(to)}: ${messageId}`);
    return { success: true, message_id: messageId };
  } catch (err) {
    console.error(`[bridge] Send failed: ${err.message}`);
    return { success: false, error: err.message };
  }
}

async function handleSendPoll(msg) {
  if (!sock) return { success: false, error: "Not connected" };
  try {
    const result = await sock.sendMessage(msg.to, {
      poll: {
        name: msg.question,
        values: msg.options,
        selectableCount: msg.allows_multiple ? 0 : 1,
      },
    });
    return { success: true, message_id: result?.key?.id || null };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

async function handleSendReaction(msg) {
  if (!sock) return { success: false, error: "Not connected" };
  try {
    await sock.sendMessage(msg.chat_jid, {
      react: {
        text: msg.emoji,
        key: {
          id: msg.message_id,
          remoteJid: msg.chat_jid,
          fromMe: msg.from_me || false,
          participant: msg.participant || undefined,
        },
      },
    });
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

// --------------------------------------------------------------------------- //
// Contact validation & management                                              //
// --------------------------------------------------------------------------- //

async function handleCheckOnWhatsApp(msg) {
  if (!sock) return { success: false, error: "Not connected" };

  const phones = msg.phones; // Array of E.164 strings, e.g. ["+919810464995"]
  if (!phones || !Array.isArray(phones) || phones.length === 0) {
    return { success: false, error: "phones array is required" };
  }

  try {
    // Baileys accepts JIDs or bare numbers with country code
    const results = await sock.onWhatsApp(...phones);
    const checked = (results || []).map((r) => ({
      jid: r.jid || null,
      exists: !!r.exists,
    }));
    if (VERBOSE) console.error(`[bridge] Checked ${phones.length} numbers: ${checked.filter((r) => r.exists).length} on WhatsApp`);
    return { success: true, results: checked };
  } catch (err) {
    console.error(`[bridge] Check on WhatsApp failed: ${err.message}`);
    return { success: false, error: err.message };
  }
}

async function handleSaveContact(msg) {
  if (!sock) return { success: false, error: "Not connected" };

  const jid = msg.jid; // e.g. "919810464995@s.whatsapp.net"
  const name = msg.name; // e.g. "Dr Sanjeev Chawla"
  if (!jid || !name) {
    return { success: false, error: "jid and name are required" };
  }

  try {
    await sock.addOrEditContact(jid, { fullName: name });
    if (VERBOSE) console.error(`[bridge] Saved contact: ${redactPhone(jid)} as "${name}"`);
    return { success: true };
  } catch (err) {
    console.error(`[bridge] Save contact failed: ${err.message}`);
    return { success: false, error: err.message };
  }
}

// --------------------------------------------------------------------------- //
// Credential save queue — prevents concurrent writes that corrupt creds.json   //
// --------------------------------------------------------------------------- //

function queueCredSave(saveCreds) {
  credSavePromise = credSavePromise
    .then(async () => {
      const credsPath = join(AUTH_DIR, "creds.json");
      if (existsSync(credsPath)) {
        try { copyFileSync(credsPath, credsPath + ".bak"); } catch {}
      }
      await saveCreds();
    })
    .catch((err) => {
      console.error(`[bridge] Credential save failed: ${err.message}`);
    });
}

// --------------------------------------------------------------------------- //
// Baileys socket                                                               //
// --------------------------------------------------------------------------- //

async function startBaileys() {
  const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);
  const { version } = await fetchLatestBaileysVersion();

  // Wrap signal key store with caching layer to reduce I/O
  let authState = state;
  try {
    authState = {
      ...state,
      keys: makeCacheableSignalKeyStore(state.keys, logger),
    };
  } catch {
    // Fallback to raw state if caching fails (older Baileys versions)
  }

  sock = makeWASocket({
    version,
    auth: authState,
    logger,
    generateHighQualityLinkPreview: false,
    syncFullHistory: false,
    // Message retry counter with bounded cache
    msgRetryCounterCache,
    // Reduce memory: don't cache messages we've already forwarded
    shouldIgnoreJid: (jid) => isGroupJid(jid) && false, // keep groups for now
    markOnlineOnConnect: false,
    // Browser identification
    browser: ["Definable", "Chrome", "22.0"],
  });

  // Prune retry cache to prevent unbounded growth
  if (msgRetryCounterCache.size > MSG_RETRY_CACHE_MAX) {
    const keysToDelete = [...msgRetryCounterCache.keys()].slice(0, msgRetryCounterCache.size - MSG_RETRY_CACHE_MAX);
    for (const key of keysToDelete) {
      msgRetryCounterCache.delete(key);
    }
  }

  // --- Auth state save (properly queued) ---
  sock.ev.on("creds.update", () => queueCredSave(saveCreds));

  // --- Connection state ---
  sock.ev.on("connection.update", (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      qrcode.generate(qr, { small: true }, (qrText) => {
        // Print to stderr so it shows in terminal but doesn't mix with PORT: protocol on stdout
        console.error("\n" + qrText);
        console.error("[bridge] Scan the QR code above with WhatsApp on your phone\n");
      });
      send({ type: "qr", data: qr });
    }

    if (connection === "open") {
      reconnectAttempts = 0;
      lastConnectedAt = Date.now() / 1000;
      selfJid = sock.user?.id || null;
      selfPhone = selfJid ? (phoneFromJid(selfJid) || normalizeE164(selfJid.split("@")[0] || "")) : null;

      send({
        type: "connected",
        self_phone: selfPhone,
        self_jid: selfJid,
      });

      if (VERBOSE) console.error(`[bridge] Connected as ${redactPhone(selfPhone)}`);
      startHeartbeat();
      startWatchdog();
    }

    if (connection === "close") {
      const statusCode = lastDisconnect?.error?.output?.statusCode;
      const isLoggedOut = statusCode === DisconnectReason.loggedOut;
      const isConflict = statusCode === 440;

      clearInterval(heartbeatTimer);
      clearInterval(watchdogTimer);

      send({
        type: "disconnected",
        reason: isLoggedOut ? "logged_out" : isConflict ? "session_conflict" : "connection_lost",
        status_code: statusCode || null,
        is_logged_out: isLoggedOut,
        reconnecting: !isLoggedOut && !isConflict && !isShuttingDown,
        attempt: reconnectAttempts,
      });

      if (isLoggedOut || isConflict) {
        lastError = isLoggedOut ? "Logged out" : "Session conflict (status 440)";
        send({
          type: "logged_out",
          reason: isLoggedOut ? "logged_out" : "session_conflict",
          message: `WhatsApp session terminated: ${lastError}. Relink required.`,
        });
        sock = null;
        selfJid = null;
        selfPhone = null;
        return;
      }

      if (!isShuttingDown) {
        lastError = lastDisconnect?.error?.message || "Unknown disconnect";
        reconnectAttempts++;

        if (reconnectAttempts > DEFAULT_RECONNECT_POLICY.maxAttempts) {
          send({
            type: "error",
            code: "MAX_RECONNECT",
            message: `Max reconnect attempts (${DEFAULT_RECONNECT_POLICY.maxAttempts}) exceeded.`,
          });
          sock = null;
          return;
        }

        const delay = computeBackoff(DEFAULT_RECONNECT_POLICY, reconnectAttempts - 1);
        if (VERBOSE) console.error(`[bridge] Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);

        setTimeout(() => {
          if (!isShuttingDown) startBaileys();
        }, delay);
      }
    }
  });

  // --- Inbound messages ---
  sock.ev.on("messages.upsert", async ({ messages: msgs, type: upsertType }) => {
    if (upsertType !== "notify") return;

    for (const msg of msgs) {
      if (!msg.message) continue;
      if (msg.key.fromMe && !msg.key.remoteJid?.endsWith("@s.whatsapp.net")) continue; // skip own echoes in groups

      const chatJid = msg.key.remoteJid || "";
      const isGroup = isGroupJid(chatJid);
      const senderJid = isGroup ? (msg.key.participant || "") : chatJid;
      const senderPhone = phoneFromJid(senderJid) || normalizeE164(senderJid.split("@")[0] || "") || "";

      // Extract text — check all known message wrapper types
      let body = "";
      const m = msg.message;
      // viewOnceMessageV2 and ephemeralMessage wrappers
      const inner = m.viewOnceMessageV2?.message || m.ephemeralMessage?.message || m;
      if (inner.conversation) body = inner.conversation;
      else if (inner.extendedTextMessage?.text) body = inner.extendedTextMessage.text;
      else if (inner.imageMessage?.caption) body = inner.imageMessage.caption;
      else if (inner.videoMessage?.caption) body = inner.videoMessage.caption;
      else if (inner.documentMessage?.caption) body = inner.documentMessage.caption;
      else if (inner.buttonsResponseMessage?.selectedButtonId) body = inner.buttonsResponseMessage.selectedButtonId;
      else if (inner.listResponseMessage?.singleSelectReply?.selectedRowId) body = inner.listResponseMessage.singleSelectReply.selectedRowId;
      else if (inner.templateButtonReplyMessage?.selectedId) body = inner.templateButtonReplyMessage.selectedId;

      // Extract media
      let media = null;
      let mediaError = null;
      if (hasMedia(msg)) {
        media = await downloadAndEncode(msg);
        if (!media) {
          mediaError = "Media download failed (expired key or network error)";
        }
      }

      // Extract reply context
      const quotedCtx = inner.extendedTextMessage?.contextInfo
        || inner.imageMessage?.contextInfo
        || inner.videoMessage?.contextInfo
        || inner.documentMessage?.contextInfo;
      const quotedMsg = quotedCtx?.quotedMessage;
      const quotedStanza = quotedCtx?.stanzaId;
      const quotedParticipant = quotedCtx?.participant;

      // Group context
      const groupSubject = null; // Would need groupMetadata call
      const mentionedJids = quotedCtx?.mentionedJid || null;

      const event = {
        type: "message",
        id: msg.key.id,
        from_phone: senderPhone,
        from_jid: senderJid,
        chat_jid: chatJid,
        body,
        push_name: msg.pushName || "",
        is_group: isGroup,
        is_from_me: !!msg.key.fromMe,
        timestamp: (msg.messageTimestamp || 0) * 1,
        reply_to_id: quotedStanza || null,
        reply_to_body: quotedMsg?.conversation || quotedMsg?.extendedTextMessage?.text || null,
        reply_to_sender: quotedParticipant || null,
        group_subject: groupSubject,
        group_participants: null,
        mentioned_jids: mentionedJids,
        was_mentioned: mentionedJids ? mentionedJids.includes(selfJid) : false,
        media,
        media_error: mediaError,
        location: inner.locationMessage
          ? { latitude: inner.locationMessage.degreesLatitude, longitude: inner.locationMessage.degreesLongitude }
          : null,
      };

      send(event);
      messagesHandled++;
      lastMessageAt = Date.now() / 1000;
    }
  });
}

// --------------------------------------------------------------------------- //
// Heartbeat + Watchdog                                                         //
// --------------------------------------------------------------------------- //

function startHeartbeat() {
  clearInterval(heartbeatTimer);
  heartbeatTimer = setInterval(() => {
    send({
      type: "status",
      connected: sock !== null && selfJid !== null,
      running: !isShuttingDown,
      reconnect_attempts: reconnectAttempts,
      last_connected_at: lastConnectedAt,
      last_message_at: lastMessageAt,
      last_error: lastError,
      messages_handled: messagesHandled,
      uptime_seconds: lastConnectedAt ? (Date.now() / 1000) - lastConnectedAt : 0,
    });
  }, HEARTBEAT_SEC * 1000);
}

function startWatchdog() {
  clearInterval(watchdogTimer);
  watchdogTimer = setInterval(() => {
    if (lastMessageAt && (Date.now() / 1000 - lastMessageAt) > MESSAGE_TIMEOUT_MS / 1000) {
      console.error("[bridge] Watchdog: no messages in 30min, forcing reconnect");
      if (sock) {
        sock.end(undefined);
      }
    }
  }, 60_000);
}

// --------------------------------------------------------------------------- //
// Startup                                                                      //
// --------------------------------------------------------------------------- //

async function main() {
  await startWsServer();
  await startBaileys();
}

// Graceful shutdown
process.on("SIGTERM", () => {
  isShuttingDown = true;
  if (sock) sock.end(undefined);
  clearInterval(heartbeatTimer);
  clearInterval(watchdogTimer);
  setTimeout(() => process.exit(0), 1000);
});

process.on("SIGINT", () => {
  isShuttingDown = true;
  if (sock) sock.end(undefined);
  clearInterval(heartbeatTimer);
  clearInterval(watchdogTimer);
  setTimeout(() => process.exit(0), 1000);
});

// Handle uncaught errors to prevent silent crashes
process.on("uncaughtException", (err) => {
  console.error(`[bridge] Uncaught exception: ${err.message}`);
  console.error(err.stack);
  send({ type: "error", code: "UNCAUGHT_EXCEPTION", message: err.message });
});

process.on("unhandledRejection", (reason) => {
  const message = reason instanceof Error ? reason.message : String(reason);
  console.error(`[bridge] Unhandled rejection: ${message}`);
  send({ type: "error", code: "UNHANDLED_REJECTION", message });
});

main().catch((err) => {
  console.error(`[bridge] Fatal: ${err.message}`);
  process.exit(1);
});
