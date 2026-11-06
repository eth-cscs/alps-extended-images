#!/usr/bin/env node
import fs from 'node:fs';
import http from 'node:http';
import https from 'node:https';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { constants } from 'node:os';
import { parseArgs } from 'node:util';
import { pathToFileURL } from 'node:url';
import * as zlib from 'node:zlib';

const HOP_HEADERS = new Set([
  'connection', 'proxy-connection', 'keep-alive', 'transfer-encoding',
  'te', 'trailer', 'upgrade', 'proxy-authorization', 'proxy-authenticate',
]);

function forwardedHeaders(headers) {
  const excluded = new Set(HOP_HEADERS);
  for (const name of (headers.connection ?? '').split(',')) excluded.add(name.trim().toLowerCase());
  return Object.fromEntries(Object.entries(headers).filter(([name]) => !excluded.has(name)));
}

function redact(text, secrets) {
  for (const secret of secrets) text = text.split(secret).join('[REDACTED]');
  return text;
}

class ResponseCapture {
  constructor(filename, secrets) {
    this.fd = fs.openSync(filename, 'wx', 0o600);
    this.secrets = secrets.map(secret => Buffer.from(secret));
    this.keep = Math.max(0, ...this.secrets.map(secret => secret.length - 1));
    this.pending = Buffer.alloc(0);
  }

  write(chunk, final = false) {
    this.pending = this.pending.length ? Buffer.concat([this.pending, chunk]) : chunk;
    // Keep possible secret prefixes until the next chunk, including SSE frame boundaries.
    while (this.pending.length) {
      let index = -1;
      let match;
      for (const secret of this.secrets) {
        const found = this.pending.indexOf(secret);
        if (found !== -1 && (index === -1 || found < index)) {
          index = found;
          match = secret;
        }
      }
      if (index !== -1) {
        fs.writeFileSync(this.fd, this.pending.subarray(0, index));
        fs.writeFileSync(this.fd, '[REDACTED]');
        this.pending = this.pending.subarray(index + match.length);
      } else {
        const length = final ? this.pending.length : Math.max(0, this.pending.length - this.keep);
        fs.writeFileSync(this.fd, this.pending.subarray(0, length));
        this.pending = this.pending.subarray(length);
        break;
      }
    }
  }

  close() {
    this.write(Buffer.alloc(0), true);
    fs.closeSync(this.fd);
  }
}

function decodedResponse(response) {
  switch (response.headers['content-encoding']?.trim().toLowerCase()) {
    case undefined:
    case 'identity': return response;
    case 'gzip': return response.pipe(zlib.createGunzip());
    case 'deflate': return response.pipe(zlib.createInflate());
    case 'br': return response.pipe(zlib.createBrotliDecompress());
    case 'zstd': return response.pipe(zlib.createZstdDecompress());
    default: throw new Error('Unsupported provider response encoding');
  }
}

export async function startProviderTrace({ upstream, output, port = 18080, secrets = [], agent }) {
  const origin = new URL(upstream);
  if (origin.protocol !== 'https:' || origin.username || origin.password) {
    throw new Error('The upstream must be an HTTPS URL without credentials');
  }
  secrets = secrets.filter(Boolean);
  fs.mkdirSync(output, { recursive: true, mode: 0o700 });
  let sequence = 0;
  const active = new Set();
  const server = http.createServer((request, response) => {
    const target = new URL(request.url, origin);
    if (target.origin !== origin.origin) {
      response.writeHead(400).end('Unexpected provider origin');
      return;
    }
    const prefix = path.join(output, String(++sequence).padStart(4, '0'));
    const metadata = {
      method: request.method, path: target.pathname, started_at: new Date().toISOString(),
      status: null, headers_ms: null, first_body_ms: null, completed_ms: null,
      received_bytes: 0, request: null,
    };
    const started = performance.now();
    let capture;
    let metadataFd;
    try {
      capture = new ResponseCapture(prefix + '.response', secrets);
      metadataFd = fs.openSync(prefix + '.json', 'wx', 0o600);
    } catch (error) {
      capture?.close();
      response.writeHead(500).end('Unable to create private provider capture');
      return;
    }
    const saveMetadata = () => {
      const text = redact(JSON.stringify(metadata, null, 2) + '\n', secrets);
      fs.ftruncateSync(metadataFd, 0);
      fs.writeSync(metadataFd, text, 0, 'utf8');
    };
    saveMetadata();
    let finished = false;
    const finish = (outcome, error) => {
      if (finished) return;
      finished = true;
      metadata.completed_ms = performance.now() - started;
      metadata.outcome = outcome;
      if (error) metadata.error = redact(String(error.message), secrets);
      capture.close();
      saveMetadata();
      fs.closeSync(metadataFd);
    };
    const headers = forwardedHeaders(request.headers);
    headers.host = target.host;
    // Prefer readable SSE; forwarding to the CLI still preserves response bytes and status.
    headers['accept-encoding'] = 'identity';
    const outgoing = https.request(target, { method: request.method, headers, agent }, incoming => {
      if (finished) {
        incoming.destroy();
        return;
      }
      metadata.status = incoming.statusCode;
      metadata.headers_ms = performance.now() - started;
      metadata.content_type = incoming.headers['content-type'] ?? null;
      metadata.content_encoding = incoming.headers['content-encoding'] ?? 'identity';
      metadata.request_id = incoming.headers['x-request-id'] ?? incoming.headers['request-id'] ?? null;
      saveMetadata();
      const responseHeaders = forwardedHeaders(incoming.headers);
      // Signal HTTP completion only after the decoded capture has been flushed.
      delete responseHeaders['content-length'];
      response.writeHead(incoming.statusCode, responseHeaders);
      incoming.on('data', chunk => {
        metadata.first_body_ms ??= performance.now() - started;
        metadata.received_bytes += chunk.length;
      });
      let decoded;
      try {
        decoded = decodedResponse(incoming);
      } catch (error) {
        finish('capture-error', error);
        outgoing.destroy(error);
        return;
      }
      decoded.on('data', chunk => {
        if (!finished) capture.write(chunk);
      });
      decoded.on('end', () => {
        finish('complete');
        if (!response.destroyed) response.end();
      });
      decoded.on('error', error => {
        finish('capture-error', error);
        response.destroy();
        outgoing.destroy();
      });
      incoming.on('aborted', () => {
        finish('upstream-aborted');
        response.destroy();
      });
      // Do not feed the redacted copy to OpenCode: it must see the original response.
      incoming.pipe(response, { end: false });
    });
    active.add(outgoing);
    outgoing.on('close', () => active.delete(outgoing));
    outgoing.on('error', error => {
      finish('upstream-error', error);
      if (!response.headersSent) response.writeHead(502).end('Provider connection failed');
      else response.destroy();
    });
    const chunks = [];
    request.on('data', chunk => chunks.push(chunk));
    request.on('end', () => {
      try {
        const body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
        // Deliberately omit request headers, messages, and system prompts from metadata.
        metadata.request = {
          model: body.model, max_tokens: body.max_tokens,
          stream: body.stream, tool_choice: body.tool_choice ?? null,
        };
      } catch {
        // Non-JSON requests still reach the provider unchanged.
      }
      if (!finished) saveMetadata();
    });
    request.on('aborted', () => outgoing.destroy());
    response.on('close', () => {
      if (!response.writableFinished) {
        finish('client-disconnected');
        outgoing.destroy();
      }
    });
    request.pipe(outgoing);
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', resolve);
  });
  return {
    url: `http://127.0.0.1:${server.address().port}`,
    async close() {
      const closed = new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
      for (const request of active) request.destroy();
      server.closeAllConnections();
      await closed;
    },
  };
}

async function main() {
  const { values, positionals: command } = parseArgs({
    allowPositionals: true,
    options: {
      upstream: { type: 'string' },
      output: { type: 'string', default: '/work/provider-trace' },
      port: { type: 'string', default: '18080' },
    },
  });
  if (!values.upstream || !command.length) {
    throw new Error('Usage: provider-trace.mjs --upstream HTTPS_URL [--output DIR] [--port PORT] -- COMMAND [ARGS]');
  }
  const port = Number(values.port);
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error('Invalid capture port');
  const capture = await startProviderTrace({
    upstream: values.upstream, output: values.output, port,
    secrets: [process.env.CSCS_INFERENCE_API_KEY],
  });
  try {
    const child = spawn(command[0], command.slice(1), { stdio: 'inherit' });
    const forward = signal => child.kill(signal);
    const onTerm = () => forward('SIGTERM');
    const onInt = () => forward('SIGINT');
    process.on('SIGTERM', onTerm);
    process.on('SIGINT', onInt);
    const result = await new Promise((resolve, reject) => {
      child.once('error', reject);
      child.once('exit', (code, signal) => resolve({ code, signal }));
    });
    process.off('SIGTERM', onTerm);
    process.off('SIGINT', onInt);
    process.exitCode = result.code ?? 128 + constants.signals[result.signal];
  } finally {
    await capture.close();
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(error => {
    console.error(redact(error.message, [process.env.CSCS_INFERENCE_API_KEY].filter(Boolean)));
    process.exitCode = 1;
  });
}
