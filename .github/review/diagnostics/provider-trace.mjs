#!/usr/bin/env node
import fs from 'node:fs';
import http from 'node:http';
import https from 'node:https';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { isIP } from 'node:net';
import { constants } from 'node:os';
import { parseArgs } from 'node:util';
import { pathToFileURL } from 'node:url';
import { StringDecoder } from 'node:string_decoder';
import * as zlib from 'node:zlib';

const HOP_HEADERS = new Set([
  'connection', 'proxy-connection', 'keep-alive', 'transfer-encoding',
  'te', 'trailer', 'upgrade', 'proxy-authorization', 'proxy-authenticate',
]);

function maskedHeaders(headers, secrets) {
  return Object.fromEntries(Object.entries(headers).map(([name, value]) => [
    name, /authorization|cookie|api[-_]?key|(?:^|[-_])(?:token|secret|password|credentials?)$/i.test(name)
      ? '[REDACTED]' : JSON.parse(redact(JSON.stringify(value), secrets)),
  ]));
}

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
  constructor(filename, secrets, onChunk = () => {}) {
    this.fd = fs.openSync(filename, 'wx', 0o600);
    this.secrets = secrets.map(secret => Buffer.from(secret));
    this.pending = Buffer.alloc(0);
    this.onChunk = onChunk;
  }

  emit(chunk) {
    if (!chunk.length) return;
    fs.writeFileSync(this.fd, chunk);
    this.onChunk(chunk);
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
        this.emit(this.pending.subarray(0, index));
        this.emit(Buffer.from('[REDACTED]'));
        this.pending = this.pending.subarray(index + match.length);
      } else {
        let keep = 0;
        if (!final) {
          for (const secret of this.secrets) {
            for (let length = Math.min(this.pending.length, secret.length - 1); length > keep; length--) {
              if (secret.subarray(0, length).equals(this.pending.subarray(-length))) {
                keep = length;
                break;
              }
            }
          }
        }
        const length = this.pending.length - keep;
        this.emit(this.pending.subarray(0, length));
        this.pending = this.pending.subarray(length);
        break;
      }
    }
  }

  close() {
    if (this.fd === null) return;
    this.write(Buffer.alloc(0), true);
    fs.closeSync(this.fd);
    this.fd = null;
    this.onChunk(Buffer.alloc(0), true);
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

export async function startProviderTrace({
  upstream, output, port = 18080, secrets = [], agent, verbose = false,
  onTrace = chunk => process.stderr.write(chunk),
}) {
  const origin = new URL(upstream);
  if (origin.protocol !== 'https:' || origin.username || origin.password) {
    throw new Error('The upstream must be an HTTPS URL without credentials');
  }
  secrets = secrets.filter(Boolean);
  fs.mkdirSync(output, { recursive: true, mode: 0o700 });
  const traceFd = verbose ? fs.openSync(path.join(output, 'trace.log'), 'wx', 0o600) : null;
  const writeTrace = record => {
    if (!verbose) return;
    const line = redact(JSON.stringify({ at: new Date().toISOString(), ...record }) + '\n', secrets);
    // The file remains complete even if a terminal buffers its live copy.
    fs.writeFileSync(traceFd, line);
    onTrace(line);
  };
  let sequence = 0;
  const active = new Map();
  const server = http.createServer((request, response) => {
    const target = new URL(request.url, origin);
    if (target.origin !== origin.origin) {
      response.writeHead(400).end('Unexpected provider origin');
      return;
    }
    const id = String(++sequence).padStart(4, '0');
    const prefix = path.join(output, id);
    const metadata = {
      method: request.method, path: target.pathname, started_at: new Date().toISOString(),
      status: null, headers_ms: null, first_body_ms: null, completed_ms: null,
      received_bytes: 0, request: null,
    };
    const started = performance.now();
    const log = (event, details = {}) => {
      if (verbose) writeTrace({ request: id, elapsed_ms: performance.now() - started, event, ...details });
    };
    const bodyLog = event => {
      const decoder = new StringDecoder('utf8');
      return (chunk, final = false) => {
        const text = final ? decoder.end() : decoder.write(chunk);
        if (text) log(event, { text });
      };
    };
    let capture;
    let requestCapture;
    let metadataFd;
    try {
      capture = new ResponseCapture(prefix + '.response', secrets, verbose ? bodyLog('response-body') : undefined);
      if (verbose) requestCapture = new ResponseCapture(prefix + '.request', secrets, bodyLog('request-body'));
      metadataFd = fs.openSync(prefix + '.json', 'wx', 0o600);
    } catch (error) {
      capture?.close();
      requestCapture?.close();
      log('capture-error', { error: error.message });
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
    let outgoing;
    let flushCapture;
    let resolveFlushed;
    const flushed = new Promise(resolve => { resolveFlushed = resolve; });
    const finish = (outcome, error) => {
      if (finished) return;
      finished = true;
      metadata.completed_ms = performance.now() - started;
      metadata.outcome = outcome;
      if (error) metadata.error = redact(String(error.message), secrets);
      const finalize = () => {
        requestCapture?.close();
        capture.close();
        saveMetadata();
        fs.closeSync(metadataFd);
        log('complete', { outcome, received_bytes: metadata.received_bytes, ...(error ? { error: error.message } : {}) });
        active.delete(outgoing);
        resolveFlushed();
      };
      if (flushCapture) flushCapture(finalize);
      else finalize();
    };
    const milestone = (event, field, details = {}) => {
      if (finished) return;
      metadata[field] = performance.now() - started;
      saveMetadata();
      log(event, details);
    };
    const headers = forwardedHeaders(request.headers);
    headers.host = target.host;
    // Prefer readable SSE; forwarding to the CLI still preserves response bytes and status.
    headers['accept-encoding'] = 'identity';
    if (verbose) metadata.request_headers = maskedHeaders(headers, secrets);
    log('request-start', { method: request.method, url: target.href, headers: metadata.request_headers });
    outgoing = https.request(target, { method: request.method, headers, agent }, incoming => {
      if (finished) {
        incoming.destroy();
        return;
      }
      metadata.status = incoming.statusCode;
      metadata.headers_ms = performance.now() - started;
      metadata.content_type = incoming.headers['content-type'] ?? null;
      metadata.content_encoding = incoming.headers['content-encoding'] ?? 'identity';
      metadata.request_id = incoming.headers['x-request-id'] ?? incoming.headers['request-id'] ?? null;
      if (verbose) metadata.response_headers = maskedHeaders(incoming.headers, secrets);
      saveMetadata();
      log('provider-headers', { status: incoming.statusCode, headers: metadata.response_headers });
      log('waiting-for-provider-body');
      const responseHeaders = forwardedHeaders(incoming.headers);
      // Signal HTTP completion only after the decoded capture has been flushed.
      delete responseHeaders['content-length'];
      response.writeHead(incoming.statusCode, responseHeaders);
      incoming.on('data', chunk => {
        if (finished) return;
        metadata.received_bytes += chunk.length;
        if (metadata.first_body_ms === null) milestone('first-body', 'first_body_ms', { bytes: chunk.length });
      });
      let decoded;
      let captureError;
      const complete = () => {
        finish(captureError ? 'capture-error' : 'complete', captureError);
        if (!response.destroyed && !response.writableEnded) response.end();
      };
      const captureFailed = error => {
        if (finished || captureError) return;
        captureError = error;
        if (decoded && decoded !== incoming) {
          incoming.unpipe(decoded);
          decoded.destroy();
        }
        capture.close();
        log('capture-error', { error: error.message });
        if (incoming.readableEnded) complete();
      };
      incoming.on('end', () => {
        if (captureError) complete();
      });
      try {
        decoded = decodedResponse(incoming);
        decoded.on('data', chunk => {
          if (!captureError && capture.fd !== null) capture.write(chunk);
        });
        decoded.on('end', complete);
        if (decoded !== incoming) {
          decoded.on('error', captureFailed);
          flushCapture = done => {
            if (decoded.readableEnded || decoded.destroyed) {
              done();
              return;
            }
            // Drain bytes already accepted by zlib even when the HTTP source aborts.
            // An incomplete compressed stream can emit useful data before its final error.
            decoded.once('end', done);
            decoded.once('error', done);
            incoming.unpipe(decoded);
            decoded.end();
          };
        }
      } catch (error) {
        captureFailed(error);
      }
      incoming.on('aborted', () => {
        finish('upstream-aborted');
        response.destroy();
      });
      // Compressed streams do not forward their source's errors automatically.
      incoming.on('error', error => {
        finish('upstream-error', error);
        response.destroy();
      });
      // Do not feed the redacted copy to OpenCode: it must see the original response.
      incoming.pipe(response, { end: false });
    });
    active.set(outgoing, { finish, flushed });
    outgoing.on('socket', socket => {
      if (finished) return;
      log('socket-assigned', { reused: outgoing.reusedSocket });
      if (!socket.connecting) return;
      if (!isIP(target.hostname)) log('dns-start', { hostname: target.hostname });
      socket.once('lookup', (error, address, family) => {
        milestone('dns-complete', 'dns_ms', { address, family, ...(error ? { error: error.message } : {}) });
      });
      socket.once('connect', () => milestone('connect', 'connect_ms'));
      socket.once('secureConnect', () => milestone('tls', 'tls_ms', { protocol: socket.getProtocol() }));
    });
    outgoing.on('finish', () => {
      milestone('upload-complete', 'upload_ms');
      // "finish" means Node handed off the upload, not that the provider has acknowledged it.
      if (!finished && metadata.headers_ms === null) log('waiting-for-provider-headers');
    });
    outgoing.on('error', error => {
      finish('upstream-error', error);
      if (!response.headersSent) response.writeHead(502).end('Provider connection failed');
      else response.destroy();
    });
    const chunks = [];
    request.on('data', chunk => {
      if (finished) return;
      chunks.push(chunk);
      requestCapture?.write(chunk);
    });
    request.on('end', () => {
      if (finished) return;
      requestCapture?.close();
      try {
        const body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
        // Full prompts are saved only by the explicit verbose request capture.
        metadata.request = {
          model: body.model, max_tokens: body.max_tokens,
          stream: body.stream, tool_choice: body.tool_choice ?? null,
        };
      } catch {
        // Non-JSON requests still reach the provider unchanged.
      }
      saveMetadata();
    });
    request.on('aborted', () => {
      finish('client-disconnected');
      outgoing.destroy();
    });
    response.on('close', () => {
      if (!response.writableFinished) {
        finish('client-disconnected');
        outgoing.destroy();
      }
    });
    request.pipe(outgoing);
  });
  try {
    await new Promise((resolve, reject) => {
      server.once('error', reject);
      server.listen(port, '127.0.0.1', resolve);
    });
  } catch (error) {
    if (traceFd !== null) fs.closeSync(traceFd);
    throw error;
  }
  const url = `http://127.0.0.1:${server.address().port}`;
  writeTrace({ event: 'recorder-ready', url });
  return {
    url,
    async close() {
      const closed = new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
      const flushed = [...active.values()].map(request => request.flushed);
      for (const [request, { finish }] of active) {
        finish('recorder-closed');
        request.destroy();
      }
      server.closeAllConnections();
      await closed;
      await Promise.all(flushed);
      writeTrace({ event: 'recorder-closed' });
      if (traceFd !== null) fs.closeSync(traceFd);
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
      verbose: { type: 'boolean', default: false },
    },
  });
  if (!values.upstream || !command.length) {
    throw new Error('Usage: provider-trace.mjs --upstream HTTPS_URL [--output DIR] [--port PORT] [--verbose] -- COMMAND [ARGS]');
  }
  const port = Number(values.port);
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error('Invalid capture port');
  const secrets = [process.env.CSCS_INFERENCE_API_KEY].filter(Boolean);
  const capture = await startProviderTrace({
    upstream: values.upstream, output: values.output, port,
    secrets, verbose: values.verbose,
  });
  let stdout;
  let stderr;
  try {
    if (values.verbose) {
      stdout = new ResponseCapture(path.join(values.output, 'opencode.events.jsonl'), secrets, chunk => process.stdout.write(chunk));
      stderr = new ResponseCapture(path.join(values.output, 'opencode.stderr.log'), secrets, chunk => process.stderr.write(chunk));
    }
    const child = spawn(command[0], command.slice(1), {
      stdio: values.verbose ? ['inherit', 'pipe', 'pipe'] : 'inherit',
    });
    const tee = (source, destination, file) => {
      source.on('data', chunk => {
        file.write(chunk);
        if (destination.writableNeedDrain) {
          source.pause();
          destination.once('drain', () => source.resume());
        }
      });
    };
    if (values.verbose) {
      tee(child.stdout, process.stdout, stdout);
      tee(child.stderr, process.stderr, stderr);
    }
    const forward = signal => child.kill(signal);
    const onTerm = () => forward('SIGTERM');
    const onInt = () => forward('SIGINT');
    process.on('SIGTERM', onTerm);
    process.on('SIGINT', onInt);
    let result;
    try {
      result = await new Promise((resolve, reject) => {
        child.once('error', reject);
        // "close", unlike "exit", includes the final stdout/stderr pipe bytes.
        child.once('close', (code, signal) => resolve({ code, signal }));
      });
    } finally {
      process.off('SIGTERM', onTerm);
      process.off('SIGINT', onInt);
    }
    process.exitCode = result.code ?? 128 + constants.signals[result.signal];
  } finally {
    stdout?.close();
    stderr?.close();
    await capture.close();
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(error => {
    console.error(redact(error.message, [process.env.CSCS_INFERENCE_API_KEY].filter(Boolean)));
    process.exitCode = 1;
  });
}
