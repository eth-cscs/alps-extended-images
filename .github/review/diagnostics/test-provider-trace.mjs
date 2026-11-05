import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import https from 'node:https';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { after, before, test } from 'node:test';
import { gzipSync } from 'node:zlib';
import { startProviderTrace } from './provider-trace.mjs';

let directory;
let key;
let cert;
before(() => {
  directory = fs.mkdtempSync(path.join(os.tmpdir(), 'review-provider-trace-'));
  const keyPath = path.join(directory, 'key.pem');
  const certPath = path.join(directory, 'cert.pem');
  execFileSync('openssl', [
    'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
    '-subj', '/CN=localhost', '-addext', 'subjectAltName=IP:127.0.0.1',
    '-keyout', keyPath, '-out', certPath,
  ], { stdio: 'ignore' });
  key = fs.readFileSync(keyPath);
  cert = fs.readFileSync(certPath);
});
after(() => fs.rmSync(directory, { recursive: true, force: true }));

async function fixture(t, handler, secrets = []) {
  const output = fs.mkdtempSync(path.join(directory, 'capture-'));
  const agent = new https.Agent({ ca: cert });
  const upstream = https.createServer({ key, cert }, handler);
  await new Promise(resolve => upstream.listen(0, '127.0.0.1', resolve));
  const trace = await startProviderTrace({
    upstream: `https://127.0.0.1:${upstream.address().port}`,
    output, port: 0, secrets, agent,
  });
  t.after(async () => {
    await trace.close();
    agent.destroy();
    upstream.closeAllConnections();
    await new Promise((resolve, reject) => upstream.close(error => error ? reject(error) : resolve()));
  });
  return { trace, output };
}

function request(url, body, headers = {}, onChunk = () => {}) {
  return new Promise((resolve, reject) => {
    const outgoing = http.request(url, {
      method: 'POST', headers: { 'content-type': 'application/json', ...headers },
    }, response => {
      response.on('data', onChunk);
      response.on('error', reject);
      response.on('end', () => resolve(response.statusCode));
    });
    outgoing.on('error', reject);
    outgoing.end(JSON.stringify(body));
  });
}

function recorded(output) {
  return {
    response: fs.readFileSync(path.join(output, '0001.response'), 'utf8'),
    metadata: JSON.parse(fs.readFileSync(path.join(output, '0001.json'), 'utf8')),
  };
}

test('redacts a credential split across chunks and omits authentication and prompts', { timeout: 10000 }, async t => {
  const secret = 'private-provider-key-12345';
  let release;
  const released = new Promise(resolve => { release = resolve; });
  const { trace, output } = await fixture(t, async (_request, response) => {
    response.writeHead(200, { 'content-type': 'text/event-stream', 'x-request-id': secret });
    response.write('data: {"text":"before ' + secret.slice(0, 9));
    // The client releases this only after the recorder has processed the first chunk.
    await released;
    response.end(secret.slice(9) + ' after"}\n\n');
  }, [secret]);
  await request(trace.url + '/v1/messages', {
    model: 'fixture', max_tokens: 16, stream: true,
    system: 'PRIVATE_SYSTEM', messages: [{ role: 'user', content: 'PRIVATE_PROMPT' }],
  }, { authorization: 'Bearer AUTH_HEADER_VALUE', 'x-api-key': secret }, release);
  const capture = recorded(output);
  assert.equal(capture.response, 'data: {"text":"before [REDACTED] after"}\n\n');
  assert.equal(capture.metadata.request_id, '[REDACTED]');
  const metadata = JSON.stringify(capture.metadata);
  for (const value of [secret, 'AUTH_HEADER_VALUE', 'PRIVATE_SYSTEM', 'PRIVATE_PROMPT']) {
    assert.equal(metadata.includes(value), false, value + ' leaked into capture metadata');
  }
  for (const name of ['0001.response', '0001.json']) {
    assert.equal(fs.statSync(path.join(output, name)).mode & 0o777, 0o600);
  }
});

test('records a compressed provider error with its HTTP status and usable timing', { timeout: 10000 }, async t => {
  const { trace, output } = await fixture(t, (_request, response) => {
    response.writeHead(429, { 'content-type': 'application/json', 'content-encoding': 'gzip' });
    response.end(gzipSync(JSON.stringify({ error: { type: 'rate_limit_error', message: 'Quota exhausted' } })));
  });
  await request(trace.url + '/v1/messages', { model: 'fixture', max_tokens: 16, stream: true });
  const capture = recorded(output);
  assert.equal(capture.metadata.status, 429);
  assert.equal(capture.metadata.outcome, 'complete');
  assert.equal(JSON.parse(capture.response).error.type, 'rate_limit_error');
  assert.ok(capture.metadata.headers_ms >= 0);
  assert.ok(capture.metadata.first_body_ms >= capture.metadata.headers_ms);
  assert.ok(capture.metadata.completed_ms >= capture.metadata.first_body_ms);
});

test('retains partial response data when the client disconnects', { timeout: 10000 }, async t => {
  let closed;
  const upstreamClosed = new Promise(resolve => { closed = resolve; });
  const { trace, output } = await fixture(t, (_request, response) => {
    response.once('close', closed);
    response.writeHead(200, { 'content-type': 'text/event-stream' });
    response.write('data: {"type":"message_start"}\n\n');
  });
  await new Promise((resolve, reject) => {
    const outgoing = http.get(trace.url + '/v1/messages', response => {
      response.once('data', () => {
        response.destroy();
        resolve();
      });
      response.on('error', error => {
        if (error.code !== 'ECONNRESET') reject(error);
      });
    });
    outgoing.on('error', reject);
  });
  await upstreamClosed;
  const capture = recorded(output);
  assert.equal(capture.metadata.outcome, 'client-disconnected');
  assert.equal(capture.response, 'data: {"type":"message_start"}\n\n');
  assert.ok(capture.metadata.completed_ms >= capture.metadata.first_body_ms);
});
