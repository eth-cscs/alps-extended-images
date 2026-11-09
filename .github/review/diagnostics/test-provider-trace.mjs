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
    '-subj', '/CN=localhost', '-addext', 'subjectAltName=IP:127.0.0.1,DNS:localhost',
    '-keyout', keyPath, '-out', certPath,
  ], { stdio: 'ignore' });
  key = fs.readFileSync(keyPath);
  cert = fs.readFileSync(certPath);
});
after(() => fs.rmSync(directory, { recursive: true, force: true }));

async function fixture(t, handler, secrets = [], options = {}) {
  const { hostname = '127.0.0.1', agentOptions = {}, ...traceOptions } = options;
  const output = fs.mkdtempSync(path.join(directory, 'capture-'));
  const agent = new https.Agent({ ca: cert, ...agentOptions });
  const upstream = https.createServer({ key, cert }, handler);
  await new Promise(resolve => upstream.listen(0, '127.0.0.1', resolve));
  const trace = await startProviderTrace({
    upstream: `https://${hostname}:${upstream.address().port}`,
    ...traceOptions, output, port: 0, secrets, agent,
  });
  t.after(async () => {
    await trace.close();
    agent.destroy();
    upstream.closeAllConnections();
    await new Promise((resolve, reject) => upstream.close(error => error ? reject(error) : resolve()));
  });
  return { trace, output };
}

function request(url, body, headers = {}, onChunk = () => {}, onHeaders = () => {}) {
  return new Promise((resolve, reject) => {
    const outgoing = http.request(url, {
      method: 'POST', headers: { 'content-type': 'application/json', ...headers },
    }, response => {
      onHeaders(response);
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

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

function traceSink() {
  const events = [];
  const waiting = new Set();
  return {
    events,
    onTrace(line) {
      const event = JSON.parse(line);
      events.push(event);
      for (const waiter of waiting) {
        if (event.event === waiter.name && waiter.predicate(event)) {
          waiting.delete(waiter);
          waiter.resolve(event);
        }
      }
    },
    until(name, predicate = () => true) {
      const event = events.find(event => event.event === name && predicate(event));
      if (event) return Promise.resolve(event);
      return new Promise(resolve => waiting.add({ name, predicate, resolve }));
    },
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
  assert.equal(fs.existsSync(path.join(output, '0001.request')), false);
  assert.equal(fs.existsSync(path.join(output, 'trace.log')), false);
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

test('verbose capture exposes upload, connection, headers and decoded body before completion without altering provider bytes', { timeout: 10000 }, async t => {
  const secret = 'verbose-inference-key-12345';
  const prompt = 'FULL_PRIVATE_PROMPT_AND_TOOL_PAYLOAD';
  const body = { model: 'fixture', stream: true, messages: [{ role: 'user', content: prompt }] };
  const upstreamRequest = deferred();
  const resolveDns = deferred();
  const sendHeaders = deferred();
  const sendBody = deferred();
  const complete = deferred();
  const sink = traceSink();
  const chunks = [];
  const first = 'data: {"text":"full response ' + secret.slice(0, 9);
  const last = secret.slice(9) + ' and tool output"}\n\n';
  let clientHeaders;
  const { trace, output } = await fixture(t, async (incoming, response) => {
    const uploaded = [];
    for await (const chunk of incoming) uploaded.push(chunk);
    upstreamRequest.resolve({ body: Buffer.concat(uploaded).toString(), headers: incoming.headers });
    await sendHeaders.promise;
    response.writeHead(202, {
      'content-type': 'text/event-stream',
      'set-cookie': 'RESPONSE_COOKIE',
      authorization: 'Bearer RESPONSE_AUTH',
      'x-trace-value': secret,
    });
    response.flushHeaders();
    await sendBody.promise;
    response.write(first);
    await complete.promise;
    response.end(last);
  }, [secret], {
    verbose: true, onTrace: sink.onTrace, hostname: 'localhost',
    agentOptions: {
      lookup: (_hostname, options, callback) => {
        resolveDns.promise.then(() => options.all
          ? callback(null, [{ address: '127.0.0.1', family: 4 }])
          : callback(null, '127.0.0.1', 4));
      },
    },
  });
  const result = request(trace.url + '/v1/messages', body, {
    authorization: 'Bearer REQUEST_AUTH', 'x-api-key': secret, cookie: 'REQUEST_COOKIE',
  }, chunk => chunks.push(chunk), response => { clientHeaders = response.headers; });
  await sink.until('dns-start');
  resolveDns.resolve();
  await sink.until('waiting-for-provider-headers');
  const uploaded = await upstreamRequest.promise;
  assert.equal(uploaded.body, JSON.stringify(body));
  assert.equal(uploaded.headers.authorization, 'Bearer REQUEST_AUTH');
  assert.equal(uploaded.headers['x-api-key'], secret);
  assert.equal(fs.readFileSync(path.join(output, '0001.request'), 'utf8'), JSON.stringify(body));
  assert.equal(recorded(output).metadata.headers_ms, null);
  assert.equal(sink.events.some(event => event.event === 'provider-headers'), false);
  for (const event of ['request-start', 'dns-start', 'dns-complete', 'connect', 'tls', 'upload-complete']) {
    assert.ok(sink.events.some(record => record.event === event), event);
  }
  sendHeaders.resolve();
  await sink.until('waiting-for-provider-body');
  assert.equal(recorded(output).metadata.status, 202);
  assert.equal(recorded(output).metadata.first_body_ms, null);
  sendBody.resolve();
  await sink.until('response-body');
  assert.equal(recorded(output).metadata.completed_ms, null);
  assert.equal(recorded(output).response, 'data: {"text":"full response ');
  assert.equal(sink.events.some(event => event.event === 'complete'), false);
  complete.resolve();
  assert.equal(await result, 202);
  assert.equal(Buffer.concat(chunks).toString(), first + last);
  assert.deepEqual(clientHeaders['set-cookie'], ['RESPONSE_COOKIE']);
  assert.equal(clientHeaders.authorization, 'Bearer RESPONSE_AUTH');
  const capture = recorded(output);
  assert.equal(capture.response, 'data: {"text":"full response [REDACTED] and tool output"}\n\n');
  assert.equal(capture.metadata.outcome, 'complete');
  assert.ok(capture.metadata.upload_ms >= capture.metadata.tls_ms);
  assert.ok(capture.metadata.headers_ms >= capture.metadata.upload_ms);
  assert.ok(capture.metadata.first_body_ms >= capture.metadata.headers_ms);
  const traceText = fs.readFileSync(path.join(output, 'trace.log'), 'utf8');
  assert.equal(traceText, sink.events.map(event => JSON.stringify(event) + '\n').join(''));
  assert.ok(sink.events.filter(event => event.event === 'request-body').map(event => event.text).join('').includes(prompt));
  assert.equal(sink.events.filter(event => event.event === 'response-body').map(event => event.text).join(''), capture.response);
  for (const credential of [secret, 'REQUEST_AUTH', 'REQUEST_COOKIE', 'RESPONSE_COOKIE', 'RESPONSE_AUTH']) {
    assert.equal(traceText.includes(credential), false, 'credential leaked into verbose trace');
    assert.equal(JSON.stringify(capture.metadata).includes(credential), false, 'credential leaked into metadata');
  }
  for (const name of ['trace.log', '0001.request', '0001.response', '0001.json']) {
    assert.equal(fs.statSync(path.join(output, name)).mode & 0o777, 0o600);
  }
});

test('verbose capture streams an unfinished upload and retains its redacted partial request after disconnect', { timeout: 10000 }, async t => {
  const secret = 'partial-upload-inference-key';
  const sink = traceSink();
  const { trace, output } = await fixture(t, incoming => incoming.resume(), [secret], {
    verbose: true, onTrace: sink.onTrace,
  });
  const outgoing = http.request(trace.url + '/v1/messages', { method: 'POST' });
  outgoing.on('error', () => {});
  outgoing.write('PRIVATE_PARTIAL_UPLOAD ' + secret.slice(0, 10));
  await sink.until('request-body');
  assert.equal(fs.readFileSync(path.join(output, '0001.request'), 'utf8'), 'PRIVATE_PARTIAL_UPLOAD ');
  assert.equal(sink.events.some(event => event.event === 'upload-complete'), false);
  outgoing.write(secret.slice(10));
  await sink.until('request-body', event => event.text === '[REDACTED]');
  outgoing.destroy();
  await sink.until('complete');
  assert.equal(fs.readFileSync(path.join(output, '0001.request'), 'utf8'), 'PRIVATE_PARTIAL_UPLOAD [REDACTED]');
  assert.equal(recorded(output).metadata.outcome, 'client-disconnected');
  assert.equal(sink.events.some(event => event.event === 'upload-complete'), false);
  assert.equal(recorded(output).response, '');
  assert.equal(fs.readFileSync(path.join(output, 'trace.log'), 'utf8').includes(secret), false);
});

test('verbose capture preserves decoded error text and the original compressed HTTP response', { timeout: 10000 }, async t => {
  const secret = 'compressed-response-inference-key';
  const text = JSON.stringify({ error: { message: 'FULL_PROVIDER_ERROR ' + secret } });
  const encoded = gzipSync(text);
  const sink = traceSink();
  const { trace, output } = await fixture(t, (_incoming, response) => {
    response.writeHead(429, { 'content-type': 'application/json', 'content-encoding': 'gzip' });
    response.end(encoded);
  }, [secret], { verbose: true, onTrace: sink.onTrace });
  const chunks = [];
  assert.equal(await request(trace.url, { model: 'fixture' }, {}, chunk => chunks.push(chunk)), 429);
  assert.deepEqual(Buffer.concat(chunks), encoded);
  const expected = text.replace(secret, '[REDACTED]');
  assert.equal(recorded(output).response, expected);
  assert.equal(sink.events.filter(event => event.event === 'response-body').map(event => event.text).join(''), expected);
});

test('verbose capture retains a partial response on provider abort and reports the failure', { timeout: 10000 }, async t => {
  const secret = 'partial-response-inference-key';
  const sink = traceSink();
  const providerResponse = deferred();
  const partial = 'data: {"text":"PARTIAL_PROVIDER_OUTPUT ' + secret.slice(0, 8);
  const { trace, output } = await fixture(t, (_incoming, response) => {
    providerResponse.resolve(response);
    response.writeHead(200, { 'content-type': 'text/event-stream' });
    response.write(partial);
  }, [secret], { verbose: true, onTrace: sink.onTrace });
  const received = [];
  const clientSawBody = deferred();
  const result = request(trace.url, { model: 'fixture' }, {}, chunk => {
    received.push(chunk);
    clientSawBody.resolve();
  });
  const failed = assert.rejects(result, { code: 'ECONNRESET' });
  await sink.until('response-body');
  await clientSawBody.promise;
  (await providerResponse.promise).destroy();
  const completed = await sink.until('complete');
  await failed;
  assert.equal(completed.outcome, 'upstream-aborted');
  assert.equal(recorded(output).metadata.outcome, 'upstream-aborted');
  assert.equal(recorded(output).response, partial);
  assert.equal(Buffer.concat(received).toString(), partial);
  assert.equal(sink.events.filter(event => event.event === 'response-body').map(event => event.text).join(''), partial);
});

test('verbose capture reports an upstream error before headers without inventing provider progress', { timeout: 10000 }, async t => {
  const sink = traceSink();
  const body = { model: 'fixture', messages: [{ role: 'user', content: 'RETAIN_FAILED_REQUEST' }] };
  const { trace, output } = await fixture(t, async incoming => {
    for await (const _chunk of incoming) {}
    incoming.socket.destroy();
  }, [], { verbose: true, onTrace: sink.onTrace });
  assert.equal(await request(trace.url, body), 502);
  const capture = recorded(output);
  assert.equal(capture.metadata.outcome, 'upstream-error');
  assert.equal(capture.metadata.headers_ms, null);
  assert.equal(capture.metadata.first_body_ms, null);
  assert.equal(capture.metadata.received_bytes, 0);
  assert.equal(capture.response, '');
  assert.equal(fs.readFileSync(path.join(output, '0001.request'), 'utf8'), JSON.stringify(body));
  assert.ok(sink.events.some(event => event.event === 'waiting-for-provider-headers'));
  assert.equal(sink.events.some(event => event.event === 'provider-headers' || event.event === 'first-body' || event.event === 'response-body'), false);
  assert.ok(sink.events.find(event => event.event === 'complete').error);
});

test('a decoding failure remains visible without truncating or replacing the provider response', { timeout: 10000 }, async t => {
  const sink = traceSink();
  const finish = deferred();
  const first = Buffer.from('not-a-valid-gzip-stream');
  const last = Buffer.from(' but the original response must still reach OpenCode');
  const { trace, output } = await fixture(t, async (_incoming, response) => {
    response.writeHead(503, { 'content-type': 'application/json', 'content-encoding': 'gzip' });
    response.write(first);
    await finish.promise;
    response.end(last);
  }, [], { verbose: true, onTrace: sink.onTrace });
  const received = [];
  const result = request(trace.url, { model: 'fixture' }, {}, chunk => received.push(chunk));
  await sink.until('capture-error');
  assert.equal(recorded(output).metadata.completed_ms, null);
  assert.equal(sink.events.some(event => event.event === 'complete'), false);
  finish.resolve();
  assert.equal(await result, 503);
  assert.deepEqual(Buffer.concat(received), Buffer.concat([first, last]));
  assert.equal(recorded(output).metadata.outcome, 'capture-error');
  assert.ok(recorded(output).metadata.error);
});

test('a truncated compressed response drains already received plaintext before closing its partial capture', { timeout: 10000 }, async t => {
  const secret = 'partial-compressed-inference-key';
  const plaintext = 'PARTIAL_COMPRESSED_PROVIDER_OUTPUT ' + secret;
  // Omit gzip's trailer so normal decompression cannot report successful completion.
  const compressed = gzipSync(plaintext).subarray(0, -8);
  const sink = traceSink();
  const providerResponse = deferred();
  const { trace, output } = await fixture(t, (_incoming, response) => {
    providerResponse.resolve(response);
    response.writeHead(200, { 'content-type': 'text/event-stream', 'content-encoding': 'gzip' });
    response.write(compressed);
  }, [secret], { verbose: true, onTrace: sink.onTrace });
  const failed = assert.rejects(request(trace.url, { model: 'fixture' }));
  await sink.until('first-body');
  (await providerResponse.promise).destroy();
  await sink.until('complete');
  await failed;
  assert.equal(recorded(output).metadata.outcome, 'upstream-aborted');
  assert.equal(recorded(output).response, plaintext.replace(secret, '[REDACTED]'));
  assert.equal(fs.readFileSync(path.join(output, 'trace.log'), 'utf8').includes(secret), false);
});
