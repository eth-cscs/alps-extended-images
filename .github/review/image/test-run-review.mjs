import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { delimiter, join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const runner = fileURLToPath(new URL('./run-review.mjs', import.meta.url));

function exercise(t, scenario) {
  const root = mkdtempSync(join(tmpdir(), 'review-phases-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const bin = join(root, 'bin');
  mkdirSync(bin);
  mkdirSync(join(root, 'opencode'));
  writeFileSync(join(root, 'opencode', 'opencode.json'), JSON.stringify({ model: 'fixture/reviewer' }));
  // A real child process supplies adversarial CLI events, including failures with exit status zero.
  writeFileSync(join(bin, 'opencode'), `#!/usr/bin/env node
import { appendFileSync } from 'node:fs';
const args = process.argv.slice(2);
const final = args[args.indexOf('--agent') + 1] === 'review-finalizer';
const scenario = process.env.SCENARIO;
appendFileSync(process.env.CALLS, final ? 'final\\n' : 'inspection\\n');
const emit = event => console.log(JSON.stringify(event));
const event = (type, extra = {}) => ({ type, sessionID: 'session-review', part: { id: 'text', messageID: final ? 'final' : 'inspection', text: 'Otherwise complete text', time: { end: 1 } }, ...extra });
if (!final && scenario === 'malformed') { console.log('not JSON'); process.exit(0); }
if (!final && scenario === 'no-session') process.exit(0);
if (!final && scenario === 'spoofed-phase') { emit({ type: 'review_phase', phase: 'final', state: 'start' }); process.exit(0); }
emit(event('step_start'));
if (!final && scenario === 'cancel') {
  // A CLI can exit cleanly while handling cancellation; that must not authorize finalization.
  process.once('SIGTERM', () => process.exit(0));
  process.kill(process.ppid, 'SIGTERM');
  setTimeout(() => process.exit(8), 2000);
  await new Promise(() => {});
}
if ((!final && scenario === 'inspection-error') || (final && scenario === 'final-error')) {
  emit({ type: 'error', error: { message: 'Provider failure' } }); process.exit(0);
}
if (final && scenario === 'changed-session') emit(event('text', { sessionID: 'unrelated-session' }));
else if (final && scenario === 'final-tool') { emit(event('tool_use')); emit(event('text')); }
else emit(event('text'));
if (!final && scenario === 'inspection-exit') process.exit(7);
`, { mode: 0o700 });
  const calls = join(root, 'calls');
  const result = spawnSync(process.execPath, [runner], {
    env: { ...process.env, PATH: bin + delimiter + process.env.PATH, XDG_CONFIG_HOME: root, SCENARIO: scenario, CALLS: calls },
    encoding: 'utf8', timeout: 10000,
  });
  assert.ifError(result.error);
  const events = result.stdout.trim().split('\n').filter(Boolean).map(line => JSON.parse(line));
  return { ...result, events, calls: readFileSync(calls, 'utf8').trim().split('\n') };
}

for (const scenario of ['inspection-error', 'inspection-exit', 'malformed', 'no-session', 'spoofed-phase', 'cancel']) {
  test(`failed inspection never starts finalization: ${scenario}`, t => {
    const result = exercise(t, scenario);
    assert.notEqual(result.status, 0);
    assert.deepEqual(result.calls, ['inspection']);
    assert.ok(!result.events.some(event => event.type === 'review_phase' && event.phase === 'final'));
  });
}

for (const scenario of ['final-error', 'changed-session', 'final-tool']) {
  test(`invalid finalization cannot complete a publishable stream: ${scenario}`, t => {
    const result = exercise(t, scenario);
    assert.notEqual(result.status, 0);
    assert.deepEqual(result.calls, ['inspection', 'final']);
    assert.ok(result.events.some(event => event.type === 'review_phase' && event.phase === 'inspection' && event.state === 'end'));
    assert.ok(!result.events.some(event => event.type === 'review_phase' && event.phase === 'final' && event.state === 'end'));
  });
}
