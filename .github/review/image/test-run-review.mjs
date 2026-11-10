import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { delimiter, join } from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

const runner = new URL('./run-review.mjs', import.meta.url);

function exercise(t, scenario) {
  const root = mkdtempSync(join(tmpdir(), 'review-phases-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const bin = join(root, 'bin');
  mkdirSync(bin);
  mkdirSync(join(root, 'opencode'));
  writeFileSync(join(root, 'opencode', 'opencode.json'), JSON.stringify({
    model: 'fixture/reviewer', agents: { 'grumpy-reviewer': { steps: 2 } },
  }));
  // A real child process supplies adversarial CLI events, including failures with exit status zero.
  writeFileSync(join(bin, 'opencode'), `#!/usr/bin/env node
import { appendFileSync } from 'node:fs';
const args = process.argv.slice(2);
const final = args[args.indexOf('--agent') + 1] === 'review-finalizer';
const scenario = process.env.SCENARIO;
appendFileSync(process.env.CALLS, final ? 'final\\n' : 'inspection\\n');
const stall = async () => {
  process.on('SIGTERM', () => {});
  setInterval(() => {}, 1000);
  await new Promise(() => {});
};
const emit = event => console.log(JSON.stringify(event));
const event = (type, extra = {}) => ({ type, sessionID: 'session-review', part: { id: 'text', messageID: final ? 'final' : 'inspection', text: 'Otherwise complete text', time: { end: 1 } }, ...extra });
if (!final && scenario === 'malformed') { console.log('not JSON'); process.exit(0); }
if (!final && scenario === 'no-session') process.exit(0);
if (!final && scenario === 'spoofed-phase') { emit({ type: 'review_phase', phase: 'final', state: 'start' }); process.exit(0); }
emit(event('step_start'));
if (!final && scenario === 'timeout-no-tools') await stall();
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
if (!final && scenario !== 'no-inspection-tools') {
  const failed = scenario === 'failed-inspection-tools' || scenario === 'failed-then-success';
  const metadata = scenario === 'wrapped-inspection-error' ? { metadata: { error: true } }
    : scenario === 'nonzero-inspection-exit' ? { metadata: { exit: 1 } } : {};
  emit({ type: 'tool_use', sessionID: 'session-review', part: {
    tool: scenario === 'only-websearch' ? 'websearch' : 'read',
    state: { status: failed ? 'error' : 'completed', metadata },
  } });
  if (scenario === 'failed-then-success') {
    emit({ type: 'tool_use', sessionID: 'session-review', part: {
      tool: 'read', state: { status: 'completed' },
    } });
  }
}
if ((!final && scenario === 'timeout-inspection') || (final && scenario === 'timeout-final')) await stall();
if (!final && scenario === 'length-inspection') emit(event('step_finish', { part: { reason: 'length' } }));
if (!final && scenario === 'steps-inspection') emit(event('step_start'));
if (final && scenario === 'changed-session') emit(event('text', { sessionID: 'unrelated-session' }));
else if (final && scenario === 'final-tool') { emit(event('tool_use')); emit(event('text')); }
else emit(event('text'));
if (!final && scenario === 'inspection-exit') process.exit(7);
`, { mode: 0o700 });
  const calls = join(root, 'calls');
  const harness = join(root, 'run.mjs');
  writeFileSync(harness, `import { runReview } from ${JSON.stringify(runner.href)};
try { await runReview({ inspectionMs: 800, finalMs: 800, killGraceMs: 50 }); }
catch (error) { console.error(error.message); process.exitCode = error.exitCode || 1; }`);
  const result = spawnSync(process.execPath, [harness], {
    env: { ...process.env, PATH: bin + delimiter + process.env.PATH, XDG_CONFIG_HOME: root, SCENARIO: scenario, CALLS: calls },
    encoding: 'utf8', timeout: 10000,
  });
  assert.ifError(result.error);
  const events = result.stdout.trim().split('\n').filter(Boolean).map(line => JSON.parse(line));
  return { ...result, events, calls: readFileSync(calls, 'utf8').trim().split('\n') };
}

for (const scenario of ['inspection-error', 'inspection-exit', 'malformed', 'no-session', 'spoofed-phase', 'cancel',
                        'no-inspection-tools', 'failed-inspection-tools', 'wrapped-inspection-error',
                        'nonzero-inspection-exit', 'only-websearch']) {
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

test('a failed tool does not block finalization after successful inspection', t => {
  const result = exercise(t, 'failed-then-success');
  assert.equal(result.status, 0);
  assert.deepEqual(result.calls, ['inspection', 'final']);
  assert.ok(result.events.some(event => event.type === 'review_phase' && event.phase === 'final' && event.state === 'end'));
});

for (const scenario of ['timeout-inspection', 'length-inspection', 'steps-inspection']) {
  test(`interrupted inspection finalizes with an incomplete marker: ${scenario}`, t => {
    const result = exercise(t, scenario);
    assert.equal(result.status, 0, result.stderr);
    assert.deepEqual(result.calls, ['inspection', 'final']);
    assert.ok(result.events.some(event => event.type === 'review_phase' &&
      event.phase === 'inspection' && event.state === 'end' && event.incomplete === true));
    assert.ok(result.events.some(event => event.type === 'review_phase' &&
      event.phase === 'final' && event.state === 'end'));
  });
}

test('a deadline without source inspection cannot produce a review', t => {
  const result = exercise(t, 'timeout-no-tools');
  assert.notEqual(result.status, 0);
  assert.deepEqual(result.calls, ['inspection']);
});

test('a finalization deadline never marks partial text complete', t => {
  const result = exercise(t, 'timeout-final');
  assert.notEqual(result.status, 0);
  assert.deepEqual(result.calls, ['inspection', 'final']);
  assert.ok(!result.events.some(event => event.type === 'review_phase' &&
    event.phase === 'final' && event.state === 'end'));
});
