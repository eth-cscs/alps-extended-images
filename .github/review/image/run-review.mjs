import { spawn } from 'node:child_process';
import { constants } from 'node:os';
import { once } from 'node:events';
import { readFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';

const INSPECTION_TOOLS = new Set(['read', 'grep', 'shell', 'execute']);

function successfulInspection(event) {
  const state = event.part?.state;
  const details = state?.metadata?.metadata;
  return event.type === 'tool_use' && INSPECTION_TOOLS.has(event.part?.tool) &&
    state?.status === 'completed' && details?.error !== true &&
    (typeof details?.exit !== 'number' || details.exit === 0);
}

// Kill the CLI's private server and tool descendants as well as the CLI itself.
// A phase deadline must not leave an old provider request running during finalization.
function stopGroup(child, signal) {
  if (!child.pid) return;
  try {
    process.kill(-child.pid, signal);
  } catch (error) {
    if (error.code !== 'ESRCH') throw error;
  }
}

async function emit(line) {
  if (!process.stdout.write(line + '\n')) await once(process.stdout, 'drain');
}

async function phase(name, agent, prompt, session, budget, control) {
  if (control.interrupted) throw new Error('Review interrupted');
  const marker = (state, extra = {}) => emit(JSON.stringify({
    type: 'review_phase', phase: name, state, ...(session ? { sessionID: session } : {}), ...extra,
  }));
  await marker('start');
  const args = ['run', '--standalone', '--auto', '--agent', agent, '--format', 'json', '--thinking'];
  if (session) args.push('--session', session);
  else args.push('--title', 'Read-only PR review', '--file', '/input/context.md');
  if (name === 'final') {
    // Deny-all permissions still leave skill discovery advertised in this CLI version.
    // Force provider-level tool_choice=none while deriving the model from its single config source.
    args.push('--model', `${control.config.model}#review-final`);
  }
  args.push(prompt);
  const child = spawn('opencode', args, { detached: true, stdio: ['ignore', 'pipe', 'inherit'] });
  let escalation;
  const stop = signal => {
    stopGroup(child, signal);
    escalation ??= setTimeout(() => stopGroup(child, 'SIGKILL'), budget.killGraceMs);
  };
  control.stop = stop;
  let startupError;
  child.once('error', error => { startupError = error; });
  const closed = new Promise(resolve => child.once('close', (code, signal) => resolve({ code, signal })));
  let inspected = false;
  let lastFinish;
  let steps = 0;
  let expired = false;
  const deadline = setTimeout(() => {
    expired = true;
    console.error(`review phases: ${name} deadline reached; stopping OpenCode`);
    stop('SIGTERM');
  }, budget.timeoutMs);
  try {
    for await (const line of createInterface({ input: child.stdout, crlfDelay: Infinity })) {
      if (!line.trim()) continue;
      let event;
      try {
        event = JSON.parse(line);
      } catch {
        throw new Error(`${name} emitted invalid JSON`);
      }
      if (!event || typeof event !== 'object' || typeof event.type !== 'string') {
        throw new Error(`${name} emitted an invalid event`);
      }
      // Phase boundaries belong to this runner, never to the model or CLI output.
      if (event.type === 'review_phase') throw new Error(`${name} emitted an unexpected phase marker`);
      // A controlled deadline can make the CLI emit an abort error.
      // Do not put that error into the publishable event stream; the phase end
      // marker records incomplete inspection and the deadline is logged above.
      if (event.type === 'error' && expired) continue;
      await emit(line);
      if (event.type === 'error') throw new Error(`${name} reported an error`);
      if (typeof event.sessionID !== 'string' || !event.sessionID) {
        throw new Error(`${name} emitted an event without a session`);
      }
      session ??= event.sessionID;
      if (event.sessionID !== session) throw new Error(`${name} changed the review session`);
      inspected ||= successfulInspection(event);
      if (event.type === 'step_finish') lastFinish = event.part?.reason;
      if (event.type === 'step_start') steps++;
      // Permissions remove tools from finalization; reject any violation rather than publishing it.
      if (name === 'final' && event.type === 'tool_use') {
        throw new Error('Finalization attempted to use a tool');
      }
    }
    const result = await closed;
    if (startupError) throw new Error(`${name} could not start OpenCode (${startupError.code})`);
    if (control.interrupted) throw new Error('Review interrupted');
    if (expired && name === 'final') throw new Error('Finalization exceeded its deadline');
    if (!expired && result.code !== 0) throw new Error(`${name} failed (${result.signal || result.code})`);
    if (!session) throw new Error(`${name} produced no session`);
    // Text copied from the supplied context is not evidence that inspection took place.
    if (name === 'inspection' && !inspected) throw new Error('Inspection completed without a successful inspection tool call');
    // At the iteration cap OpenCode emits its own summary, not proof of full coverage.
    const incomplete = name === 'inspection' &&
      (expired || lastFinish === 'length' || steps >= budget.steps);
    await marker('end', incomplete ? { incomplete: true } : {});
    return { session, incomplete };
  } finally {
    clearTimeout(deadline);
    stop('SIGTERM');
    await closed;
    // Remove descendants even if the CLI exited before its server or tools.
    stopGroup(child, 'SIGKILL');
    clearTimeout(escalation);
    control.stop = undefined;
  }
}

export async function runReview() {
  const configDirectory = `${process.env.XDG_CONFIG_HOME}/opencode`;
  const config = JSON.parse(await readFile(`${configDirectory}/opencode.json`, 'utf8'));
  const limits = JSON.parse(await readFile(`${configDirectory}/limits.json`, 'utf8'));
  const seconds = ['inspection_seconds', 'finalization_seconds', 'termination_grace_seconds'];
  if (!limits || seconds.some(key => typeof limits[key] !== 'number' || !Number.isFinite(limits[key]) || limits[key] <= 0)) {
    throw new Error('limits.json must define positive finite phase deadlines and termination grace');
  }
  const inspectionMs = limits.inspection_seconds * 1000;
  const finalMs = limits.finalization_seconds * 1000;
  const killGraceMs = limits.termination_grace_seconds * 1000;
  const inspectionSteps = config.agents['grumpy-reviewer'].steps;
  const control = { config };
  const handlers = new Map(['SIGINT', 'SIGTERM'].map(signal => [signal, () => {
    control.interrupted = signal;
    control.stop?.(signal);
  }]));
  for (const [signal, handler] of handlers) process.on(signal, handler);
  try {
    const { session, incomplete } = await phase('inspection', 'grumpy-reviewer',
      'Begin the inspection phase under the reviewer policy using the supplied context and /input/pr-source. ' +
      `You have at most ${inspectionSteps} model iterations and ${Math.floor(inspectionMs / 1000)} seconds for inspection.`,
      undefined, { timeoutMs: inspectionMs, killGraceMs, steps: inspectionSteps }, control);
    // A new message resets the agent allowance. Finalization has no tools and a separate time reserve.
    // Its step reserve avoids OpenCode's generic limit-summary prompt on the first response.
    // Reapply the same policy next to the final task. Long tool histories and the
    // CLI's limit-summary message must not become a competing review policy.
    const policy = await readFile(new URL('./grumpy-reviewer.md', import.meta.url), 'utf8');
    await phase('final', 'review-finalizer',
      'Begin finalization under the reviewer policy below and return only the final review body. ' +
      (incomplete ? 'Inspection was cut short; apply the policy for incomplete coverage. ' : '') +
      `\n\n${policy}`,
      session, { timeoutMs: finalMs, killGraceMs }, control);
  } catch (error) {
    if (control.interrupted) error.exitCode = 128 + constants.signals[control.interrupted];
    throw error;
  } finally {
    for (const [signal, handler] of handlers) process.removeListener(signal, handler);
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    await runReview();
  } catch (error) {
    console.error(`review phases: ${error.message}`);
    process.exitCode = error.exitCode || 1;
  }
}
