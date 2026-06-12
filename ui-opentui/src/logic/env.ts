/**
 * env — shared boolean env-flag parsing (one source for the TRUE/FALSE regexes).
 *
 * Recognized truthy values: 1/true/yes/on; falsy: 0/false/no/off (case-insensitive,
 * surrounding whitespace trimmed). Anything else (incl. unset) is "unrecognized".
 */
export const TRUE_RE = /^(?:1|true|yes|on)$/i
export const FALSE_RE = /^(?:0|false|no|off)$/i

/** Parse a boolean env var; returns `fallback` when unset/unrecognized. */
export function envFlag(value: string | undefined, fallback: boolean): boolean {
  const v = value?.trim() ?? ''
  if (TRUE_RE.test(v)) return true
  if (FALSE_RE.test(v)) return false
  return fallback
}

/**
 * The diagnostics master switch — `HERMES_TUI_DIAGNOSTICS` (default OFF).
 *
 * Gates the developer/profiling surface a regular user should never trip
 * over: the diagnostic slash commands (`/mem`, `/heapdump`) and the default
 * for `HERMES_TUI_WINDOW_STATS` (which can still be set individually). It is
 * an enable switch, not a secret: anyone CAN set it (support flows say
 * "relaunch with HERMES_TUI_DIAGNOSTICS=1"), it just keeps the day-to-day
 * surface clean. Read per call so tests (and long-lived processes whose
 * wrapper mutates env before launch) see the live value.
 */
export function diagnosticsEnabled(): boolean {
  return envFlag(process.env.HERMES_TUI_DIAGNOSTICS, false)
}

/**
 * Parse `HERMES_TUI_TOOL_OUTPUT_LINES` (a TUI-only env var — deliberately NOT
 * a config.yaml knob): how many output lines an expanded tool body shows.
 * UNSET → Infinity (UNLIMITED — expanded tool output is uncapped by default;
 * setting the var is how you RESTORE a cap, e.g. `=200`). A positive integer
 * → that cap. `0` → Infinity too (back-compat: it was the old opt-in
 * "unlimited" value). Garbage → Infinity (unrecognized ≙ no cap asked for —
 * the semantic is "cap only when the user asked for one").
 */
export function envOutputLines(value: string | undefined): number {
  const v = value?.trim() ?? ''
  if (!/^\d+$/.test(v)) return Number.POSITIVE_INFINITY
  const n = Number.parseInt(v, 10)
  return n === 0 ? Number.POSITIVE_INFINITY : n
}

/**
 * Default visible-height cap for the composer textarea, in rows (Ink composer
 * parity — 8 lines, ref feature request #10418). Beyond this the textarea
 * scrolls INTERNALLY (the native edit buffer keeps the cursor in view).
 */
export const COMPOSER_MAX_ROWS = 8

/**
 * Parse `HERMES_TUI_COMPOSER_ROWS` (a TUI-only env var — deliberately NOT a
 * config.yaml knob): the composer's visible-height cap before internal scroll
 * kicks in. A positive integer → that cap; unset / `0` / garbage → the
 * COMPOSER_MAX_ROWS default.
 */
export function envComposerRows(value: string | undefined): number {
  const v = value?.trim() ?? ''
  if (!/^\d+$/.test(v)) return COMPOSER_MAX_ROWS
  const n = Number.parseInt(v, 10)
  return n > 0 ? n : COMPOSER_MAX_ROWS
}

/**
 * Whether NO line cap applies (unset / `0` / unparseable). When unlimited,
 * the store prefers the always-full raw `result` over a gateway tail-capped
 * `result_text` — an "unlimited" view of a tail would still be missing its
 * head — see store.ts tool.complete. With an explicit finite cap the gateway
 * tail (+ honest omitted note) is kept: the user asked for a bounded view.
 */
export function envOutputUnlimited(value: string | undefined): boolean {
  return envOutputLines(value) === Number.POSITIVE_INFINITY
}
