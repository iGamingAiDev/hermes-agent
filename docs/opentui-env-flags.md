# OpenTUI env flags — the consolidated ledger

Every environment variable the OpenTUI TUI reads (grep-verified 2026-06-12),
classified by who should ever touch it. The design rule shipped with this doc:
**regular users see zero diagnostic surface by default; one master switch
(`HERMES_TUI_DIAGNOSTICS=1`) turns all of it on when needed.**

## 1. The master switch

| var | default | effect |
|---|---|---|
| `HERMES_TUI_DIAGNOSTICS` | **off** | Enables the diagnostic slash commands (`/mem`, `/heapdump`) — hidden from `/help` and completion otherwise, and invoking them while off prints the enable hint rather than executing. Also flips the *default* of `HERMES_TUI_WINDOW_STATS` to on. Not a secret — support flows are "relaunch with `HERMES_TUI_DIAGNOSTICS=1`". |

## 2. User-facing configuration (fine to document publicly)

| var | default | effect |
|---|---|---|
| `HERMES_TUI_ENGINE` | auto (`opentui` if Node≥26.3 + built, else `ink`) | Engine pick; also `display.tui_engine` in config.yaml. |
| `HERMES_TUI_MOUSE` | on (launcher sets it) | Mouse support (wheel scroll, selection, click-to-expand). |
| `HERMES_TUI_MAX_MESSAGES` | ceiling | Scrollback rows kept in the TUI. Can LOWER the ceiling, never raise: 3000 with windowing, 1000 with windowing off (handle-table safety). |
| `HERMES_TUI_TOOL_OUTPUT_LINES` | unlimited | Cap expanded tool-output lines (set a number to restore a cap). |
| `HERMES_TUI_COMPOSER_ROWS` | default rows | Composer height. |

## 3. Escape hatches & tuning (dev-facing, individually settable)

| var | default | effect |
|---|---|---|
| `HERMES_TUI_WINDOWING` | **on** | `0` = bit-exact pre-windowing renderer (every row mounts; cap clamps back to 1000). The A/B + regression escape hatch. |
| `HERMES_TUI_WINDOW_IDLE_MS` | ~1000 | Idle-measure pulse cadence (the spacer-exactness march). Test knob. |
| `HERMES_TUI_WINDOW_STATS` | = `HERMES_TUI_DIAGNOSTICS` | Exposes live/peak mounted-row counters (`globalThis.__hermesTuiWindowStats`) for bench/live-attach reads. |
| `HERMES_TUI_LOG_LEVEL` / `HERMES_TUI_LOG_FILE` | engine defaults | Logging verbosity/destination (`/logs` reads the ring buffer regardless). Deliberately independent of the master switch — support often wants logs without the full diag surface. |

## 4. Internal plumbing (set by the launcher/bench/tests — humans never set these)

| var | set by | effect |
|---|---|---|
| `HERMES_PYTHON`, `HERMES_PYTHON_SRC_ROOT`, `HERMES_CWD` | launcher / bench | Which gateway python + repo root + cwd the TUI spawns against (the bench's fake-gateway seam). |
| `HERMES_TUI_ACTIVE_SESSION_FILE` | launcher/bench | Session handoff file. |
| `HERMES_TUI_RESUME`, `HERMES_TUI_PROMPT`, `HERMES_TUI_FAKE` | launcher/tests | Resume-at-boot, seeded prompt, fake-mode. |
| `HERMES_TUI_RPC_TIMEOUT_MS`, `HERMES_TUI_STARTUP_TIMEOUT_MS` | tests/CI | Protocol timeouts. |
| (`ui-tui` only) `HERMES_TUI_MEMSAMPLE_FD/MS` | bench | Ink fd-3 node sampler. |

## How the pieces compose (the support script)

- Regular user, normal day: zero flags, zero diagnostic commands visible.
- "My TUI feels heavy" support flow: `HERMES_TUI_DIAGNOSTICS=1 hermes` → `/mem`
  for the live numbers, `/heapdump` for a snapshot to attach, window stats
  exposed for `bench/live-attach.sh <pid>` to read.
- Developer profiling: same master switch + the individual knobs
  (`HERMES_TUI_WINDOWING=0` A/B, `WINDOW_IDLE_MS` tuning) as needed.
- Anything in section 4 appearing in a user-facing doc is a bug.

Gating implementation: `logic/env.ts` (`diagnosticsEnabled()`),
`logic/slash.ts` (`DIAGNOSTIC_COMMANDS` — dispatch hint, help + completion
filtering), `view/transcript.tsx` (stats default). Tests:
`slash.test.ts` (gating both states), `utilityCommands.test.ts` (commands
themselves, gate enabled suite-wide).
