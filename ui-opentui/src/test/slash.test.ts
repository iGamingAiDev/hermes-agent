/**
 * Slash dispatch test (spec §5 Layer 3/4). Pure logic: parse + the dispatch
 * ladder (client → slash.exec → command.dispatch) against a fake SlashContext.
 */
import { afterEach, describe, expect, test } from 'vitest'

import type { DetailsMode } from '../logic/details.ts'
import {
  buildModelTabs,
  clientCommandNames,
  dispatchSlash,
  mapCompletions,
  parseSlash,
  pickerTabs,
  planCompletion,
  readReplaceFrom,
  registerModelPrefetch,
  registerPickerRefresh,
  registerPickerTabs,
  runPickerRefresh,
  type SlashContext
} from '../logic/slash.ts'
import type { SessionTabId } from '../logic/sessionPicker.ts'
import type { PickerItem } from '../logic/store.ts'

// the picker-refresh/tabs/prefetch seams are module-level state — never leak them across tests
afterEach(() => {
  registerPickerRefresh(undefined)
  registerPickerTabs(undefined)
  registerModelPrefetch(undefined)
})

/** A `model.options` payload: two authed providers + two unconfigured skeleton
 *  rows (the gateway sends them with `include_unconfigured=True,
 *  picker_hints=True`: empty models + `key_env`/`warning` setup hints). */
const MODEL_OPTIONS = {
  model: 'claude-sonnet-4.6',
  provider: 'anthropic',
  providers: [
    {
      authenticated: true,
      models: ['claude-sonnet-4.6', 'claude-opus-4.6'],
      name: 'Anthropic',
      slug: 'anthropic'
    },
    {
      authenticated: false,
      key_env: 'OPENAI_API_KEY',
      models: [],
      name: 'OpenAI API',
      slug: 'openai-api',
      warning: 'paste OPENAI_API_KEY to activate'
    },
    { authenticated: true, models: ['hermes-4-405b'], name: 'Nous Research', slug: 'nous' },
    {
      authenticated: false,
      key_env: '',
      models: [],
      name: 'OpenAI Codex',
      slug: 'openai-codex',
      warning: 'run `hermes model` to configure (oauth_external)'
    }
  ]
}

describe('mapCompletions', () => {
  test('maps complete.slash items → candidates (display/meta default)', () => {
    expect(
      mapCompletions({ items: [{ display: '/compact', meta: 'compress', text: '/compact' }, { text: '/details' }] })
    ).toEqual([
      { display: '/compact', meta: 'compress', text: '/compact' },
      { display: '/details', meta: '', text: '/details' }
    ])
    expect(mapCompletions({ items: [] })).toEqual([])
    expect(mapCompletions(null)).toEqual([])
  })
})

describe('planCompletion (items 5 + 13)', () => {
  test('a slash line → complete.slash with the full text (name AND args)', () => {
    expect(planCompletion('/mod')).toEqual({ from: 0, method: 'complete.slash', params: { text: '/mod' } })
    // args too — the gateway completes e.g. /details section names
    expect(planCompletion('/details thi')).toEqual({
      from: 0,
      method: 'complete.slash',
      params: { text: '/details thi' }
    })
  })

  test('a trailing path-like word → complete.path with that word + token start offset', () => {
    expect(planCompletion('explain @src/fo')).toEqual({
      from: 'explain '.length,
      method: 'complete.path',
      params: { word: '@src/fo' }
    })
    expect(planCompletion('cat ./rea')).toEqual({
      from: 'cat '.length,
      method: 'complete.path',
      params: { word: './rea' }
    })
    expect(planCompletion('open ~/proj')).toEqual({
      from: 'open '.length,
      method: 'complete.path',
      params: { word: '~/proj' }
    })
  })

  test('plain prose / multiline → no completion', () => {
    expect(planCompletion('just some words')).toBeNull()
    expect(planCompletion('hello')).toBeNull()
    expect(planCompletion('/cmd with\nnewline')).toBeNull()
  })
})

describe('readReplaceFrom', () => {
  test('reads gateway replace_from, falls back when absent/non-number', () => {
    expect(readReplaceFrom({ items: [], replace_from: 9 }, 0)).toBe(9)
    expect(readReplaceFrom({ items: [] }, 4)).toBe(4)
    expect(readReplaceFrom({ replace_from: 'nope' }, 7)).toBe(7)
    expect(readReplaceFrom(null, 2)).toBe(2)
  })
})

describe('parseSlash', () => {
  test('splits name + arg; rejects non-slash / empty', () => {
    expect(parseSlash('/help')).toEqual({ name: 'help', arg: '' })
    expect(parseSlash('/model anthropic/claude')).toEqual({ name: 'model', arg: 'anthropic/claude' })
    expect(parseSlash('hello')).toBeNull()
    expect(parseSlash('/')).toBeNull()
  })
})

interface Probe {
  ctx: SlashContext
  calls: Array<{ method: string; params: Record<string, unknown> }>
  system: string[]
  submitted: string[]
  confirmed: Array<{ message: string; onConfirm: () => void }>
  paged: Array<{ title: string; text: string }>
  sessionPickers: SessionTabId[]
  resumed: string[]
  pickers: Array<{ title: string; items: PickerItem[]; onPick: (value: string) => void }>
  quit: { value: boolean }
  cleared: { value: boolean }
  dashboard: { value: boolean }
  copied: number[]
  copyN: { value: (n: number) => boolean }
  /** The cached /model rows (Epic 7) — seed to simulate a prefetched catalog. */
  modelCache: { value: PickerItem[] | undefined }
  /** Display flags (/compact, /details — Epic 3). */
  compactFlag: { value: boolean }
  detailsFlag: { value: DetailsMode }
}

function makeCtx(request: (method: string, params: Record<string, unknown>) => Promise<unknown>): Probe {
  const calls: Probe['calls'] = []
  const system: string[] = []
  const submitted: string[] = []
  const confirmed: Probe['confirmed'] = []
  const paged: Probe['paged'] = []
  const sessionPickers: SessionTabId[] = []
  const resumed: string[] = []
  const pickers: Probe['pickers'] = []
  const quit = { value: false }
  const cleared = { value: false }
  const dashboard = { value: false }
  const copied: number[] = []
  const copyN: Probe['copyN'] = { value: () => false }
  const modelCache: Probe['modelCache'] = { value: undefined }
  const compactFlag: Probe['compactFlag'] = { value: false }
  const detailsFlag: Probe['detailsFlag'] = { value: 'collapsed' }
  const ctx: SlashContext = {
    clearTranscript: () => (cleared.value = true),
    compact: () => compactFlag.value,
    setCompact: on => (compactFlag.value = on),
    details: () => detailsFlag.value,
    setDetails: mode => (detailsFlag.value = mode),
    renderableCount: () => undefined,
    confirm: (message, onConfirm) => confirmed.push({ message, onConfirm }),
    copyResponse: n => {
      copied.push(n)
      return copyN.value(n)
    },
    logTail: () => ['gateway: spawned', 'bootstrap: session created'],
    modelItems: () => modelCache.value,
    setModelItems: items => (modelCache.value = items),
    openDashboard: () => (dashboard.value = true),
    openPager: (title, text) => paged.push({ text, title }),
    openPicker: p => pickers.push(p),
    openSessionPicker: tab => sessionPickers.push(tab),
    resumeSession: id => resumed.push(id),
    pushSystem: text => system.push(text),
    quit: () => (quit.value = true),
    request: (method, params) => {
      calls.push({ method, params })
      return request(method, params)
    },
    sessionId: () => 'sid-1',
    submit: text => submitted.push(text)
  }
  return {
    calls,
    cleared,
    compactFlag,
    confirmed,
    copied,
    copyN,
    ctx,
    dashboard,
    detailsFlag,
    modelCache,
    paged,
    pickers,
    quit,
    resumed,
    sessionPickers,
    submitted,
    system
  }
}

describe('dispatchSlash — client commands', () => {
  test('/quit quits without hitting the gateway', async () => {
    const p = makeCtx(async () => ({}))
    await dispatchSlash('/quit', p.ctx)
    expect(p.quit.value).toBe(true)
    expect(p.calls).toHaveLength(0)
  })

  test('/clear opens a confirm; running onConfirm clears the transcript', async () => {
    const p = makeCtx(async () => ({}))
    await dispatchSlash('/clear', p.ctx)
    expect(p.confirmed).toHaveLength(1)
    expect(p.cleared.value).toBe(false)
    p.confirmed[0]!.onConfirm()
    expect(p.cleared.value).toBe(true)
  })

  test('/logs opens the pager with the recent ring lines', async () => {
    const p = makeCtx(async () => ({}))
    await dispatchSlash('/logs', p.ctx)
    expect(p.paged[0]?.title).toBe('Logs')
    expect(p.paged[0]?.text).toContain('session created')
  })

  test('/sessions (and bare /resume) open the resume picker on the Recent tab', async () => {
    const p = makeCtx(async () => ({}))
    await dispatchSlash('/sessions', p.ctx)
    expect(p.sessionPickers).toEqual(['recent'])
    expect(p.calls).toHaveLength(0) // the overlay fetches its own rows
    const p2 = makeCtx(async () => ({}))
    await dispatchSlash('/resume', p2.ctx)
    expect(p2.sessionPickers).toEqual(['recent'])
  })

  test('/sessions cron|gateways pre-select that tab; garbage -> usage', async () => {
    const p = makeCtx(async () => ({}))
    await dispatchSlash('/sessions cron', p.ctx)
    await dispatchSlash('/sessions gateways', p.ctx)
    await dispatchSlash('/sessions all', p.ctx)
    expect(p.sessionPickers).toEqual(['cron', 'gateways', 'all'])
    await dispatchSlash('/sessions bogus', p.ctx)
    expect(p.sessionPickers).toHaveLength(3)
    expect(p.system.at(-1)).toContain('usage: /sessions')
  })

  test('/resume <id|name> keeps the DIRECT path: resolves against session.list and resumes', async () => {
    const rows = {
      sessions: [
        { id: 'abc-123', message_count: 5, preview: 'hello', source: 'tui', started_at: 1, title: 'First chat' },
        { id: 'def-456', message_count: 2, preview: 'yo', source: 'cli', started_at: 2, title: 'Goal v4' }
      ],
      truncated: false
    }
    const p = makeCtx(async method => (method === 'session.list' ? rows : {}))
    await dispatchSlash('/resume abc-123', p.ctx) // exact id
    await dispatchSlash('/resume def', p.ctx) // unique id prefix
    await dispatchSlash('/resume goal v4', p.ctx) // exact title (ci)
    expect(p.resumed).toEqual(['abc-123', 'def-456', 'def-456'])
    expect(p.sessionPickers).toHaveLength(0) // never opened the overlay
    await dispatchSlash('/resume nope', p.ctx)
    expect(p.resumed).toHaveLength(3)
    expect(p.system.at(-1)).toContain('no session matching')
  })

  test('/model (bare) opens a GROUPED picker of authenticated providers’ models; pick switches', async () => {
    const p = makeCtx(async method => {
      if (method === 'model.options') return MODEL_OPTIONS
      return { output: 'switched' }
    })
    await dispatchSlash('/model', p.ctx)
    expect(p.pickers).toHaveLength(1)
    expect(p.pickers[0]!.title).toBe('Switch model')
    // authenticated providers' models are the SELECTABLE rows; values carry the
    // explicit provider so a pick under a different provider switches both.
    const selectable = p.pickers[0]!.items.filter(i => !i.unavailable)
    expect(selectable.map(i => i.value)).toEqual([
      'claude-sonnet-4.6 --provider anthropic',
      'claude-opus-4.6 --provider anthropic',
      'hermes-4-405b --provider nous'
    ])
    // grouped by the provider's display (lab) name; slug+lab are fuzzy haystacks
    expect(selectable.map(i => i.group)).toEqual(['Anthropic', 'Anthropic', 'Nous Research'])
    expect(selectable[2]!.haystacks).toEqual(['nous', 'Nous Research'])
    // current is FLAGGED (not baked into the label, so fuzzy never matches the ✓)
    expect(selectable[0]!.current).toBe(true)
    expect(selectable[0]!.label).toBe('claude-sonnet-4.6')
    expect(selectable[1]!.current).toBeUndefined()
    // picking switches via slash.exec `model <model> --provider <slug>`
    p.pickers[0]!.onPick('claude-opus-4.6 --provider anthropic')
    await new Promise(r => setTimeout(r, 0))
    expect(
      p.calls.some(c => c.method === 'slash.exec' && c.params.command === 'model claude-opus-4.6 --provider anthropic')
    ).toBe(true)
  })

  test('/model maps UNCONFIGURED providers to dimmed hint rows (key_env → env-var hint, else warning)', async () => {
    const p = makeCtx(async method => (method === 'model.options' ? MODEL_OPTIONS : { output: 'switched' }))
    await dispatchSlash('/model', p.ctx)
    const unavailable = p.pickers[0]!.items.filter(i => i.unavailable)
    expect(unavailable).toHaveLength(2)
    // api_key provider → the `no API key — set <ENV_VAR>` hint as the row label
    expect(unavailable[0]).toEqual({
      group: 'OpenAI API',
      haystacks: ['openai-api', 'OpenAI API'],
      label: 'no API key — set OPENAI_API_KEY',
      unavailable: true,
      value: 'openai-api'
    })
    // oauth provider (no key_env) → the gateway's own warning text
    expect(unavailable[1]!.group).toBe('OpenAI Codex')
    expect(unavailable[1]!.label).toBe('run `hermes model` to configure (oauth_external)')
    // payload (canonical) order is preserved — unconfigured rows interleave
    expect(p.pickers[0]!.items.map(i => i.group)).toEqual([
      'Anthropic',
      'Anthropic',
      'OpenAI API',
      'Nous Research',
      'OpenAI Codex'
    ])
  })

  test('/model with ONLY unconfigured providers keeps the no-models notice', async () => {
    const p = makeCtx(async () => ({
      providers: [{ authenticated: false, key_env: 'XAI_API_KEY', models: [], name: 'xAI', slug: 'xai' }]
    }))
    await dispatchSlash('/model', p.ctx)
    expect(p.pickers).toHaveLength(0)
    expect(p.system).toEqual(['No models available (no authenticated providers).'])
  })

  test('/model registers the picker refresh seam; running it does ONE RPC and re-syncs the cache', async () => {
    const p = makeCtx(async method => (method === 'model.options' ? MODEL_OPTIONS : { output: 'switched' }))
    await dispatchSlash('/model', p.ctx)
    const opened = p.calls.filter(c => c.method === 'model.options').length // 1 (uncached open)
    const refreshed = await runPickerRefresh()
    expect(p.calls.filter(c => c.method === 'model.options')).toHaveLength(opened + 1)
    expect(refreshed!.filter(i => !i.unavailable)).toHaveLength(3)
    expect(p.modelCache.value).toEqual(refreshed) // cache re-synced for the next open
  })

  test('/skills clears the picker refresh seam (Ctrl+R is a no-op there)', async () => {
    registerPickerRefresh(() => Promise.resolve([]))
    const p = makeCtx(async () => ({ skills: { General: ['memory'] } }))
    await dispatchSlash('/skills', p.ctx)
    expect(runPickerRefresh()).toBeUndefined()
  })

  test('/model with a CACHED catalog opens instantly — ZERO RPCs on open', async () => {
    const p = makeCtx(async () => {
      throw new Error('no RPC expected on open')
    })
    p.modelCache.value = [
      {
        group: 'Anthropic',
        haystacks: ['anthropic', 'Anthropic'],
        label: 'claude-sonnet-4.6',
        value: 'claude-sonnet-4.6 --provider anthropic'
      },
      {
        group: 'Nous Research',
        haystacks: ['nous', 'Nous Research'],
        label: 'hermes-4-405b',
        value: 'hermes-4-405b --provider nous'
      }
    ]
    await dispatchSlash('/model', p.ctx)
    expect(p.pickers).toHaveLength(1)
    expect(p.pickers[0]!.items).toHaveLength(2)
    expect(p.calls).toHaveLength(0) // the whole point: open = memory, not network
  })

  test('/model uncached fetches ONCE, caches, and a pick refreshes the cache', async () => {
    const p = makeCtx(async method => (method === 'model.options' ? MODEL_OPTIONS : { output: 'switched' }))
    await dispatchSlash('/model', p.ctx)
    expect(p.calls.filter(c => c.method === 'model.options')).toHaveLength(1)
    expect(p.modelCache.value).toHaveLength(5) // first open seeded the cache (3 models + 2 unconfigured hints)
    // cross-provider pick: switch lands on the gateway, then a background
    // refresh re-fetches model.options so the cached ✓ stays fresh.
    p.pickers[0]!.onPick('hermes-4-405b --provider nous')
    await new Promise(r => setTimeout(r, 0))
    expect(
      p.calls.some(c => c.method === 'slash.exec' && c.params.command === 'model hermes-4-405b --provider nous')
    ).toBe(true)
    expect(p.calls.filter(c => c.method === 'model.options')).toHaveLength(2)
  })

  test('buildModelTabs: Nous-identified groups first, then catalog order; unconfigured providers get NO tab', () => {
    const items: PickerItem[] = [
      { group: 'Anthropic', haystacks: ['anthropic', 'Anthropic'], label: 'claude-sonnet-4.6', value: 'a' },
      { group: 'Anthropic', haystacks: ['anthropic', 'Anthropic'], label: 'claude-opus-4.6', value: 'b' },
      {
        group: 'OpenAI API',
        haystacks: ['openai-api', 'OpenAI API'],
        label: 'no API key',
        unavailable: true,
        value: 'openai-api'
      },
      { group: 'GitHub Copilot', haystacks: ['copilot', 'GitHub Copilot'], label: 'gpt-5', value: 'c' },
      // Nous identified via the SLUG haystack even when the display name hides it
      { group: 'Portal', haystacks: ['nous-portal', 'Portal'], label: 'hermes-4-405b', value: 'd' }
    ]
    expect(buildModelTabs(items)).toEqual(['Portal', 'Anthropic', 'GitHub Copilot'])
    expect(buildModelTabs([])).toEqual([])
  })

  test('/model registers the provider-tab seam (buildModelTabs); /skills clears it back to stripless', async () => {
    const p = makeCtx(async method => (method === 'model.options' ? MODEL_OPTIONS : { output: 'switched' }))
    await dispatchSlash('/model', p.ctx)
    // the open picker derives Nous-first tabs through the seam
    expect(pickerTabs(p.pickers[0]!.items)).toEqual(['Nous Research', 'Anthropic'])
    const p2 = makeCtx(async () => ({ skills: { General: ['memory'] } }))
    await dispatchSlash('/skills', p2.ctx)
    expect(pickerTabs(p.pickers[0]!.items)).toEqual([])
  })

  test('/model during an in-flight bootstrap prefetch performs ZERO additional model.options RPCs', async () => {
    const p = makeCtx(async () => {
      throw new Error('no RPC expected — the prefetch owns model.options')
    })
    // a slow prefetch (entry/main.tsx shape): resolving it fills the cache
    let finish: () => void = () => {}
    registerModelPrefetch(
      new Promise<void>(resolve => {
        finish = () => {
          p.modelCache.value = [
            { group: 'Anthropic', haystacks: ['anthropic'], label: 'claude-sonnet-4.6', value: 'a' }
          ]
          resolve()
        }
      }),
      5000
    )
    const dispatched = dispatchSlash('/model', p.ctx)
    finish() // prefetch lands while /model awaits it
    await dispatched
    expect(p.pickers).toHaveLength(1) // opened from the prefetched cache
    expect(p.pickers[0]!.items).toHaveLength(1)
    expect(p.calls).toHaveLength(0) // the dedupe: no second model.options
  })

  test('/model with a HUNG prefetch still opens via its own fetch after the bound', async () => {
    const p = makeCtx(async method => (method === 'model.options' ? MODEL_OPTIONS : { output: 'switched' }))
    registerModelPrefetch(new Promise(() => {}), 10) // never settles; tiny test bound
    await dispatchSlash('/model', p.ctx)
    expect(p.pickers).toHaveLength(1) // fell back to fetching itself
    expect(p.calls.filter(c => c.method === 'model.options')).toHaveLength(1)
  })

  test('/model <name> switches directly without opening the picker', async () => {
    const p = makeCtx(async () => ({ output: 'ok' }))
    await dispatchSlash('/model anthropic/claude-opus-4.6', p.ctx)
    expect(p.pickers).toHaveLength(0)
    expect(p.calls[0]).toEqual({
      method: 'slash.exec',
      params: { command: 'model anthropic/claude-opus-4.6', session_id: 'sid-1' }
    })
  })

  test('/copy copies via copyResponse; no system line on success', async () => {
    const p = makeCtx(async () => ({}))
    p.copyN.value = () => true
    await dispatchSlash('/copy', p.ctx)
    expect(p.copied).toEqual([1])
    expect(p.system).toHaveLength(0)
  })

  test('/copy 2 passes the n-th index through', async () => {
    const p = makeCtx(async () => ({}))
    p.copyN.value = () => true
    await dispatchSlash('/copy 2', p.ctx)
    expect(p.copied).toEqual([2])
  })

  test('/copy when nothing to copy pushes a system notice', async () => {
    const p = makeCtx(async () => ({}))
    p.copyN.value = () => false
    await dispatchSlash('/copy', p.ctx)
    expect(p.system).toContain('Nothing to copy yet.')
  })

  test('/agents (and /tasks) open the agents dashboard', async () => {
    const p = makeCtx(async () => ({}))
    await dispatchSlash('/agents', p.ctx)
    expect(p.dashboard.value).toBe(true)
    const p2 = makeCtx(async () => ({}))
    await dispatchSlash('/tasks', p2.ctx)
    expect(p2.dashboard.value).toBe(true)
  })

  test('/skills opens a picker flattened from skills.manage list', async () => {
    const p = makeCtx(async method =>
      method === 'skills.manage' ? { skills: { media: ['ffmpeg', 'whisper'], web: ['firecrawl'] } } : {}
    )
    await dispatchSlash('/skills', p.ctx)
    expect(p.pickers).toHaveLength(1)
    expect(p.pickers[0]!.title).toBe('Skills')
    expect(p.pickers[0]!.items.map(i => i.value).sort()).toEqual(['ffmpeg', 'firecrawl', 'whisper'])
  })

  test('/help renders the gateway catalog', async () => {
    const p = makeCtx(async method =>
      method === 'commands.catalog' ? { pairs: [['/model', 'switch model']], canon: {} } : {}
    )
    await dispatchSlash('/help', p.ctx)
    expect(p.calls[0]?.method).toBe('commands.catalog')
    expect(p.system.join('\n')).toContain('/model — switch model')
  })
})

describe('dispatchSlash — server ladder', () => {
  test('unknown command → slash.exec; SHORT output shown as a system line', async () => {
    const p = makeCtx(async method => (method === 'slash.exec' ? { output: 'all good' } : {}))
    await dispatchSlash('/status', p.ctx)
    expect(p.calls[0]).toEqual({ method: 'slash.exec', params: { command: 'status', session_id: 'sid-1' } })
    expect(p.system).toContain('all good')
    expect(p.paged).toHaveLength(0)
  })

  test('LONG slash.exec output opens the pager (titled by command)', async () => {
    const longText = Array.from({ length: 6 }, (_, i) => `output line ${i}`).join('\n')
    const p = makeCtx(async method => (method === 'slash.exec' ? { output: longText } : {}))
    await dispatchSlash('/status', p.ctx)
    expect(p.paged).toHaveLength(1)
    expect(p.paged[0]?.title).toBe('Status')
    expect(p.paged[0]?.text).toContain('output line 5')
    expect(p.system).toHaveLength(0)
  })

  test('slash.exec rejects → command.dispatch; send result submits a user turn', async () => {
    const p = makeCtx(async method => {
      if (method === 'slash.exec') throw new Error('not a worker command')
      if (method === 'command.dispatch') return { type: 'send', message: 'run the thing' }
      return {}
    })
    await dispatchSlash('/dothing', p.ctx)
    expect(p.calls.map(c => c.method)).toEqual(['slash.exec', 'command.dispatch'])
    expect(p.submitted).toEqual(['run the thing'])
  })

  test('command.dispatch exec → system output', async () => {
    const p = makeCtx(async method => {
      if (method === 'slash.exec') throw new Error('reject')
      return { type: 'exec', output: 'done' }
    })
    await dispatchSlash('/whatever', p.ctx)
    expect(p.system).toContain('done')
  })
})

describe('diagnostic command gating (HERMES_TUI_DIAGNOSTICS)', () => {
  const KEY = 'HERMES_TUI_DIAGNOSTICS'
  const prev = process.env[KEY]
  afterEach(() => {
    if (prev === undefined) delete process.env[KEY]
    else process.env[KEY] = prev
  })

  test('OFF (default): /mem and /heapdump respond with the enable hint, not the command', async () => {
    delete process.env[KEY]
    const p = makeCtx(async () => ({}))
    await dispatchSlash('/mem', p.ctx)
    await dispatchSlash('/heapdump', p.ctx)
    expect(p.system[0]).toContain('HERMES_TUI_DIAGNOSTICS=1')
    expect(p.system[1]).toContain('HERMES_TUI_DIAGNOSTICS=1')
    expect(p.calls).toHaveLength(0) // never reached the gateway ladder either
  })

  test('OFF: the diagnostic names are absent from clientCommandNames()', () => {
    delete process.env[KEY]
    const names = clientCommandNames()
    expect(names).not.toContain('mem')
    expect(names).not.toContain('heapdump')
    expect(names).toContain('logs') // non-diagnostic neighbors stay
  })

  test('ON: /mem executes (live memory stats), names are listed', async () => {
    process.env[KEY] = '1'
    expect(clientCommandNames()).toContain('mem')
    expect(clientCommandNames()).toContain('heapdump')
    const p = makeCtx(async () => ({}))
    await dispatchSlash('/mem', p.ctx)
    const out = [...p.system, ...p.paged.map(x => x.text)].join('\n')
    expect(out).toMatch(/rss|heap/i)
  })
})
