import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { resolveManualSessionOrderIds, sessionRecency } from './order'

describe('sessionRecency', () => {
  it('prefers last_active over started_at', () => {
    const session = { last_active: 2_000, started_at: 1_000 } as SessionInfo
    expect(sessionRecency(session)).toBe(2_000)
  })

  it('falls back to started_at when last_active is missing', () => {
    const session = { last_active: 0, started_at: 1_000 } as SessionInfo
    expect(sessionRecency(session)).toBe(1_000)
  })
})

describe('resolveManualSessionOrderIds', () => {
  it('clears legacy auto-seeded order until the user manually reorders sessions', () => {
    expect(resolveManualSessionOrderIds(['newest', 'older'], ['older', 'newest'], false)).toEqual([])
  })

  it('keeps a manual order and surfaces newly seen sessions first', () => {
    expect(resolveManualSessionOrderIds(['newest', 'older', 'oldest'], ['oldest', 'older'], true)).toEqual([
      'newest',
      'oldest',
      'older'
    ])
  })

  it('clears manual order when none of the saved ids still exist', () => {
    expect(resolveManualSessionOrderIds(['newest'], ['gone'], true)).toEqual([])
  })
})
