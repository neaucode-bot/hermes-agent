import type { SessionInfo } from '@/types/hermes'

/** Recency key for sidebar ordering — matches backend `order=recent` (last_active). */
export const sessionRecency = (s: SessionInfo) => s.last_active || s.started_at || 0

export function resolveManualSessionOrderIds(currentIds: string[], orderIds: string[], manual: boolean): string[] {
  if (!manual || !currentIds.length || !orderIds.length) {
    return []
  }

  const current = new Set(currentIds)
  const retained = orderIds.filter(id => current.has(id))

  if (!retained.length) {
    return []
  }

  const retainedSet = new Set(retained)
  const fresh = currentIds.filter(id => !retainedSet.has(id))

  return [...fresh, ...retained]
}
