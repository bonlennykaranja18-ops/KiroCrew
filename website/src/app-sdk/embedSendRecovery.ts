/**
 * Slot-keyed hand-back buffer for ChatEmbed send receipts.
 *
 * ChatEmbed's recovery state (the failure / unconfirmed row and the text it
 * hands back to the composer) is component-local: the draft lives in
 * `useComposerDraft`, the tail row in a `useState`. A send is asynchronous, and
 * a host app can unmount the embed while one is in flight -- switching specs,
 * opening a new one, closing a drawer. When the receipt then lands, writing it
 * into the dead instance loses the only copy of a REFUSED send outright (the
 * server never took it, so no poll will ever show it) and drops the
 * "unconfirmed" warning for a late one.
 *
 * So a receipt that finds its embed gone (or re-propped to another slot) is
 * parked HERE, keyed by the slot the send belonged to, and the next ChatEmbed
 * mounted for that slot drains it -- the same shape SideChat uses through the
 * store (`sideHandBackText` / `SideState.sendStatus`), kept module-local
 * because the app-sdk embed deliberately owns no store slice of its own.
 *
 * One record per slot: a slot has at most one send in flight (the composer
 * guards on `isPending`), and a newer receipt for the same slot is the one the
 * user needs to see. If an embed for the slot is already mounted when the
 * receipt lands (the user navigated away and back within the deadline), the
 * subscriber is notified and drains immediately, so nothing waits for a
 * remount that already happened.
 */

export interface PendingEmbedRecovery {
  /** The transcript-tail row to show: a failure, or an unconfirmed notice. */
  tail: {
    role: 'error' | 'notice'
    content: string
    seenCount: number
    sendId: string
  }
  /** The composer text to hand back, as typed. Absent for a chip send, which
   *  never consumed the draft. */
  restoreText?: string
}

const pending = new Map<string, PendingEmbedRecovery>()
const listeners = new Map<string, Set<() => void>>()

/** Park a receipt for a slot whose embed is not mounted (or not showing that
 *  slot). Notifies a mounted subscriber for the slot, if any. */
export function stashEmbedRecovery(slot: string, rec: PendingEmbedRecovery): void {
  pending.set(slot, rec)
  const subs = listeners.get(slot)
  if (subs) for (const cb of subs) cb()
}

/** Take (and clear) the parked receipt for a slot. */
export function drainEmbedRecovery(slot: string): PendingEmbedRecovery | undefined {
  const rec = pending.get(slot)
  if (rec) pending.delete(slot)
  return rec
}

/** Be told when a receipt is parked for `slot` while subscribed. Returns the
 *  unsubscribe. */
export function subscribeEmbedRecovery(slot: string, cb: () => void): () => void {
  let subs = listeners.get(slot)
  if (!subs) { subs = new Set(); listeners.set(slot, subs) }
  subs.add(cb)
  return () => {
    subs.delete(cb)
    if (subs.size === 0) listeners.delete(slot)
  }
}

/** Test seam: forget every parked receipt and subscriber. */
export function resetEmbedRecoveryForTests(): void {
  pending.clear()
  listeners.clear()
}
