import { useState, useEffect, useCallback, useRef } from 'react'
import { fetchInteractions } from '@/lib/api'
import type { Interaction } from '@/types'
import { Skeleton } from '@/components/ui/skeleton'
import { placeholderDataURI } from '@/lib/utils'

function fmtDate(iso: string | null): string {
  if (!iso) return ''
  const d = new Date(iso)
  return isNaN(d.getTime())
    ? ''
    : d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

function InteractionCard({ item }: { item: Interaction }) {
  return (
    <a
      href={`/book/${item.book_id}`}
      className="flex gap-3 rounded-lg border border-border bg-card p-3 transition-colors hover:bg-accent/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <img
        src={item.cover_url_small}
        alt=""
        className="h-16 w-11 shrink-0 rounded object-cover"
        loading="lazy"
        onError={(e) => { (e.currentTarget as HTMLImageElement).src = placeholderDataURI(item.title) }}
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between gap-2 mb-0.5">
          <p className="text-sm font-semibold leading-tight line-clamp-1">{item.title}</p>
          <div className="flex items-center gap-2 shrink-0 text-xs text-muted-foreground">
            {item.user_rating != null && (
              <span className="font-medium text-foreground">{item.user_rating}/10</span>
            )}
            {fmtDate(item.rated_at) && <span>{fmtDate(item.rated_at)}</span>}
          </div>
        </div>
        <p className="text-xs text-muted-foreground italic mb-1">{item.author}</p>
        {item.comment && (
          <p className="text-xs text-foreground/80 line-clamp-2">{item.comment}</p>
        )}
      </div>
    </a>
  )
}

const PAGE_SIZE = 10

export function InteractionsList() {
  const [items, setItems] = useState<Interaction[]>([])
  const [totalCount, setTotalCount] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [nextCursor, setNextCursor] = useState<number | undefined>(undefined)
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState(false)
  const sentinelRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const loadInitial = useCallback(async () => {
    setLoading(true)
    setError(false)
    try {
      const data = await fetchInteractions(PAGE_SIZE)
      setItems(data.items)
      setTotalCount(data.total_count)
      setHasMore(data.has_more)
      setNextCursor(data.next_cursor ?? undefined)
    } catch {
      setError(true)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void loadInitial() }, [loadInitial])

  const loadMore = useCallback(async () => {
    if (loadingMore || !hasMore) return
    setLoadingMore(true)
    try {
      const data = await fetchInteractions(PAGE_SIZE, nextCursor)
      setItems((prev) => [...prev, ...data.items])
      setHasMore(data.has_more)
      setNextCursor(data.next_cursor ?? undefined)
    } catch {
      // silently fail
    } finally {
      setLoadingMore(false)
    }
  }, [loadingMore, hasMore, nextCursor])

  // Intersection observer watching the sentinel inside the scroll container
  useEffect(() => {
    const sentinel = sentinelRef.current
    const root = scrollRef.current
    if (!sentinel || !root) return

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) void loadMore()
      },
      { root, threshold: 0.1 },
    )
    observer.observe(sentinel)
    return () => observer.disconnect()
  }, [loadMore])

  return (
    <section>
      <div className="flex items-baseline gap-2 mb-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Your Interactions
        </h2>
        {!loading && (
          <span className="text-xs text-muted-foreground/60">
            {totalCount.toLocaleString()}
          </span>
        )}
      </div>

      {loading && (
        <div className="space-y-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-20 w-full rounded-lg" />
          ))}
        </div>
      )}

      {error && (
        <p className="text-sm text-muted-foreground">Unable to load interactions.</p>
      )}

      {!loading && !error && items.length === 0 && (
        <p className="text-sm text-muted-foreground">You haven't logged any interactions yet.</p>
      )}

      {!loading && !error && items.length > 0 && (
        <div ref={scrollRef} className="max-h-96 overflow-y-auto space-y-2 pr-1">
          {items.map((item, i) => (
            <InteractionCard key={`${item.book_id}-${i}`} item={item} />
          ))}
          <div ref={sentinelRef} className="h-4" />
          {loadingMore && (
            <div className="py-2 text-center text-xs text-muted-foreground">Loading…</div>
          )}
        </div>
      )}
    </section>
  )
}
