import { useState, useEffect, useCallback, useRef } from 'react'
import { fetchComments } from '@/lib/api'
import type { Comment } from '@/types'
import { Skeleton } from '@/components/ui/skeleton'

interface CommentsSectionProps {
  itemIdx: number
}

function avatarColor(username: string): string {
  const colors = [
    'bg-blue-100 text-blue-700',
    'bg-green-100 text-green-700',
    'bg-purple-100 text-purple-700',
    'bg-orange-100 text-orange-700',
    'bg-pink-100 text-pink-700',
    'bg-teal-100 text-teal-700',
  ]
  return colors[username.charCodeAt(0) % colors.length]
}

function initials(username: string): string {
  return username.slice(0, 2).toUpperCase()
}

function fmtDate(iso: string | null): string {
  if (!iso) return ''
  const d = new Date(iso)
  return isNaN(d.getTime())
    ? ''
    : d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

function CommentCard({ comment }: { comment: Comment }) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className={`flex h-8 w-8 items-center justify-center rounded-full text-xs font-semibold shrink-0 ${avatarColor(comment.username)}`}>
            {initials(comment.username)}
          </div>
          <span className="text-sm font-medium">{comment.username}</span>
        </div>
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          {comment.rating != null && (
            <span className="font-medium text-foreground">{comment.rating}/10</span>
          )}
          {fmtDate(comment.rated_at) && <span>{fmtDate(comment.rated_at)}</span>}
        </div>
      </div>
      {comment.comment && (
        <p className="text-sm text-foreground/80 leading-relaxed">{comment.comment}</p>
      )}
    </div>
  )
}

const PAGE_SIZE = 10

export function CommentsSection({ itemIdx }: CommentsSectionProps) {
  const [comments, setComments] = useState<Comment[]>([])
  const [totalCount, setTotalCount] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [cursor, setCursor] = useState<number | undefined>(undefined)
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState(false)
  const sentinelRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const loadInitial = useCallback(async () => {
    setLoading(true)
    setError(false)
    try {
      const data = await fetchComments(itemIdx, PAGE_SIZE)
      setComments(data.items)
      setTotalCount(data.total_count)
      setHasMore(data.has_more)
      if (data.items.length > 0) {
        setCursor(data.items[data.items.length - 1].user_id)
      }
    } catch {
      setError(true)
    } finally {
      setLoading(false)
    }
  }, [itemIdx])

  useEffect(() => { void loadInitial() }, [loadInitial])

  const loadMore = useCallback(async () => {
    if (loadingMore || !hasMore) return
    setLoadingMore(true)
    try {
      const data = await fetchComments(itemIdx, PAGE_SIZE, cursor)
      setComments((prev) => [...prev, ...data.items])
      setHasMore(data.has_more)
      if (data.items.length > 0) {
        setCursor(data.items[data.items.length - 1].user_id)
      }
    } catch {
      // silently fail
    } finally {
      setLoadingMore(false)
    }
  }, [itemIdx, loadingMore, hasMore, cursor])

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
          Comments
        </h2>
        {!loading && (
          <span className="text-xs text-muted-foreground/60">
            {totalCount.toLocaleString()}
          </span>
        )}
      </div>

      {loading && (
        <div className="space-y-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-20 w-full rounded-lg" />
          ))}
        </div>
      )}

      {error && (
        <p className="text-sm text-muted-foreground">Unable to load comments.</p>
      )}

      {!loading && !error && comments.length === 0 && (
        <p className="text-sm text-muted-foreground">No comments yet.</p>
      )}

      {!loading && !error && comments.length > 0 && (
        <div ref={scrollRef} className="max-h-72 overflow-y-auto space-y-3 pr-1">
          {comments.map((c, i) => (
            <CommentCard key={`${c.user_id}-${i}`} comment={c} />
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
