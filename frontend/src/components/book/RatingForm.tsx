import { useState } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { postRating, deleteRating } from '@/lib/api'
import type { UserRating } from '@/types'

interface RatingFormProps {
  itemIdx: number
  loggedIn: boolean
  initialRating: UserRating | null
}

type Mode = 'read' | 'rated'

export function RatingForm({ itemIdx, loggedIn, initialRating }: RatingFormProps) {
  const [mode, setMode] = useState<Mode>(
    initialRating?.rating != null ? 'rated' : 'read',
  )
  const [rating, setRating] = useState<string>(
    initialRating?.rating != null ? String(initialRating.rating) : '',
  )
  const [comment, setComment] = useState(initialRating?.comment ?? '')
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [hasRating, setHasRating] = useState(initialRating !== null)

  if (!loggedIn) {
    return (
      <section>
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground mb-3">
          Your Rating
        </h2>
        <p className="text-sm text-muted-foreground">
          <a href="/login" className="text-primary hover:underline">Log in</a> to rate this book.
        </p>
      </section>
    )
  }

  async function handleSave() {
    setSaving(true)
    try {
      const ratingVal = mode === 'rated' && rating ? parseInt(rating, 10) : null
      await postRating(itemIdx, ratingVal, comment)
      setHasRating(true)
      toast.success('Rating saved')
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to save rating')
    } finally {
      setSaving(false)
    }
  }

  async function handleDelete() {
    if (!confirm('Delete your rating for this book?')) return
    setDeleting(true)
    try {
      await deleteRating(itemIdx)
      setHasRating(false)
      setRating('')
      setComment('')
      setMode('read')
      toast.success('Rating deleted')
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to delete rating')
    } finally {
      setDeleting(false)
    }
  }

  return (
    <section>
      <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground mb-3">
        {hasRating ? 'Edit Your Rating' : 'Rate or Mark as Read'}
      </h2>

      {/* Mode toggle */}
      <div className="inline-flex rounded-lg border border-border overflow-hidden mb-4">
        {(['read', 'rated'] as Mode[]).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            className={`px-4 py-2 text-sm font-medium transition-colors cursor-pointer ${
              mode === m
                ? 'bg-primary text-primary-foreground'
                : 'bg-background text-muted-foreground hover:bg-accent'
            }`}
          >
            {m === 'read' ? "I've read this" : 'Rate it'}
          </button>
        ))}
      </div>

      <div className="space-y-3 max-w-md">
        {mode === 'rated' && (
          <div>
            <label className="block text-sm font-medium mb-1" htmlFor="rating-input">
              Rating (1–10)
            </label>
            <input
              id="rating-input"
              type="number"
              min={1}
              max={10}
              step={1}
              value={rating}
              onChange={(e) => setRating(e.target.value)}
              className="h-9 w-24 rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
        )}

        <div>
          <label className="block text-sm font-medium mb-1" htmlFor="comment-input">
            Comment <span className="text-muted-foreground font-normal">(optional)</span>
          </label>
          <textarea
            id="comment-input"
            rows={3}
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring resize-none"
          />
        </div>

        <div className="flex gap-2">
          <Button onClick={handleSave} disabled={saving} size="sm">
            {saving ? 'Saving…' : 'Save'}
          </Button>
          {hasRating && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleDelete}
              disabled={deleting}
              className="text-destructive border-destructive/40 hover:bg-destructive/10 hover:text-destructive"
            >
              {deleting ? 'Deleting…' : 'Delete'}
            </Button>
          )}
        </div>
      </div>
    </section>
  )
}
