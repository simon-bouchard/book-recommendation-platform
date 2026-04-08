import { useState } from 'react'
import { createPortal } from 'react-dom'
import type { ChatBook } from '@/types'
import { coverUrl, placeholderDataURI } from '@/lib/utils'

interface BookHoverCardProps {
  book: ChatBook
  top: number
  left: number
}

const CARD_WIDTH = 220

export function BookHoverCard({ book, top, left }: BookHoverCardProps) {
  const [imgSrc, setImgSrc] = useState(
    book.cover_id ? coverUrl(book.cover_id, 'M') : placeholderDataURI(book.title),
  )

  return createPortal(
    <div
      style={{ width: CARD_WIDTH, position: 'fixed', zIndex: 9999, top, left, pointerEvents: 'none' }}
      className="rounded-lg border border-border bg-card shadow-lg p-3 text-sm"
    >
      <div className="flex gap-2">
        <img
          key={book.cover_id ?? book.item_idx}
          src={imgSrc}
          alt=""
          className="h-20 w-14 shrink-0 rounded object-cover"
          onError={() => setImgSrc(placeholderDataURI(book.title))}
        />
        <div className="min-w-0">
          <p className="font-semibold leading-tight line-clamp-3 mb-1">{book.title}</p>
          {book.author && (
            <p className="text-xs text-muted-foreground">{book.author}</p>
          )}
          {book.year && (
            <p className="text-xs text-muted-foreground">{book.year}</p>
          )}
          {book.num_ratings != null && (
            <p className="text-xs text-muted-foreground">{book.num_ratings.toLocaleString()} ratings</p>
          )}
          {book.genre && (
            <p className="text-xs text-muted-foreground">{book.genre}</p>
          )}
        </div>
      </div>
    </div>,
    document.body,
  )
}
