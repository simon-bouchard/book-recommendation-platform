import { useState } from 'react'
import type { SearchResult } from '@/types'
import { coverUrl, placeholderDataURI } from '@/lib/utils'
import { Skeleton } from '@/components/ui/skeleton'
import { trackClick } from '@/lib/api'

interface BookCardProps {
  book: SearchResult;
  source?: string;
  mode?: string;
}

export function BookCard({ book, source, mode }: BookCardProps) {
  const [imgSrc, setImgSrc] = useState(
    book.cover_id ? coverUrl(book.cover_id, 'M') : placeholderDataURI(book.title),
  )
  const [imgLoaded, setImgLoaded] = useState(false)

  return (
    <a
      href={`/book/${book.item_idx}`}
      onClick={() => {
  console.log('[track] click', book.item_idx, source, mode)
  if (source && mode) trackClick(book.item_idx, source, mode)
}}
      className="group flex flex-col rounded-lg border border-border bg-card p-3 pb-5 shadow-sm transition-all duration-300 hover:-translate-y-1 hover:shadow-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <div className="relative mb-3 w-full aspect-[2/3] shrink-0">
        {!imgLoaded && <Skeleton className="absolute inset-0 rounded" />}
        <img
          src={imgSrc}
          alt={book.title}
          className="absolute inset-0 h-full w-full rounded object-cover"
          onLoad={() => setImgLoaded(true)}
          onError={() => {
            setImgSrc(placeholderDataURI(book.title))
            setImgLoaded(true)
          }}
        />
      </div>

      <h3 className="mb-1 flex min-h-[2.4em] items-center justify-center text-center text-sm font-semibold leading-tight">
        <span className="line-clamp-2">{book.title}</span>
      </h3>

      {book.author && (
        <p className="truncate text-center text-sm text-muted-foreground italic">
          {book.author}
        </p>
      )}

      {book.year && (
        <p className="truncate text-center text-xs text-muted-foreground/70">
          {book.year}
        </p>
      )}
    </a>
  )
}
