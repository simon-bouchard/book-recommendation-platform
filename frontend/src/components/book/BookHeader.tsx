import { useState } from 'react'
import type { Book } from '@/types'
import { coverUrl, placeholderDataURI } from '@/lib/utils'
import { Skeleton } from '@/components/ui/skeleton'

interface BookHeaderProps {
  book: Book
}

export function BookHeader({ book }: BookHeaderProps) {
  const [imgSrc, setImgSrc] = useState(
    book.cover_id ? coverUrl(book.cover_id, 'L') : placeholderDataURI(book.title),
  )
  const [imgLoaded, setImgLoaded] = useState(false)

  return (
    <div className="relative w-full aspect-[2/3] rounded-lg overflow-hidden shadow-md">
      {!imgLoaded && <Skeleton className="absolute inset-0" />}
      <img
        src={imgSrc}
        alt={`Cover for ${book.title}`}
        className="absolute inset-0 h-full w-full object-cover"
        onLoad={() => setImgLoaded(true)}
        onError={() => { setImgSrc(placeholderDataURI(book.title)); setImgLoaded(true) }}
      />
    </div>
  )
}
