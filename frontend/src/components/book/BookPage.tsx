import { useState, useEffect } from 'react'
import { fetchBookDetail } from '@/lib/api'
import type { BookDetail } from '@/types'
import { BookHeader } from './BookHeader'
import { BookDescription } from './BookDescription'
import { BookSubjects } from './BookSubjects'
import { RatingForm } from './RatingForm'
import { CommentsSection } from './CommentsSection'
import { SimilarBooks } from './SimilarBooks'
import { Skeleton } from '@/components/ui/skeleton'

function getItemIdx(): number {
  return parseInt(document.getElementById('root')?.dataset.itemIdx ?? '0', 10)
}

function BookPageSkeleton() {
  return (
    <div className="mx-auto max-w-5xl px-6 pt-10 pb-16">
      <div className="flex flex-col sm:flex-row gap-8">
        <div className="w-full sm:w-52 shrink-0 space-y-4">
          <Skeleton className="w-full aspect-[2/3] rounded-lg" />
          <Skeleton className="h-24 w-full rounded-lg" />
        </div>
        <div className="flex-1 space-y-4 pt-2">
          <Skeleton className="h-8 w-3/4" />
          <Skeleton className="h-5 w-1/3" />
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-32 w-full mt-6" />
        </div>
      </div>
    </div>
  )
}

export function BookPage() {
  const itemIdx = getItemIdx()
  const [detail, setDetail] = useState<BookDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchBookDetail(itemIdx)
      .then(setDetail)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : 'Failed to load book'))
  }, [itemIdx])

  if (error) {
    return (
      <div className="mx-auto max-w-5xl px-6 pt-16 text-center text-muted-foreground">
        {error}
      </div>
    )
  }

  if (!detail) return <BookPageSkeleton />

  const { book, user_rating, has_real_subjects, logged_in } = detail

  return (
    <div className="mx-auto max-w-5xl px-6 pt-10 pb-16 space-y-12">

      {/* Two-column section */}
      <div className="flex flex-col sm:flex-row gap-10">

        {/* Left column: cover + rating form (rating hidden on mobile) */}
        <div className="w-full sm:w-52 shrink-0 space-y-6">
          <BookHeader book={book} />
          <div className="hidden sm:block">
            <RatingForm itemIdx={itemIdx} loggedIn={logged_in} initialRating={user_rating} />
          </div>
        </div>

        {/* Right column: title, meta, description, subjects, comments + rating on mobile */}
        <div className="flex-1 min-w-0 space-y-10">
          <div>
            <h1 className="text-3xl font-bold leading-tight mb-1">{book.title}</h1>
            {book.author && (
              <p className="text-lg text-muted-foreground italic mb-4">{book.author}</p>
            )}
            <BookMeta book={book} />
          </div>

          {book.description && <BookDescription description={book.description} />}

          {book.subjects.length > 0 && <BookSubjects subjects={book.subjects} />}

          <CommentsSection itemIdx={itemIdx} />

          <div className="sm:hidden">
            <RatingForm itemIdx={itemIdx} loggedIn={logged_in} initialRating={user_rating} />
          </div>
        </div>

      </div>

      {/* Full-width similar books section */}
      <div className="border-t border-border pt-10">
        <SimilarBooks itemIdx={itemIdx} hasRealSubjects={has_real_subjects} />
      </div>

    </div>
  )
}

import { Calendar, BookOpen, Star } from 'lucide-react'
import type { Book } from '@/types'

function BookMeta({ book }: { book: Book }) {
  return (
    <div className="flex flex-wrap gap-x-5 gap-y-2 text-sm text-muted-foreground">
      {book.year && (
        <span className="flex items-center gap-1.5">
          <Calendar className="h-4 w-4 shrink-0" />
          {book.year}
        </span>
      )}
      {book.num_pages && (
        <span className="flex items-center gap-1.5">
          <BookOpen className="h-4 w-4 shrink-0" />
          {book.num_pages.toLocaleString()} pages
        </span>
      )}
      {book.average_rating && (
        <span className="flex items-center gap-1.5">
          <Star className="h-4 w-4 shrink-0 text-yellow-400 fill-yellow-400" />
          {book.average_rating}/10
          <span className="text-muted-foreground/60">
            ({book.rating_count.toLocaleString()} ratings)
          </span>
        </span>
      )}
      {book.isbn && (
        <span className="font-mono text-xs text-muted-foreground/60">ISBN {book.isbn}</span>
      )}
    </div>
  )
}
