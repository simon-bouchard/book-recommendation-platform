import type { SearchResult } from '@/types'
import { BookCard } from './BookCard'
import { Skeleton } from '@/components/ui/skeleton'

interface BookGridProps {
  results: SearchResult[];
  loading: boolean;
}

const SKELETON_COUNT = 12

export function BookGrid({ results, loading }: BookGridProps) {
  if (loading) {
    return (
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-5">
        {Array.from({ length: SKELETON_COUNT }).map((_, i) => (
          <div key={i} className="flex flex-col gap-2 rounded-lg border border-border p-3 pb-5">
            <Skeleton className="mx-auto h-[220px] w-[150px] rounded" />
            <Skeleton className="mx-auto h-4 w-3/4 rounded" />
            <Skeleton className="mx-auto h-3 w-1/2 rounded" />
          </div>
        ))}
      </div>
    )
  }

  if (results.length === 0) {
    return (
      <div className="py-16 text-center text-muted-foreground">
        No books found. Try a different query or remove some filters.
      </div>
    )
  }

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-5">
      {results.map((book) => (
        <BookCard key={book.item_idx} book={book} />
      ))}
    </div>
  )
}
