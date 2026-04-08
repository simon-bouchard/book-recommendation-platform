import { useState, useEffect, useRef, useCallback } from 'react'
import { SearchInput } from './SearchInput'
import { SubjectPicker } from './SubjectPicker'
import { BookGrid } from './BookGrid'
import { PaginationControls } from './PaginationControls'
import { searchBooks } from '@/lib/api'
import type { SearchResult, Pagination } from '@/types'

function parseUrlParams() {
  const p = new URLSearchParams(window.location.search)
  return {
    query: p.get('query') ?? '',
    subjects: p.get('subjects') ? p.get('subjects')!.split(',').filter(Boolean) : [],
    page: parseInt(p.get('page') ?? '0', 10),
  }
}

const EMPTY_PAGINATION: Pagination = {
  current_page: 0,
  page_size: 60,
  total_results: 0,
  total_pages: 0,
  has_next: false,
  has_prev: false,
}

export function SearchPage() {
  const initial = parseUrlParams()
  const [query, setQuery] = useState(initial.query)
  const [subjects, setSubjects] = useState<string[]>(initial.subjects)
  const [page, setPage] = useState(initial.page)

  const [results, setResults] = useState<SearchResult[]>([])
  const [pagination, setPagination] = useState<Pagination>(EMPTY_PAGINATION)
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)

  const resultsRef = useRef<HTMLDivElement>(null)
  const isFirstRender = useRef(true)

  const runSearch = useCallback(
    async (q: string, subs: string[], p: number, scrollToResults = false) => {
      setLoading(true)
      setSearched(true)
      try {
        const data = await searchBooks({ query: q, subjects: subs, page: p })
        setResults(data.results)
        setPagination(data.pagination)

        const params = new URLSearchParams()
        if (q) params.set('query', q)
        if (subs.length > 0) params.set('subjects', subs.join(','))
        if (p > 0) params.set('page', String(p))
        const newUrl = params.size > 0 ? `?${params}` : window.location.pathname
        window.history.replaceState(null, '', newUrl)

        if (scrollToResults) {
          setTimeout(() => {
            resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
          }, 50)
        }
      } finally {
        setLoading(false)
      }
    },
    [],
  )

  // Always search on mount — empty query returns popular books
  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false
      void runSearch(initial.query, initial.subjects, initial.page, false)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  function handleSubmit() {
    setPage(0)
    void runSearch(query, subjects, 0, false)
  }

  function handlePrev() {
    const newPage = page - 1
    setPage(newPage)
    void runSearch(query, subjects, newPage, true)
  }

  function handleNext() {
    const newPage = page + 1
    setPage(newPage)
    void runSearch(query, subjects, newPage, true)
  }

  return (
    <div className="px-6 pt-12 pb-10">
      {/* Search form — centered, narrow */}
      <div className="mx-auto max-w-2xl mb-10">
        <SearchInput value={query} onChange={setQuery} onSubmit={handleSubmit} loading={loading} />
        <SubjectPicker selected={subjects} onChange={setSubjects} />
      </div>

      {/* Results — full width */}
      {searched && (
        <div ref={resultsRef} className="mx-auto max-w-5xl">
          {!loading && results.length > 0 && (
            <p className="mb-4 text-sm text-muted-foreground">
              {pagination.total_results.toLocaleString()} result
              {pagination.total_results !== 1 ? 's' : ''}
            </p>
          )}

          <BookGrid results={results} loading={loading} />

          {!loading && (
            <PaginationControls
              currentPage={pagination.current_page}
              totalPages={pagination.total_pages}
              hasPrev={pagination.has_prev}
              hasNext={pagination.has_next}
              onPrev={handlePrev}
              onNext={handleNext}
            />
          )}
        </div>
      )}
    </div>
  )
}
