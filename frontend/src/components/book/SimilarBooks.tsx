import { useState, useEffect } from 'react'
import { fetchHasAls, fetchSimilarBooks } from '@/lib/api'
import type { SimilarBook, SearchResult } from '@/types'
import { Button } from '@/components/ui/button'
import { BookGrid } from '@/components/search/BookGrid'

interface SimilarBooksProps {
  itemIdx: number
  hasRealSubjects: boolean
}

type Mode = 'subject' | 'als' | 'hybrid'

function toSearchResult(b: SimilarBook): SearchResult {
  return {
    item_idx: b.item_idx,
    title: b.title,
    author: b.author,
    cover_id: b.cover_id,
    isbn: b.isbn,
    year: b.year,
    description_snippet: null,
    _score: b.score,
  }
}

export function SimilarBooks({ itemIdx, hasRealSubjects }: SimilarBooksProps) {
  const [hasAls, setHasAls] = useState(false)
  const [mode, setMode] = useState<Mode>(hasRealSubjects ? 'subject' : 'als')
  const [alpha, setAlpha] = useState(0.5)
  const [results, setResults] = useState<SimilarBook[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [fetched, setFetched] = useState(false)

  useEffect(() => {
    fetchHasAls(itemIdx).then((has) => {
      setHasAls(has)
      if (!hasRealSubjects && has) setMode('als')
    })
  }, [itemIdx, hasRealSubjects])

  const tabs: { mode: Mode; label: string; disabled: boolean }[] = [
    { mode: 'subject', label: 'Subject', disabled: !hasRealSubjects },
    { mode: 'als', label: 'Behavioral', disabled: !hasAls },
    { mode: 'hybrid', label: 'Hybrid', disabled: !hasAls || !hasRealSubjects },
  ]

  async function getRecommendations() {
    setLoading(true)
    setError(null)
    setFetched(true)
    try {
      const data = await fetchSimilarBooks(itemIdx, mode, alpha)
      setResults(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load recommendations')
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  return (
    <section>
      <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground mb-4">
        Similar Books
      </h2>

      {/* Mode tabs + slider + button */}
      <div className="flex flex-col sm:flex-row sm:items-center gap-3 mb-6">
        <div className="inline-flex w-fit mx-auto sm:mx-0 rounded-lg border border-border overflow-hidden shrink-0">
          {tabs.map((tab) => (
            <button
              key={tab.mode}
              type="button"
              disabled={tab.disabled}
              onClick={() => { if (!tab.disabled) setMode(tab.mode) }}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                tab.disabled
                  ? 'opacity-40 cursor-not-allowed bg-background text-muted-foreground'
                  : mode === tab.mode
                  ? 'bg-primary text-primary-foreground cursor-pointer'
                  : 'bg-background text-muted-foreground hover:bg-accent cursor-pointer'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
        {mode === 'hybrid' && hasAls && (
          <div className="flex items-center gap-3 flex-1">
            <span className="text-xs text-muted-foreground shrink-0">Subject</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
              className="flex-1 cursor-pointer accent-primary"
            />
            <span className="text-xs text-muted-foreground shrink-0">Behavioral</span>
          </div>
        )}
        <Button type="button" onClick={getRecommendations} disabled={loading} className="shrink-0 w-full max-w-xs mx-auto sm:w-auto sm:max-w-none sm:mx-0 sm:ml-auto">
          {loading ? 'Loading…' : 'Get Recommendations'}
        </Button>
      </div>


      {error && (
        <p className="text-sm text-destructive mb-4">{error}</p>
      )}

      {fetched && (
        <BookGrid
          results={results.map(toSearchResult)}
          loading={loading}
        />
      )}
    </section>
  )
}
