import { useState, useEffect } from 'react'
import { fetchIsWarm, fetchProfileRecommendations } from '@/lib/api'
import type { SimilarBook, SearchResult } from '@/types'
import { Button } from '@/components/ui/button'
import { BookGrid } from '@/components/search/BookGrid'

interface ProfileRecommendationsProps {
  userId: number
  numRatings: number
}

type Mode = 'subject' | 'behavioral'

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

const RATINGS_THRESHOLD = 10

export function ProfileRecommendations({ userId, numRatings }: ProfileRecommendationsProps) {
  const [isWarm, setIsWarm] = useState(false)
  const [mode, setMode] = useState<Mode>('subject')
  const [w, setW] = useState(0.6)
  const [results, setResults] = useState<SimilarBook[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [fetched, setFetched] = useState(false)

  useEffect(() => {
    fetchIsWarm().then(setIsWarm)
  }, [userId])

  const coldUser = numRatings < RATINGS_THRESHOLD
  const eligibleNotWarm = !isWarm && numRatings >= RATINGS_THRESHOLD

  async function getRecommendations() {
    setLoading(true)
    setError(null)
    setFetched(true)
    try {
      const data = await fetchProfileRecommendations(userId, mode, w)
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
        Personalised Recommendations
      </h2>

      {/* Warm/cold status */}
      {coldUser && (
        <div className="mb-4 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          <span className="shrink-0">⚠</span>
          <span>
            Rate at least <strong>{RATINGS_THRESHOLD}</strong> books to unlock personalised recommendations.
            You've rated <strong>{numRatings}</strong> so far.
          </span>
        </div>
      )}
      {eligibleNotWarm && (
        <div className="mb-4 flex items-start gap-2 rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-800">
          <span className="shrink-0">ℹ</span>
          <span>
            You've rated <strong>{numRatings}</strong> books — you're eligible for personalised recommendations,
            but the server hasn't updated yet. Check back soon!
          </span>
        </div>
      )}

      {/* Mode tabs + slider + button */}
      <div className="flex flex-col sm:flex-row sm:items-center gap-3 mb-6">
        <div className="inline-flex w-fit mx-auto sm:mx-0 rounded-lg border border-border overflow-hidden shrink-0">
          {(['subject', 'behavioral'] as Mode[]).map((m) => (
            <button
              key={m}
              type="button"
              disabled={m === 'behavioral' && !isWarm}
              onClick={() => { if (!(m === 'behavioral' && !isWarm)) setMode(m) }}
              className={`px-4 py-2 text-sm font-medium transition-colors capitalize ${
                m === 'behavioral' && !isWarm
                  ? 'opacity-40 cursor-not-allowed bg-background text-muted-foreground'
                  : mode === m
                  ? 'bg-primary text-primary-foreground cursor-pointer'
                  : 'bg-background text-muted-foreground hover:bg-accent cursor-pointer'
              }`}
            >
              {m === 'subject' ? 'Subject' : 'Behavioral'}
            </button>
          ))}
        </div>
        {mode === 'subject' && (
          <div className="flex items-center gap-3 flex-1">
            <span className="text-xs text-muted-foreground shrink-0">Popularity</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={w}
              onChange={(e) => setW(parseFloat(e.target.value))}
              className="flex-1 cursor-pointer accent-primary"
            />
            <span className="text-xs text-muted-foreground shrink-0">Subject</span>
          </div>
        )}
        <Button type="button" onClick={getRecommendations} disabled={loading} className="shrink-0 w-full max-w-xs mx-auto sm:w-auto sm:max-w-none sm:mx-0 sm:ml-auto">
          {loading ? 'Loading…' : 'Get Recommendations'}
        </Button>
      </div>


      {error && <p className="text-sm text-destructive mb-4">{error}</p>}

      {fetched && (
        <BookGrid results={results.map(toSearchResult)} loading={loading} />
      )}
    </section>
  )
}
