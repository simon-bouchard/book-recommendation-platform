import { useState, useEffect, useRef, useCallback } from 'react'
import { Search } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { autocomplete } from '@/lib/api'
import type { AutocompleteItem } from '@/types'

interface SearchInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  loading?: boolean;
}

const DEBOUNCE_MS = 300

export function SearchInput({ value, onChange, onSubmit, loading }: SearchInputProps) {
  const [suggestions, setSuggestions] = useState<AutocompleteItem[]>([])
  const [selectedIdx, setSelectedIdx] = useState(-1)
  const [open, setOpen] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const fetchSuggestions = useCallback((q: string) => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    if (q.length < 2) {
      setSuggestions([])
      setOpen(false)
      return
    }
    debounceRef.current = setTimeout(async () => {
      abortRef.current?.abort()
      abortRef.current = new AbortController()
      try {
        const results = await autocomplete(q, abortRef.current.signal)
        setSuggestions(results)
        setOpen(results.length > 0)
        setSelectedIdx(-1)
      } catch {
        // aborted — ignore
      }
    }, DEBOUNCE_MS)
  }, [])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchSuggestions(value)
  }, [value, fetchSuggestions])

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!open) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setSelectedIdx((i) => Math.min(i + 1, suggestions.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setSelectedIdx((i) => Math.max(i - 1, -1))
    } else if (e.key === 'Enter' && selectedIdx >= 0) {
      e.preventDefault()
      const item = suggestions[selectedIdx]
      if (item) window.location.href = `/book/${item.item_idx}`
    } else if (e.key === 'Escape') {
      setOpen(false)
      setSelectedIdx(-1)
    }
  }

  function close() {
    setOpen(false)
    setSelectedIdx(-1)
  }

  return (
    <div className="relative">
      <div className="relative flex items-center">
        <Search className="absolute left-4 h-5 w-5 text-muted-foreground pointer-events-none" />
        <input
          type="search"
          placeholder="Search by title, author, or description…"
          className="h-13 w-full rounded-xl border border-border bg-background pl-12 pr-28 text-base shadow-sm transition-colors focus:outline-none focus:ring-2 focus:ring-ring"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && selectedIdx < 0) {
              close()
              onSubmit()
            } else {
              handleKeyDown(e)
            }
          }}
          onBlur={() => setTimeout(close, 150)}
          onFocus={() => suggestions.length > 0 && setOpen(true)}
          autoComplete="off"
          aria-autocomplete="list"
          aria-controls="autocomplete-list"
          aria-activedescendant={selectedIdx >= 0 ? `ac-item-${selectedIdx}` : undefined}
        />
        <Button
          className="absolute right-2 h-9"
          onClick={onSubmit}
          disabled={loading}
        >
          {loading ? 'Searching…' : 'Search'}
        </Button>
      </div>

      {open && suggestions.length > 0 && (
        <ul
          id="autocomplete-list"
          role="listbox"
          onMouseDown={(e) => e.preventDefault()}
          className="absolute z-50 mt-1 w-full overflow-hidden rounded-xl border border-border bg-popover shadow-md"
        >
          {suggestions.map((item, idx) => (
            <li key={item.item_idx} role="option" aria-selected={idx === selectedIdx}>
              <a
                id={`ac-item-${idx}`}
                href={`/book/${item.item_idx}`}
                className={`block border-b border-border/50 px-4 py-3 last:border-0 hover:bg-accent ${
                  idx === selectedIdx ? 'bg-primary text-primary-foreground' : ''
                }`}
              >
                <span className={`block text-sm font-semibold ${idx === selectedIdx ? 'text-primary-foreground' : 'text-foreground'}`}>
                  {item.title}
                </span>
                {item.author && (
                  <span className={`block text-xs italic ${idx === selectedIdx ? 'text-primary-foreground/80' : 'text-muted-foreground'}`}>
                    {item.author}
                  </span>
                )}
              </a>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
