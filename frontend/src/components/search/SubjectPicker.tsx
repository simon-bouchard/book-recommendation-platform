import { useState, useEffect, useCallback } from 'react'
import { X, Plus } from 'lucide-react'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command'
import { subjectSuggestions } from '@/lib/api'
import type { Subject } from '@/types'

const MAX_SUBJECTS = 5

interface SubjectPickerProps {
  selected: string[];
  onChange: (subjects: string[]) => void;
}

export function SubjectPicker({ selected, onChange }: SubjectPickerProps) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState<Subject[]>([])

  const fetchSuggestions = useCallback(async (q: string) => {
    const results = await subjectSuggestions(q || undefined)
    setSuggestions(results)
  }, [])

  useEffect(() => {
    fetchSuggestions(query)
  }, [query, fetchSuggestions])

  function add(subject: string) {
    if (selected.includes(subject) || selected.length >= MAX_SUBJECTS) return
    onChange([...selected, subject])
    setQuery('')
    setOpen(false)
  }

  function remove(subject: string) {
    onChange(selected.filter((s) => s !== subject))
  }

  const atMax = selected.length >= MAX_SUBJECTS

  return (
    <div className="mt-3 flex flex-wrap items-center gap-2">
      {selected.map((s) => (
        <span
          key={s}
          className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-3 py-1 text-sm font-medium text-primary"
        >
          {s}
          <button
            type="button"
            onClick={() => remove(s)}
            className="ml-0.5 rounded-full hover:text-primary/60 transition-colors"
            aria-label={`Remove ${s}`}
          >
            <X className="h-3 w-3" />
          </button>
        </span>
      ))}

      {!atMax && (
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <button
              type="button"
              className="inline-flex items-center gap-1 rounded-full border border-dashed border-border px-3 py-1 text-sm text-muted-foreground transition-colors hover:border-primary hover:text-primary"
            >
              <Plus className="h-3 w-3" />
              Add filter
            </button>
          </PopoverTrigger>
          <PopoverContent
            className="w-64 p-0"
            align="start"
            sideOffset={8}
            onMouseDown={(e) => e.preventDefault()}
          >
            <Command shouldFilter={false}>
              <CommandInput
                placeholder="Search subjects…"
                value={query}
                onValueChange={setQuery}
              />
              <CommandList>
                <CommandEmpty>No subjects found.</CommandEmpty>
                <CommandGroup>
                  {suggestions
                    .filter((s) => !selected.includes(s.subject))
                    .map((s) => (
                      <CommandItem
                        key={s.subject}
                        value={s.subject}
                        onMouseDown={(e) => { e.preventDefault(); add(s.subject) }}
                      >
                        <span className="flex-1">{s.subject}</span>
                        <span className="text-xs text-muted-foreground">
                          {s.count.toLocaleString()}
                        </span>
                      </CommandItem>
                    ))}
                </CommandGroup>
              </CommandList>
            </Command>
          </PopoverContent>
        </Popover>
      )}
    </div>
  )
}
