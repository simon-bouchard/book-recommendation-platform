import { useState } from 'react'

interface BookSubjectsProps {
  subjects: string[]
}

const INITIAL_VISIBLE = 12

export function BookSubjects({ subjects }: BookSubjectsProps) {
  const [expanded, setExpanded] = useState(false)
  const visible = expanded ? subjects : subjects.slice(0, INITIAL_VISIBLE)
  const hidden = subjects.length - INITIAL_VISIBLE

  return (
    <section>
      <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground mb-3">
        Subjects
      </h2>
      <div className="flex flex-wrap gap-2">
        {visible.map((s) => (
          <span
            key={s}
            className="inline-block rounded-full bg-secondary px-3 py-1 text-xs font-medium text-secondary-foreground"
          >
            {s}
          </span>
        ))}
        {!expanded && hidden > 0 && (
          <button
            type="button"
            onClick={() => setExpanded(true)}
            className="inline-block rounded-full border border-dashed border-border px-3 py-1 text-xs text-muted-foreground hover:text-primary hover:border-primary transition-colors cursor-pointer"
          >
            +{hidden} more
          </button>
        )}
        {expanded && hidden > 0 && (
          <button
            type="button"
            onClick={() => setExpanded(false)}
            className="inline-block rounded-full border border-dashed border-border px-3 py-1 text-xs text-muted-foreground hover:text-primary hover:border-primary transition-colors cursor-pointer"
          >
            Show less
          </button>
        )}
      </div>
    </section>
  )
}
