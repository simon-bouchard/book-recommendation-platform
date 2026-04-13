import { useState, useRef, useEffect } from 'react'
import { parseSseStream } from '@/lib/sse'
import { clearChatHistory, fetchChatHistory, trackClick } from '@/lib/api'
import type { ChatMessage, ChatBook, ChatChunk } from '@/types'
import { MessageBubble } from './MessageBubble'
import { ChatInput } from './ChatInput'
import { BookHoverCard } from './BookHoverCard'

interface ChatPageProps {
  loggedIn: boolean
}

type RateLimitState =
  | { kind: 'ok' }
  | { kind: 'countdown'; remaining: number; label: string }
  | { kind: 'stopped'; message: string }

function genId() {
  return Math.random().toString(36).slice(2)
}

const USE_PROFILE_KEY = 'chat.useProfile'

export function ChatPage({ loggedIn }: ChatPageProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [disabled, setDisabled] = useState(false)
  const [rateLimit, setRateLimit] = useState<RateLimitState>({ kind: 'ok' })
  const [usageText, setUsageText] = useState<string | null>(null)
  const [useProfile, setUseProfile] = useState(() => {
    if (!loggedIn) return false
    const saved = localStorage.getItem(USE_PROFILE_KEY)
    return saved !== null ? saved === 'true' : true
  })
  const [hoverState, setHoverState] = useState<{ book: ChatBook; top: number; left: number } | null>(null)

  const bookDataMap = useRef<Map<string, ChatBook>>(new Map())
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null)

  function scrollToBottom() {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Restore history on mount
  useEffect(() => {
    fetchChatHistory().then((turns) => {
      if (turns.length === 0) return
      const hydrated: ChatMessage[] = turns.flatMap((t) => [
        { id: genId(), role: 'user', text: t.u, status: null, isStreaming: false, isDone: true },
        { id: genId(), role: 'bot',  text: t.a, status: null, isStreaming: false, isDone: true },
      ])
      setMessages(hydrated)
    })
  }, [])

  // Persist use-profile toggle
  function handleUseProfileChange(v: boolean) {
    setUseProfile(v)
    localStorage.setItem(USE_PROFILE_KEY, String(v))
  }

  function startCountdown(seconds: number, label: string) {
    if (countdownRef.current) clearInterval(countdownRef.current)
    let remaining = Math.max(1, seconds | 0)
    setRateLimit({ kind: 'countdown', remaining, label })
    setDisabled(true)
    countdownRef.current = setInterval(() => {
      remaining -= 1
      if (remaining <= 0) {
        clearInterval(countdownRef.current!)
        countdownRef.current = null
        setRateLimit({ kind: 'ok' })
        setDisabled(false)
      } else {
        setRateLimit({ kind: 'countdown', remaining, label })
      }
    }, 1000)
  }

  function updateUsageFromHeaders(headers: Headers) {
    const dayC = headers.get('X-RateLimit-Day-Count')
    const dayL = headers.get('X-RateLimit-Day-Limit')
    const minC = headers.get('X-RateLimit-Min-Count')
    const minL = headers.get('X-RateLimit-Min-Limit')
    const sysC = headers.get('X-RateLimit-System-Day-Count')
    const sysL = headers.get('X-RateLimit-System-Day-Limit')
    if (dayC && dayL && minC && minL && sysC && sysL) {
      setUsageText(`Today: ${dayC}/${dayL} · Per minute: ${minC}/${minL} · System: ${sysC}/${sysL}`)
    }
  }

  function updateBotMessage(id: string, patch: Partial<ChatMessage>) {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, ...patch } : m)),
    )
  }

  async function sendMessage(text: string) {
    // Append user message
    const userMsg: ChatMessage = { id: genId(), role: 'user', text, status: null, isStreaming: false, isDone: true }
    const botId = genId()
    const botMsg: ChatMessage = { id: botId, role: 'bot', text: '', status: 'Thinking…', isStreaming: false, isDone: false }
    setMessages((prev) => [...prev, userMsg, botMsg])
    setDisabled(true)

    try {
      const resp = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ user_text: text, use_profile: useProfile }),
      })

      updateUsageFromHeaders(resp.headers)

      if (resp.status === 401) {
        updateBotMessage(botId, { text: 'Please log in to use the chatbot.', status: null, isDone: true })
        setDisabled(false)
        return
      }

      if (resp.status === 429) {
        const reason = resp.headers.get('X-RateLimit-Block-Reason') ?? ''
        const retryAfter = parseInt(resp.headers.get('Retry-After') ?? '60', 10)
        let detail = 'Rate limit exceeded.'
        try {
          const j = await resp.json() as { detail?: string }
          if (j?.detail) detail = j.detail
        } catch { /* ignore */ }

        updateBotMessage(botId, { text: detail, status: null, isDone: true })

        if (reason === 'identity_minute') {
          startCountdown(retryAfter, 'You are sending messages too quickly')
        } else if (reason === 'identity_day' || reason === 'system_day') {
          setRateLimit({ kind: 'stopped', message: `${detail} Resets at midnight.` })
          setDisabled(true)
        } else {
          startCountdown(retryAfter, detail)
        }
        return
      }

      if (!resp.ok) {
        updateBotMessage(botId, {
          text: `Something went wrong (${resp.status}). Please try again.`,
          status: null,
          isDone: true,
        })
        setDisabled(false)
        return
      }

      // Stream processing
      let streamingStarted = false
      let accText = ''

      for await (const raw of parseSseStream(resp)) {
        const chunk = raw as ChatChunk

        if (chunk.type === 'status') {
          updateBotMessage(botId, { status: chunk.content ?? 'Processing…' })

        } else if (chunk.type === 'token' && chunk.content) {
          if (!streamingStarted) {
            streamingStarted = true
            updateBotMessage(botId, { status: null, isStreaming: true })
          }
          accText += chunk.content
          updateBotMessage(botId, { text: accText })

        } else if (chunk.type === 'complete') {
          const books = chunk.data?.books ?? []
          books.forEach((book) => {
            if (book?.item_idx != null) {
              bookDataMap.current.set(String(book.item_idx), book)
            }
          })
          updateBotMessage(botId, { isStreaming: false, isDone: true })
          setDisabled(false)
        }
      }

      // Guard: stream closed without complete chunk
      updateBotMessage(botId, { isStreaming: false, isDone: true })
      setDisabled(false)

    } catch (err) {
      console.error(err)
      updateBotMessage(botId, {
        text: 'Network error. Please try again.',
        status: null,
        isStreaming: false,
        isDone: true,
      })
      setDisabled(false)
    }
  }

  // Document-level mouseover: any movement to a non-book-link clears the card;
  // movement onto a book link shows it. No mouseout needed.
  useEffect(() => {
    const CARD_WIDTH = 220

    const handleOver = (e: MouseEvent) => {
      const link = (e.target as Element).closest?.('a.inline-book-ref[data-book-id]') as HTMLAnchorElement | null
      if (!link) {
        setHoverState(null)
        return
      }
      const book = bookDataMap.current.get(link.dataset.bookId ?? '')
      if (!book) {
        setHoverState(null)
        return
      }
      const rect = link.getBoundingClientRect()
      let left = rect.right + 12
      let top = rect.top
      if (left + CARD_WIDTH > window.innerWidth - 10) left = rect.left - CARD_WIDTH - 12
      top = Math.min(top, window.innerHeight - 200)
      setHoverState({ book, top, left })
    }

    const handleClick = (e: MouseEvent) => {
      const link = (e.target as Element).closest?.('a.inline-book-ref[data-book-id]') as HTMLAnchorElement | null
      if (!link) return
      const bookId = parseInt(link.dataset.bookId ?? '', 10)
      if (!isNaN(bookId)) trackClick(bookId, 'chatbot', 'chatbot')
    }

    document.addEventListener('mouseover', handleOver)
    document.addEventListener('click', handleClick)
    return () => {
      document.removeEventListener('mouseover', handleOver)
      document.removeEventListener('click', handleClick)
    }
  }, [])

  async function handleClearConversation() {
    await clearChatHistory()
    setMessages([])
  }

  const inputProps = {
    disabled: disabled || rateLimit.kind === 'stopped',
    loggedIn,
    useProfile,
    onUseProfileChange: handleUseProfileChange,
    usageText,
    onSend: sendMessage,
    hasMessages: messages.length > 0,
    onClear: handleClearConversation,
  }

  const rateLimitBanner = rateLimit.kind !== 'ok' ? (
    <div className={`mx-4 mt-2 rounded-lg border px-4 py-2.5 text-sm ${
      rateLimit.kind === 'stopped'
        ? 'border-destructive/30 bg-destructive/10 text-destructive'
        : 'border-amber-200 bg-amber-50 text-amber-800'
    }`}>
      {rateLimit.kind === 'countdown'
        ? `${rateLimit.label} — wait ${rateLimit.remaining}s`
        : rateLimit.message}
    </div>
  ) : null

  // ── Empty state: centered input like ChatGPT/Claude ──────────────────────
  if (messages.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-6 px-6 pb-8">
        <div className="text-center max-w-sm space-y-3">
          <p className="text-sm text-muted-foreground">
            Ask for book recommendations, compare titles, or ask about how this site works.
            {loggedIn && ' Toggle "Use my profile" below to personalise results.'}
          </p>
          {!loggedIn && (
            <div className="flex items-start gap-2 rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-800 text-left">
              <span className="shrink-0">ℹ</span>
              <span>
                <strong>Tip:</strong> You'll have higher daily and per-minute limits if you{' '}
                <a href="/login?next=/chat" className="underline font-medium">log in</a>.
              </span>
            </div>
          )}
        </div>

        <div className="w-full max-w-2xl rounded-xl border border-border bg-card shadow-sm">
          {rateLimitBanner}
          <ChatInput {...inputProps} />
        </div>
      </div>
    )
  }

  // ── Messages state: standard chat layout ─────────────────────────────────
  return (
    <>
      <div className="flex flex-1 flex-col min-h-0 overflow-hidden mx-auto w-full max-w-3xl px-6 pt-4 pb-4">
        <div
          ref={scrollContainerRef}
          onScroll={() => setHoverState(null)}
          className="flex-1 overflow-y-auto py-4 space-y-4"
        >
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
          <div ref={messagesEndRef} />
        </div>

        {rateLimitBanner}

        <div className="shrink-0 rounded-xl border border-border bg-card shadow-sm mt-2">
          <ChatInput {...inputProps} />
        </div>
      </div>

      {hoverState && (
        <BookHoverCard
          book={hoverState.book}
          top={hoverState.top}
          left={hoverState.left}
        />
      )}
    </>
  )
}
