import { useRef } from 'react'
import { Button } from '@/components/ui/button'

interface ChatInputProps {
  disabled: boolean
  loggedIn: boolean
  useProfile: boolean
  onUseProfileChange: (v: boolean) => void
  usageText: string | null
  onSend: (text: string) => void
}

const MAX_HEIGHT = 150

export function ChatInput({
  disabled,
  loggedIn,
  useProfile,
  onUseProfileChange,
  usageText,
  onSend,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  function resize() {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    const next = Math.min(ta.scrollHeight, MAX_HEIGHT)
    ta.style.height = `${next}px`
    ta.style.overflowY = ta.scrollHeight > MAX_HEIGHT ? 'auto' : 'hidden'
  }

  function handleSend() {
    const text = (textareaRef.current?.value ?? '').trim()
    if (!text || disabled) return
    if (textareaRef.current) {
      textareaRef.current.value = ''
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.overflowY = 'hidden'
    }
    onSend(text)
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="px-4 pt-3 pb-4">
      <div className="flex gap-2 mb-2">
        <textarea
          ref={textareaRef}
          rows={1}
          disabled={disabled}
          placeholder="Type your message…"
          onInput={resize}
          onKeyDown={handleKeyDown}
          className="flex-1 resize-none overflow-hidden rounded-lg border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <Button
          type="button"
          onClick={handleSend}
          disabled={disabled}
          className="self-end"
        >
          Send
        </Button>
      </div>
      <div className="flex items-center justify-between">
        {loggedIn ? (
          <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer select-none">
            <input
              type="checkbox"
              checked={useProfile}
              onChange={(e) => onUseProfileChange(e.target.checked)}
              className="rounded"
            />
            Use my profile
          </label>
        ) : (
          <span />
        )}
        {usageText && (
          <p className="text-xs text-muted-foreground">{usageText}</p>
        )}
      </div>
    </div>
  )
}
