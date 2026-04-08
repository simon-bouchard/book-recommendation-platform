import { useRef } from 'react'
import { renderMarkdown } from '@/lib/markdown'
import type { ChatMessage } from '@/types'

interface MessageBubbleProps {
  message: ChatMessage
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const innerRef = useRef<HTMLDivElement>(null)

  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-2xl rounded-tr-sm bg-primary px-4 py-2.5 text-sm text-primary-foreground">
          {message.text}
        </div>
      </div>
    )
  }

  // Bot bubble
  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-2xl rounded-tl-sm border border-border bg-card px-4 py-2.5 text-sm">
        {message.status && !message.isStreaming ? (
          <p className="text-muted-foreground italic">{message.status}</p>
        ) : (
          <div
            ref={innerRef}
            className="prose prose-sm max-w-none [&_a.inline-book-ref]:text-primary [&_a.inline-book-ref]:underline [&_a.inline-book-ref]:decoration-dotted [&_a.inline-book-ref]:cursor-pointer"
            // biome-ignore lint/security/noDangerouslySetInnerHtml: trusted markdown renderer output
            dangerouslySetInnerHTML={{ __html: renderMarkdown(message.text) || (message.isStreaming ? '' : '') }}
          />
        )}
        {message.isStreaming && (
          <span className="inline-block w-1.5 h-3.5 ml-0.5 bg-current opacity-70 animate-pulse align-middle" />
        )}
      </div>
    </div>
  )
}
