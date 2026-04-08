/**
 * Lightweight Markdown → HTML renderer that also handles the custom
 * <book id="X">Title</book> syntax emitted by the LLM.
 * Ported from static/js/chat_agent.js.
 */

interface BookRef {
  id: string
  title: string
}

export function escapeHtml(s: string): string {
  return String(s ?? '').replace(/[&<>"']/g, (c) => {
    const map: Record<string, string> = {
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }
    return map[c] ?? c
  })
}

export function renderMarkdown(md: string): string {
  const bookRefs: BookRef[] = []

  // 1) Extract book refs before HTML-escaping
  let html = md.replace(/<book\s+id="(\d+)">([^<]+)<\/book>/g, (_, id, title) => {
    const idx = bookRefs.length
    bookRefs.push({ id: String(id), title: String(title) })
    return `\x00BOOKREF${idx}\x00`
  })

  html = html.replace(/\[([^\]]+)\]\((\d+)\)/g, (_, title, id) => {
    const idx = bookRefs.length
    bookRefs.push({ id: String(id), title: String(title) })
    return `\x00BOOKREF${idx}\x00`
  })

  // 2) Escape everything else
  html = escapeHtml(html)

  // 3) Fenced code blocks
  html = html.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code}</code></pre>`)

  // 4) Inline code
  html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>')

  // 5) External links
  html = html.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener">$1</a>',
  )

  // 6) Bold / italic
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  html = html.replace(/\*([^*\n]+)\*/g, '<em>$1</em>')

  // 7) Headings
  for (let lvl = 6; lvl >= 1; lvl--) {
    const hashes = '#'.repeat(lvl)
    html = html.replace(new RegExp(`^${hashes} (.*)$`, 'gm'), `<h${lvl}>$1</h${lvl}>`)
  }

  // 8) Unordered lists
  html = html.replace(/(?:^|\n)((?:[-*] .+\n?)+)/g, (_, block: string) => {
    const items = block
      .trim()
      .split(/\n/)
      .map((line: string) => line.replace(/^[-*] (.*)$/, '<li>$1</li>'))
      .join('')
    return `\n<ul>${items}</ul>`
  })

  // 9) Single newlines → <br>
  html = html.replace(/(?<!<\/(?:h[1-6]|li|ul|pre)>)\n/g, '<br>')

  // 10) Restore book refs as hoverable citation links
  bookRefs.forEach(({ id, title }, index) => {
    const safeId = escapeHtml(id)
    const safeTitle = escapeHtml(title)
    html = html.replace(
      `\x00BOOKREF${index}\x00`,
      `<a href="/book/${safeId}" class="inline-book-ref" data-book-id="${safeId}">${safeTitle}</a>`,
    )
  })

  return html
}
