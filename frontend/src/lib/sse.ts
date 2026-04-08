/**
 * Async generator that reads a fetch Response body as Server-Sent Events
 * and yields parsed JSON objects from `data: {...}` lines.
 */
export async function* parseSseStream(response: Response): AsyncGenerator<unknown> {
  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed.startsWith('data:')) continue
        const raw = trimmed.slice(5).trim()
        if (!raw || raw === '[DONE]') continue
        try {
          yield JSON.parse(raw)
        } catch {
          // malformed JSON — skip
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}
