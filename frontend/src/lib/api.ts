import type { SearchResponse, AutocompleteItem, Subject, BookDetail, CommentsResponse, SimilarBook, UserProfile, InteractionsResponse } from '@/types'

export interface SearchParams {
  query: string;
  subjects: string[];
  page: number;
  pageSize?: number;
}

export async function searchBooks(
  params: SearchParams,
  signal?: AbortSignal,
): Promise<SearchResponse> {
  const p = new URLSearchParams()
  p.set('query', params.query)
  if (params.subjects.length > 0) p.set('subjects', params.subjects.join(','))
  p.set('page', String(params.page))
  p.set('page_size', String(params.pageSize ?? 60))

  const res = await fetch(`/search/json?${p}`, { signal })
  if (!res.ok) throw new Error(`Search failed: ${res.status}`)
  return res.json() as Promise<SearchResponse>
}

export async function autocomplete(
  q: string,
  signal?: AbortSignal,
): Promise<AutocompleteItem[]> {
  if (q.length < 2) return []
  const res = await fetch(
    `/search/autocomplete?q=${encodeURIComponent(q)}&limit=5`,
    { signal },
  )
  if (!res.ok) return []
  const data = await res.json() as { suggestions: AutocompleteItem[] }
  return data.suggestions
}

export async function fetchBookDetail(itemIdx: number): Promise<BookDetail> {
  const res = await fetch(`/book/${itemIdx}/json`)
  if (!res.ok) throw new Error(`Failed to load book: ${res.status}`)
  return res.json() as Promise<BookDetail>
}

export async function fetchHasAls(itemIdx: number): Promise<boolean> {
  const res = await fetch(`/book/${itemIdx}/has_als`)
  if (!res.ok) return false
  const data = await res.json() as { has_als: boolean }
  return data.has_als
}

export async function fetchSimilarBooks(
  itemIdx: number,
  mode: 'subject' | 'als' | 'hybrid',
  alpha: number,
): Promise<SimilarBook[]> {
  const res = await fetch(`/book/${itemIdx}/similar?mode=${mode}&alpha=${alpha}`)
  if (!res.ok) {
    const data = await res.json() as { detail?: string }
    throw new Error(data.detail ?? 'Failed to load similar books')
  }
  return res.json() as Promise<SimilarBook[]>
}

export async function fetchComments(
  itemIdx: number,
  limit: number,
  cursor?: number,
): Promise<CommentsResponse> {
  const p = new URLSearchParams({ limit: String(limit) })
  if (cursor !== undefined) p.set('cursor', String(cursor))
  const res = await fetch(`/book/${itemIdx}/comments?${p}`)
  if (!res.ok) throw new Error('Failed to load comments')
  return res.json() as Promise<CommentsResponse>
}

export async function postRating(
  itemIdx: number,
  rating: number | null,
  comment: string,
): Promise<void> {
  const res = await fetch('/rating', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ item_idx: itemIdx, rating, comment }),
  })
  if (!res.ok) {
    const data = await res.json() as { detail?: string }
    throw new Error(data.detail ?? 'Failed to save rating')
  }
}

export async function deleteRating(itemIdx: number): Promise<void> {
  const res = await fetch(`/rating/${itemIdx}`, { method: 'DELETE' })
  if (!res.ok) {
    const data = await res.json() as { detail?: string }
    throw new Error(data.detail ?? 'Failed to delete rating')
  }
}

export async function fetchProfile(): Promise<UserProfile> {
  const res = await fetch('/profile/json')
  if (!res.ok) throw new Error('Failed to load profile')
  return res.json() as Promise<UserProfile>
}

export async function fetchIsWarm(): Promise<boolean> {
  const res = await fetch('/profile/is_warm')
  if (!res.ok) return false
  const data = await res.json() as { is_warm: boolean }
  return data.is_warm
}

export async function updateProfile(data: {
  username: string
  email: string
  favorite_subjects: string[]
}): Promise<void> {
  const res = await fetch('/profile/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) {
    const err = await res.json() as { detail?: string }
    throw new Error(err.detail ?? 'Failed to update profile')
  }
}

export async function deleteProfile(): Promise<void> {
  const res = await fetch('/profile', { method: 'DELETE' })
  if (!res.ok) {
    const err = await res.json() as { detail?: string }
    throw new Error(err.detail ?? 'Failed to delete profile')
  }
}

export async function fetchInteractions(
  limit: number,
  cursor?: number,
): Promise<InteractionsResponse> {
  const p = new URLSearchParams({ limit: String(limit), sort: 'newest' })
  if (cursor !== undefined) p.set('cursor', String(cursor))
  const res = await fetch(`/profile/ratings?${p}`)
  if (!res.ok) throw new Error('Failed to load interactions')
  return res.json() as Promise<InteractionsResponse>
}

export async function fetchProfileRecommendations(
  userId: number,
  mode: 'subject' | 'behavioral',
  w: number,
): Promise<SimilarBook[]> {
  const p = new URLSearchParams({
    user: String(userId),
    _id: 'true',
    top_n: '200',
    mode,
    w: String(w),
  })
  const res = await fetch(`/profile/recommend?${p}`)
  if (!res.ok) {
    const err = await res.json() as { detail?: string }
    throw new Error(err.detail ?? 'Failed to load recommendations')
  }
  return res.json() as Promise<SimilarBook[]>
}

export async function fetchChatHistory(): Promise<Array<{ u: string; a: string }>> {
  const res = await fetch('/chat/history', { credentials: 'same-origin' })
  if (!res.ok) return []
  const data = await res.json() as { turns: Array<{ u: string; a: string }> }
  return data.turns ?? []
}

export async function clearChatHistory(): Promise<void> {
  await fetch('/chat/history', { method: 'DELETE', credentials: 'same-origin' })
}

export function trackClick(itemIdx: number, source: string, mode: string): void {
  // sendBeacon guarantees delivery even when the page navigates away immediately after,
  // which is the common case when clicking a book card or inline link.
  const payload = new Blob(
    [JSON.stringify({ item_idx: itemIdx, source, mode })],
    { type: 'application/json' },
  )
  navigator.sendBeacon('/track/click', payload)
}

export async function subjectSuggestions(q?: string): Promise<Subject[]> {
  const url = q
    ? `/subjects/suggestions?q=${encodeURIComponent(q)}`
    : '/subjects/suggestions'
  const res = await fetch(url)
  if (!res.ok) return []
  const data = await res.json() as { subjects: Subject[] }
  return data.subjects
}
