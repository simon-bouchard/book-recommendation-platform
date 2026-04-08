export interface SearchResult {
  item_idx: number;
  title: string;
  author: string | null;
  cover_id: number | null;
  isbn: string | null;
  year: number | null;
  description_snippet: string | null;
  _score?: number | null;
}

export interface Pagination {
  current_page: number;
  page_size: number;
  total_results: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface SearchResponse {
  results: SearchResult[];
  pagination: Pagination;
}

export interface AutocompleteItem {
  item_idx: number;
  title: string;
  author: string | null;
  type: string;
}

export interface Subject {
  subject: string;
  count: number;
}

export interface Book {
  item_idx: number;
  title: string;
  author: string;
  year: number | null;
  description: string | null;
  cover_id: number | null;
  isbn: string | null;
  average_rating: number | null;
  rating_count: number;
  subjects: string[];
  num_pages: number | null;
}

export interface UserRating {
  rating: number | null;
  comment: string | null;
}

export interface BookDetail {
  book: Book;
  user_rating: UserRating | null;
  has_real_subjects: boolean;
  logged_in: boolean;
}

export interface Comment {
  user_id: number;
  username: string;
  rating: number | null;
  comment: string;
  rated_at: string | null;
}

export interface CommentsResponse {
  items: Comment[];
  total_count: number;
  has_more: boolean;
}

export interface UserProfile {
  id: number;
  username: string;
  email: string;
  num_books_read: number;
  num_ratings: number;
  favorite_subjects: string[];
}

export interface Interaction {
  book_id: number;
  title: string;
  author: string;
  year: number | null;
  cover_url_small: string;
  rated_at: string | null;
  user_rating?: number;
  comment?: string;
}

export interface InteractionsResponse {
  items: Interaction[];
  total_count: number;
  has_more: boolean;
  next_cursor: number | null;
}

export interface SimilarBook {
  item_idx: number;
  title: string;
  author: string | null;
  year: number | null;
  isbn: string | null;
  cover_id: number | null;
  score: number;
}

export interface ChatBook {
  item_idx: number;
  title: string;
  author?: string | null;
  year?: number | null;
  cover_id?: number | null;
  num_ratings?: number | null;
  genre?: string | null;
}

export interface ChatChunk {
  type: 'status' | 'token' | 'complete';
  content?: string | null;
  data?: {
    books?: ChatBook[];
    book_ids?: number[];
    target?: string;
    success?: boolean;
    text?: string;
  } | null;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'bot';
  text: string;
  status: string | null;
  isStreaming: boolean;
  isDone: boolean;
}
