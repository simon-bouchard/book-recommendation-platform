import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function coverUrl(coverId: number, size: 'S' | 'M' | 'L' = 'M'): string {
  return `https://covers.openlibrary.org/b/id/${coverId}-${size}.jpg`
}

export function placeholderDataURI(title: string): string {
  const initials = title
    .split(' ')
    .slice(0, 2)
    .map((w) => w[0]?.toUpperCase() ?? '')
    .join('')
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="150" height="220" viewBox="0 0 150 220">
    <rect width="150" height="220" fill="#e8e8e8"/>
    <text x="75" y="115" text-anchor="middle" dominant-baseline="middle"
      font-family="sans-serif" font-size="36" fill="#aaa">${initials}</text>
  </svg>`
  return `data:image/svg+xml;base64,${btoa(svg)}`
}
