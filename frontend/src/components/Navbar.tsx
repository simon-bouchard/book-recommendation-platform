import { useState, useEffect, useRef } from 'react'
import { Menu, X } from 'lucide-react'

const links = [
  { href: '/', label: 'Home' },
  { href: '/search', label: 'Search' },
  { href: '/chat', label: 'Chat' },
  { href: '/profile', label: 'Profile' },
]

export function Navbar() {
  const loggedIn = document.getElementById('root')?.dataset.loggedIn === 'true'
  const current = window.location.pathname
  const [open, setOpen] = useState(false)
  const navRef = useRef<HTMLElement>(null)

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (navRef.current && !navRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    if (open) document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [open])

  return (
    <nav ref={navRef} className="sticky top-0 z-40 border-b border-border bg-background/95 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-5xl items-center justify-between px-6">

        {/* Logo */}
        <a href="/" aria-label="Home">
          <img src="/static/favicon.png" alt="" className="h-9 w-9" />
        </a>

        {/* Desktop nav */}
        <div className="hidden md:flex items-center gap-6">
          {links.map(({ href, label }) => (
            <a
              key={href}
              href={href}
              className={`text-base font-medium transition-colors hover:text-primary ${
                current === href ? 'text-primary' : 'text-muted-foreground'
              }`}
            >
              {label}
            </a>
          ))}
        </div>

        {/* Desktop auth */}
        <div className="hidden md:block">
          {loggedIn ? (
            <a href="/logout" className="text-base font-medium text-muted-foreground hover:text-primary transition-colors">
              Logout
            </a>
          ) : (
            <a href="/login" className="text-base font-medium text-muted-foreground hover:text-primary transition-colors">
              Log In
            </a>
          )}
        </div>

        {/* Hamburger button */}
        <button
          type="button"
          className="md:hidden p-2 text-muted-foreground hover:text-foreground transition-colors"
          onClick={() => setOpen((o) => !o)}
          aria-label="Toggle menu"
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {/* Mobile menu */}
      {open && (
        <div className="md:hidden absolute top-16 left-0 right-0 border-t border-b border-border bg-background/95 backdrop-blur shadow-md px-6 py-4 space-y-1">
          {links.map(({ href, label }) => (
            <a
              key={href}
              href={href}
              onClick={() => setOpen(false)}
              className={`block py-2 text-base font-medium transition-colors hover:text-primary ${
                current === href ? 'text-primary' : 'text-muted-foreground'
              }`}
            >
              {label}
            </a>
          ))}
          <div className="border-t border-border pt-3 mt-3">
            {loggedIn ? (
              <a href="/logout" className="block py-2 text-base font-medium text-muted-foreground hover:text-primary transition-colors">
                Logout
              </a>
            ) : (
              <a href="/login" className="block py-2 text-base font-medium text-muted-foreground hover:text-primary transition-colors">
                Log In
              </a>
            )}
          </div>
        </div>
      )}
    </nav>
  )
}
