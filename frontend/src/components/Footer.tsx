export function Footer() {
  return (
    <footer className="mt-auto border-t border-border bg-background">
      <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-6">
        <a
          href="https://simonbouchard.space"
          target="_blank"
          rel="noopener"
          className="text-xs text-muted-foreground hover:text-primary transition-colors"
        >
          © {new Date().getFullYear()} Simon Bouchard — Machine Learning Developer
        </a>
        <div className="flex gap-4">
          <a href="mailto:simon.bouchard31@gmail.com" className="text-xs text-muted-foreground hover:text-primary transition-colors">Email</a>
          <a href="https://www.linkedin.com/in/simon-bouchard-54580b339" target="_blank" rel="noopener" className="text-xs text-muted-foreground hover:text-primary transition-colors">LinkedIn</a>
          <a href="https://github.com/simon-bouchard" target="_blank" rel="noopener" className="text-xs text-muted-foreground hover:text-primary transition-colors">GitHub</a>
        </div>
      </div>
    </footer>
  )
}
