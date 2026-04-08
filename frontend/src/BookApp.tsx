import { Toaster } from 'sonner'
import { Navbar } from '@/components/Navbar'
import { Footer } from '@/components/Footer'
import { BookPage } from '@/components/book/BookPage'

export default function BookApp() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />
      <main className="flex-1">
        <BookPage />
      </main>
      <Footer />
      <Toaster position="bottom-right" richColors />
    </div>
  )
}
