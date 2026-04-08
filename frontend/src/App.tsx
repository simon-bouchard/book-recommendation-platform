import { Navbar } from '@/components/Navbar'
import { Footer } from '@/components/Footer'
import { SearchPage } from '@/components/search/SearchPage'

export default function App() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />
      <main className="flex-1">
        <SearchPage />
      </main>
      <Footer />
    </div>
  )
}
