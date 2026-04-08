import { Navbar } from '@/components/Navbar'
import { Footer } from '@/components/Footer'
import { HomePage } from '@/components/home/HomePage'

const root = document.getElementById('root')!
const loggedIn = root.dataset.loggedIn === 'true'

export default function HomeApp() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />
      <main className="flex-1">
        <HomePage loggedIn={loggedIn} />
      </main>
      <Footer />
    </div>
  )
}
