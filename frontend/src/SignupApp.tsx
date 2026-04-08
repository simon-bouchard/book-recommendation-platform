import { Navbar } from '@/components/Navbar'
import { Footer } from '@/components/Footer'
import { SignupPage } from '@/components/auth/SignupPage'

const root = document.getElementById('root')!

export default function SignupApp() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />
      <main className="flex-1">
        <SignupPage initialError={root.dataset.error ?? ''} />
      </main>
      <Footer />
    </div>
  )
}
