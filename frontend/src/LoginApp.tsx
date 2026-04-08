import { Navbar } from '@/components/Navbar'
import { Footer } from '@/components/Footer'
import { LoginPage } from '@/components/auth/LoginPage'

const root = document.getElementById('root')!

export default function LoginApp() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />
      <main className="flex-1">
        <LoginPage
          next={root.dataset.next ?? ''}
          initialError={root.dataset.error ?? ''}
          initialWarning={root.dataset.warning ?? ''}
        />
      </main>
      <Footer />
    </div>
  )
}
