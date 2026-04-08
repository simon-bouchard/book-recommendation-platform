import { Toaster } from 'sonner'
import { Navbar } from '@/components/Navbar'
import { Footer } from '@/components/Footer'
import { ProfilePage } from '@/components/profile/ProfilePage'

export default function ProfileApp() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />
      <main className="flex-1">
        <ProfilePage />
      </main>
      <Footer />
      <Toaster position="bottom-right" richColors />
    </div>
  )
}
