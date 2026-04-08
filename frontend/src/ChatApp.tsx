import { Toaster } from 'sonner'
import { Navbar } from '@/components/Navbar'
import { Footer } from '@/components/Footer'
import { ChatPage } from '@/components/chat/ChatPage'

const root = document.getElementById('root')!
const loggedIn = root.dataset.loggedIn === 'true'

export default function ChatApp() {
  return (
    <div className="flex h-screen flex-col bg-background">
      <Navbar />
      <main className="flex-1 flex flex-col overflow-hidden">
        <ChatPage loggedIn={loggedIn} />
      </main>
      <Footer />
      <Toaster position="bottom-right" richColors />
    </div>
  )
}
