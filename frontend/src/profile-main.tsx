import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import ProfileApp from './ProfileApp'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ProfileApp />
  </StrictMode>,
)
