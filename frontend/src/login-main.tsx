import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import LoginApp from './LoginApp'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <LoginApp />
  </StrictMode>,
)
