import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import SignupApp from './SignupApp'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <SignupApp />
  </StrictMode>,
)
