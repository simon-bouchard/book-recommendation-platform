import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import HomeApp from './HomeApp'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <HomeApp />
  </StrictMode>,
)
