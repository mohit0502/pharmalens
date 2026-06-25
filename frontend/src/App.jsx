import { useState, useEffect, useCallback } from 'react'
import { fetchIndications, fetchCompanies, fetchStocks, fetchNews } from './api'
import TickerBar from './components/TickerBar'
import Sidebar from './components/Sidebar'
import IndicationHub from './components/IndicationHub'
import CompanyView from './components/CompanyView'
import NewsView from './components/NewsView'
import AIBar from './components/AIBar'

const SIDEBAR_MIN = 200, SIDEBAR_MAX = 480, SIDEBAR_DEFAULT = 280
const AI_MIN = 280, AI_MAX = 600, AI_DEFAULT = 360

function usePersistedWidth(key, defaultValue, min, max) {
  const [width, setWidth] = useState(() => {
    const stored = Number(localStorage.getItem(key))
    return stored >= min && stored <= max ? stored : defaultValue
  })
  useEffect(() => { localStorage.setItem(key, String(width)) }, [key, width])
  const clamp = useCallback(v => Math.min(max, Math.max(min, v)), [min, max])
  return [width, setWidth, clamp]
}

// Drag-to-resize handle: `direction` is +1 if dragging right grows the panel
// (left sidebar) or -1 if dragging right shrinks it (right AI panel).
function ResizeHandle({ width, setWidth, clamp, direction }) {
  const [dragging, setDragging] = useState(false)

  const onMouseDown = e => {
    e.preventDefault()
    setDragging(true)
    const startX = e.clientX
    const startWidth = width

    const onMouseMove = me => {
      const dx = (me.clientX - startX) * direction
      setWidth(clamp(startWidth + dx))
    }
    const onMouseUp = () => {
      setDragging(false)
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
  }

  return <div className={`resize-handle${dragging ? ' dragging' : ''}`} onMouseDown={onMouseDown} />
}

export default function App() {
  const [indications, setIndications] = useState([])
  const [companies, setCompanies] = useState([])
  const [news, setNews] = useState([])
  const [stocks, setStocks] = useState({}) // keyed by ticker
  const [activeIndication, setActiveIndication] = useState(null)
  const [activeCompany, setActiveCompany] = useState(null)
  const [activeArticle, setActiveArticle] = useState(null)

  const [sidebarWidth, setSidebarWidth, clampSidebar] =
    usePersistedWidth('pharmalens.sidebarWidth', SIDEBAR_DEFAULT, SIDEBAR_MIN, SIDEBAR_MAX)
  const [aiWidth, setAiWidth, clampAi] =
    usePersistedWidth('pharmalens.aiWidth', AI_DEFAULT, AI_MIN, AI_MAX)

  const aiDisplayName = activeCompany
    ? companies.find(c => c.slug === activeCompany)?.full_name ?? activeCompany
    : activeIndication
      ? indications.find(i => i.slug === activeIndication)?.display_name ?? activeIndication
      : null

  const loadStocks = () =>
    fetchStocks().then(data => {
      const byTicker = {}
      data.forEach(s => { byTicker[s.ticker] = s })
      setStocks(byTicker)
    })

  useEffect(() => {
    fetchIndications().then(setIndications)
    fetchCompanies().then(data => {
      setCompanies(data)
      if (data.length) {
        const sorted = [...data].sort((a, b) => a.full_name.localeCompare(b.full_name))
        setActiveCompany(sorted[0].slug)
      }
    })
    fetchNews().then(data => setNews(data.articles ?? []))
    loadStocks()

    // Refresh stock prices every 60 s
    const interval = setInterval(loadStocks, 60_000)
    return () => clearInterval(interval)
  }, [])

  const selectArticle = url => { setActiveArticle(url); setActiveIndication(null); setActiveCompany(null) }

  // Sidebar feed only shows the last 24h — articles get scraped daily, so an
  // unbounded list would just grow forever; the per-company teaser below
  // covers older news on a company's own page instead.
  const dayAgo = Date.now() - 24 * 60 * 60 * 1000
  const recentNews = news.filter(a => new Date(a.published_date).getTime() >= dayAgo)

  return (
    <div className="app-wrapper">
      <TickerBar stocks={Object.values(stocks)} />
      <div className="app" style={{ gridTemplateColumns: `${sidebarWidth}px auto 1fr auto ${aiWidth}px` }}>
        <Sidebar
          indications={indications}
          companies={companies}
          news={recentNews}
          activeIndication={activeIndication}
          activeCompany={activeCompany}
          activeArticle={activeArticle}
          onSelectIndication={slug => { setActiveIndication(slug); setActiveCompany(null); setActiveArticle(null) }}
          onSelectCompany={slug => { setActiveCompany(slug); setActiveIndication(null); setActiveArticle(null) }}
          onSelectArticle={selectArticle}
        />
        <ResizeHandle width={sidebarWidth} setWidth={setSidebarWidth} clamp={clampSidebar} direction={1} />
        <main className="main">
          {activeIndication && (
            <IndicationHub
              key={activeIndication}
              slug={activeIndication}
              stocks={stocks}
              companies={companies}
              onSelectCompany={slug => { setActiveCompany(slug); setActiveIndication(null) }}
            />
          )}
          {activeCompany && (
            <CompanyView
              key={activeCompany}
              slug={activeCompany}
              stocks={stocks}
              onSelectArticle={selectArticle}
              onSelectIndication={slug => { setActiveIndication(slug); setActiveCompany(null) }}
            />
          )}
          {activeArticle && <NewsView key={activeArticle} url={activeArticle} />}
        </main>
        <ResizeHandle width={aiWidth} setWidth={setAiWidth} clamp={clampAi} direction={-1} />
        <AIBar
          indication={activeIndication}
          company={activeCompany}
          article={activeArticle}
          displayName={aiDisplayName}
        />
      </div>
    </div>
  )
}
