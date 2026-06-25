import { useState, useEffect } from 'react'
import { fetchCompany, fetchCompanyEvents } from '../api'
import { eventColor } from '../parseWiki'
import TrialsPanel from './TrialsPanel'
import StockChart from './StockChart'


export default function CompanyView({ slug, onSelectIndication, news = [], onSelectArticle }) {
  const [data, setData] = useState(null)
  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    setData(null)
    setEvents([])
    fetchCompany(slug).then(d => { setData(d); setLoading(false) })
    fetchCompanyEvents(slug).then(d => setEvents(d.events ?? []))
  }, [slug])

  if (loading) return <div className="loading">Loading {slug}…</div>
  if (!data) return null

  const { meta, stock, drug_indications = {} } = data
  // news is pre-sorted newest-first by the backend
  const latestNews = news.find(a => a.companies?.includes(slug))
  const secEvents         = events.filter(e => e.type === 'sec')
  const researchEvents    = events.filter(e => e.type === 'research')
  const oneYearAgo        = new Date(); oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1)
  const cutoff            = oneYearAgo.toISOString().slice(0, 10)
  const recentCompletions = events.filter(e => e.type === 'trial' && e.date >= cutoff).length

  const changePct = stock?.change_pct
  const priceClass = changePct > 0 ? 'price-pos' : changePct < 0 ? 'price-neg' : 'price-neu'

  const fmtIndication = slug =>
    slug.replace('glp1', 'GLP-1').replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())

  return (
    <div>
      {/* Company header */}
      <div className="ind-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
          <h1 className="ind-title">{meta.full_name}</h1>
          {stock?.price != null && (
            <span className={priceClass} style={{ fontSize: 16, fontWeight: 500 }}>
              {meta.ticker} ${stock.price.toFixed(2)}&nbsp;
              ({changePct > 0 ? '+' : ''}{changePct?.toFixed(2)}%)
            </span>
          )}
        </div>
        <div className="ind-meta">
          <span className="chip">{meta.exchange ?? 'NYSE'}</span>
          {meta.headquarters && <span className="chip">{meta.headquarters}</span>}
          {meta.indications_active?.length > 0 && (
            <span className="chip">{meta.indications_active.length} indications</span>
          )}
          {meta.drugs?.length > 0 && (
            <span className="chip">{meta.drugs.length} drugs</span>
          )}
        </div>
      </div>

      {/* Latest news teaser — title + date only, full text lives in the article view */}
      {latestNews && (
        <div
          className="card"
          style={{ marginBottom: 10, cursor: onSelectArticle ? 'pointer' : 'default' }}
          onClick={() => onSelectArticle?.(latestNews.url)}
        >
          <p className="sec-label" style={{ marginBottom: 6 }}>Latest news</p>
          <div className="co-row">
            <div>
              <div className="evt-date">{latestNews.published_date?.slice(0, 10)} · BioSpace</div>
              <div className="co-name">{latestNews.title}</div>
            </div>
          </div>
        </div>
      )}

      {/* Stock chart */}
      {meta.ticker && (
        <div className="card" style={{ marginBottom: 10 }}>
          <StockChart slug={slug} />
        </div>
      )}

      {/* 1. Earnings & regulatory — top, full width (if exists) */}
      {secEvents.length > 0 && (
        <div className="card" style={{ marginBottom: 10 }}>
          <p className="sec-label" style={{ marginBottom: 12 }}>Earnings &amp; regulatory</p>
          <div className="event-list">
            {secEvents.slice(0, 8).map((e, i) => (
              <div key={i} className="event-row">
                <span className="evt-dot" style={{ background: eventColor(e.event) }} />
                <div>
                  <div className="evt-date">{e.date}</div>
                  <div className="evt-text">{e.event}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 2. Drug portfolio */}
      {meta.drugs?.length > 0 && (
        <>
          <p className="sec-label" style={{ marginBottom: 10 }}>Drug portfolio</p>
          <div className="drug-grid" style={{ marginBottom: 10 }}>
            {meta.drugs.map((drug, i) => {
              const indications = drug_indications[drug] ?? []
              return (
                <div key={i} className="drug-card">
                  <div className="drug-name">{drug}</div>
                  {indications.length > 0 && (
                    <div className="drug-co" style={{ marginBottom: 8 }}>
                      {indications.map(fmtIndication).join(' · ')}
                    </div>
                  )}
                  <span className="badge badge-approved">Active</span>
                </div>
              )
            })}
          </div>
        </>
      )}

      {/* 3. Clinical evidence */}
      <div style={{ marginBottom: 20 }}>
        <TrialsPanel slug={slug} researchEvents={researchEvents} recentCompletions={recentCompletions} />
      </div>

      {/* Active indications tile — commented out, restore by un-commenting the grid below */}
      {/* <div style={{
        display: 'grid',
        gridTemplateColumns: meta.indications_active?.length > 0 ? '1fr 200px' : '1fr',
        gap: 10,
        marginBottom: 20,
        alignItems: 'start',
      }}>
        <TrialsPanel slug={slug} researchEvents={researchEvents} recentCompletions={recentCompletions} />

        {meta.indications_active?.length > 0 && (
          <div className="card">
            <p className="sec-label" style={{ marginBottom: 12 }}>Active indications</p>
            <div className="co-list">
              {meta.indications_active.map((ind, i) => (
                <div key={ind}>
                  {i > 0 && <hr className="divider" style={{ margin: 0 }} />}
                  <div
                    className="co-row"
                    style={{ cursor: onSelectIndication ? 'pointer' : 'default' }}
                    onClick={() => onSelectIndication?.(ind)}
                  >
                    <div className="co-name">{fmtIndication(ind)}</div>
                    <span className="badge badge-approved" style={{ fontSize: 11 }}>Active</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div> */}

    </div>
  )
}
