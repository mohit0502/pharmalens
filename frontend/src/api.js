// In production (Vercel), set VITE_API_URL to your Render backend URL.
// In local dev, leave it unset — Vite's proxy forwards /api to localhost:8000.
const BASE = import.meta.env.VITE_API_URL ?? ''

export async function fetchIndications() {
  const r = await fetch(`${BASE}/api/indications`)
  return r.json()
}

export async function fetchCompanies() {
  const r = await fetch(`${BASE}/api/companies`)
  return r.json()
}

export async function fetchStocks() {
  const r = await fetch(`${BASE}/api/stocks`)
  return r.json()
}

export async function fetchIndication(slug) {
  const r = await fetch(`${BASE}/api/indication/${slug}`)
  return r.json()
}

export async function fetchCompany(slug) {
  const r = await fetch(`${BASE}/api/company/${slug}`)
  return r.json()
}

export async function fetchCompanyTrials(slug) {
  const r = await fetch(`${BASE}/api/company/${slug}/trials`)
  return r.json()
}

export async function fetchCompanyEvents(slug) {
  const r = await fetch(`${BASE}/api/company/${slug}/events`)
  return r.json()
}

export async function fetchStockHistory(slug, period = '1d') {
  const r = await fetch(`${BASE}/api/company/${slug}/stock-history?period=${period}`)
  return r.json()
}

export async function fetchNews() {
  const r = await fetch(`${BASE}/api/news`)
  return r.json()
}

export async function fetchCompanyNews(slug) {
  const r = await fetch(`${BASE}/api/company/${slug}/news`)
  return r.json()
}

export async function fetchArticle(url) {
  const r = await fetch(`${BASE}/api/news/article?url=${encodeURIComponent(url)}`)
  return r.json()
}

// POST /api/ask — returns an async generator of SSE event objects
export async function* streamAsk(question, indication, company, article, history = []) {
  const response = await fetch(`${BASE}/api/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      indication: indication || null,
      company: company || null,
      article: article || null,
      history,
    }),
  })

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // SSE frames are separated by double newline
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''

    for (const frame of frames) {
      for (const line of frame.split('\n')) {
        if (line.startsWith('data: ')) {
          try {
            yield JSON.parse(line.slice(6))
          } catch {
            // skip malformed frame
          }
        }
      }
    }
  }
}
