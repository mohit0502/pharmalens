import { useState, useRef, useEffect } from 'react'
import { streamAsk } from '../api'

const SparkIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
    <circle cx="8" cy="8" r="6.5" stroke="#7a8099" strokeWidth="1.2" />
    <path d="M6 8h4M8 6v4" stroke="#7a8099" strokeWidth="1.2" strokeLinecap="round" />
  </svg>
)

function toolLabel(name, input) {
  if (name === 'read_wiki_page') return `Reading ${input.page_path ?? '…'}`
  if (name === 'list_wiki_pages') return `Listing ${input.prefix || 'wiki'}…`
  if (name === 'get_stock_price') return `Fetching price for ${input.company_slug}…`
  if (name === 'get_stock_history') return `Fetching ${input.period ?? '1mo'} history for ${input.company_slug}…`
  if (name === 'get_company_news') return `Fetching news for ${input.company_slug}…`
  if (name === 'search_wiki') return `Searching wiki for "${input.query}"…`
  if (name === 'search_company_wiki') return `Searching ${input.company_slug} for "${input.query}"…`
  return name
}

export default function AIBar({ indication, company, article, displayName }) {
  const [question, setQuestion] = useState('')
  // history = [{question, toolCalls: [{name, input, done}], answer, streaming}]
  const [history, setHistory] = useState([])
  const [streaming, setStreaming] = useState(false)
  const inputRef = useRef(null)
  const bottomRef = useRef(null)

  // Scroll to bottom whenever history updates
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [history])

  const updateLast = (updater) =>
    setHistory(prev => {
      const next = [...prev]
      next[next.length - 1] = updater(next[next.length - 1])
      return next
    })

  const submit = async q => {
    if (!q.trim() || streaming) return
    setStreaming(true)
    setQuestion('')

    // Prior turns sent to the backend as context — only the question/answer
    // text, not tool calls, since that's all the backend's contents-builder
    // replays (see api/agent.py:run_agent).
    const priorTurns = history
      .filter(entry => entry.answer && !entry.streaming)
      .map(entry => ({ question: entry.question, answer: entry.answer }))

    // Append new entry — keep all previous entries intact
    setHistory(prev => [...prev, { question: q, toolCalls: [], answer: '', streaming: true }])

    try {
      for await (const event of streamAsk(q, indication ?? null, company ?? null, article ?? null, priorTurns)) {
        if (event.type === 'tool_call') {
          updateLast(entry => ({
            ...entry,
            toolCalls: [...entry.toolCalls, { name: event.name, input: event.input, done: false }],
          }))
        } else if (event.type === 'tool_result') {
          updateLast(entry => {
            const toolCalls = entry.toolCalls.map((t, i) =>
              i === entry.toolCalls.length - 1 ? { ...t, done: true } : t
            )
            return { ...entry, toolCalls }
          })
        } else if (event.type === 'text') {
          updateLast(entry => ({ ...entry, answer: entry.answer + event.content }))
        } else if (event.type === 'done') {
          updateLast(entry => ({ ...entry, streaming: false }))
          setStreaming(false)
        }
      }
    } catch (err) {
      updateLast(entry => ({ ...entry, answer: `Error: ${err.message}`, streaming: false }))
      setStreaming(false)
    }
  }

  const contextLabel = article
    ? 'Asking about this article'
    : displayName
      ? `Asking about ${displayName}`
      : 'Ask about any pharma topic'

  return (
    <div className="ai-panel">
      <div className="ai-panel-header">
        <div className="ai-panel-header-text">
          <div className="ai-panel-title">Research</div>
          <div className="ai-panel-context">{contextLabel}</div>
        </div>
        {history.length > 0 && (
          <button
            type="button"
            className="ai-clear-btn"
            onClick={() => setHistory([])}
            disabled={streaming}
            title="Clear chat"
          >
            Clear chat
          </button>
        )}
      </div>

      <div className="ai-section">
        {/* Conversation history — grows to fill available space */}
        {history.length > 0 ? (
          <div className="ai-history">
            {history.map((entry, i) => (
              <div key={i} className="ai-exchange">
                <div className="ai-question">{entry.question}</div>
                <div className="ai-response">
                  {entry.toolCalls.length > 0 && (
                    <div className="tool-chips">
                      {entry.toolCalls.map((t, j) => (
                        <span key={j} className={`tool-chip ${t.done ? 'done' : ''}`}>
                          {t.done ? '✓' : '⟳'} {toolLabel(t.name, t.input)}
                        </span>
                      ))}
                    </div>
                  )}
                  {entry.streaming && !entry.answer && (
                    <div className="ai-thinking">Thinking…</div>
                  )}
                  {entry.answer && <div className="ai-text">{entry.answer}</div>}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        ) : (
          <div className="ai-empty" />
        )}

        {/* Input + chips — always pinned to bottom */}
        <div className="ai-footer">
          <div className="ai-bar" onClick={() => inputRef.current?.focus()}>
            <SparkIcon />
            <input
              ref={inputRef}
              className="ai-input"
              type="text"
              placeholder="Ask anything…"
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && submit(question)}
              disabled={streaming}
            />
            {streaming
              ? <span className="ai-spinner" />
              : <span className="ai-hint">↵</span>
            }
          </div>
        </div>
      </div>
    </div>
  )
}
