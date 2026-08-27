import { useState } from 'react'

// The roadmap's transparency requirement: show the answer AND the passages it was
// built from, with their scores, so retrieval is inspectable rather than a black box.
// The role switch exists to make the RBAC layer visible — the same question returns
// an answer or a refusal depending on who is asking.

const ROLES = [
  { id: 'public', label: 'Public', hint: 'Default. Sees only documents tagged public.' },
  { id: 'admin', label: 'Admin', hint: 'Also sees admin-only documents.' },
]

const EXAMPLES = [
  'Where was Cannabis sativa first domesticated?',
  'How many accessions were analysed in the study?',
  'What is the internal codename for the Q3 product launch?',
]

function scoreLabel(source) {
  // Hybrid retrieval fuses two rankings via RRF, so its score is a fused rank score,
  // not a similarity. Labelling both "score" would invite reading them on one scale.
  if (source.similarity != null) {
    return { name: 'cosine similarity', value: source.similarity.toFixed(4) }
  }
  return { name: 'RRF score', value: source.score.toFixed(5) }
}

function Sources({ sources }) {
  const [open, setOpen] = useState(null)
  if (!sources.length) return null

  return (
    <section className="sources">
      <h2>
        Retrieved passages <span className="count">{sources.length}</span>
      </h2>
      <p className="sources-note">
        These are the only passages the model was allowed to use. Ranked by retrieval
        score, highest first.
      </p>
      <ol className="source-list">
        {sources.map((s, i) => {
          const { name, value } = scoreLabel(s)
          const isOpen = open === i
          return (
            <li key={`${s.source}-${s.chunk_index}`} className="source">
              <button
                className="source-head"
                onClick={() => setOpen(isOpen ? null : i)}
                aria-expanded={isOpen}
              >
                <span className="rank">#{i + 1}</span>
                <span className="source-name">
                  {s.source.split('/').pop()}
                  <span className="chunk">chunk {s.chunk_index}</span>
                </span>
                <span className="score" title={name}>
                  {value}
                  <em>{name}</em>
                </span>
                <span className={`chev ${isOpen ? 'open' : ''}`} aria-hidden="true">›</span>
              </button>
              {isOpen && <p className="passage">{s.content}</p>}
            </li>
          )
        })}
      </ol>
    </section>
  )
}

export default function App() {
  const [question, setQuestion] = useState(EXAMPLES[0])
  const [role, setRole] = useState('public')
  const [k, setK] = useState(5)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [elapsed, setElapsed] = useState(null)

  async function ask(e) {
    e?.preventDefault()
    if (!question.trim() || loading) return

    setLoading(true)
    setError(null)
    setResult(null)
    const started = performance.now()

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-User-Roles': role },
        body: JSON.stringify({ question, k: Number(k) }),
      })
      if (!res.ok) throw new Error(`API returned ${res.status}: ${await res.text()}`)
      setResult(await res.json())
      setElapsed(((performance.now() - started) / 1000).toFixed(2))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const refused = result?.answer?.toLowerCase().includes("i don't know")

  return (
    <div className="app">
      <header>
        <h1>RAG Evaluation System</h1>
        <p className="sub">
          Answers grounded strictly in retrieved passages. Serving the configuration
          the Phase&nbsp;5 experiment matrix selected:{' '}
          <code>recursive-256 × voyage-4-large</code>, hybrid retrieval.
        </p>
      </header>

      <form onSubmit={ask}>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) ask(e)
          }}
          placeholder="Ask a question about the indexed corpus…"
          rows={3}
        />

        <div className="controls">
          <div className="roles" role="group" aria-label="Query as role">
            <span className="ctl-label">Query as</span>
            {ROLES.map((r) => (
              <button
                key={r.id}
                type="button"
                title={r.hint}
                className={`role ${role === r.id ? 'active' : ''}`}
                onClick={() => setRole(r.id)}
              >
                {r.label}
              </button>
            ))}
          </div>

          <label className="k">
            <span className="ctl-label">Top-K</span>
            <input
              type="number"
              min="1"
              max="20"
              value={k}
              onChange={(e) => setK(e.target.value)}
            />
          </label>

          <button type="submit" className="ask" disabled={loading || !question.trim()}>
            {loading ? 'Retrieving…' : 'Ask'}
          </button>
        </div>

        <div className="examples">
          {EXAMPLES.map((ex) => (
            <button key={ex} type="button" onClick={() => setQuestion(ex)}>
              {ex}
            </button>
          ))}
        </div>
      </form>

      {error && (
        <div className="error">
          <strong>Request failed.</strong> {error}
        </div>
      )}

      {result && (
        <>
          <section className={`answer ${refused ? 'refused' : ''}`}>
            <h2>
              Answer
              <span className="meta">
                as <strong>{role}</strong>
                {elapsed && ` · ${elapsed}s`}
              </span>
            </h2>
            <p>{result.answer}</p>
            {refused && (
              <p className="refusal-note">
                The pipeline refused rather than guessing — either the corpus doesn't
                contain the answer, or access control filtered it out for this role.
                Try the same question as <strong>Admin</strong>.
              </p>
            )}
          </section>
          <Sources sources={result.sources} />
        </>
      )}
    </div>
  )
}
