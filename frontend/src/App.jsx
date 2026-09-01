import { useEffect, useState } from 'react'

// The roadmap's transparency requirement: show the answer AND the passages it was
// built from, with their scores, so retrieval is inspectable rather than a black box.
// The role switch exists to make the RBAC layer visible — the same question returns
// an answer or a refusal depending on who is asking.

const ROLES = [
  { id: 'public', label: 'Public', hint: 'Default. Sees only documents tagged public.' },
  { id: 'admin', label: 'Admin', hint: 'Also sees admin-only documents.' },
]

// Each example declares the document it depends on. The questions stay here — nobody
// can auto-generate good demo questions from an arbitrary PDF — but the server tells us
// which documents are actually ingested, so we never offer a question that cannot be
// answered. An example button that returns a confusing refusal reads as a broken system.
const EXAMPLES = [
  { q: 'Where was Cannabis sativa first domesticated?', source: 'data/mota-origenes.pdf' },
  { q: 'How many accessions were analysed in the study?', source: 'data/mota-origenes.pdf' },
  { q: 'What is the internal codename for the Q3 product launch?',
    source: 'data/confidential.md', role: 'admin' },
]

function scoreLabel(source) {
  // Hybrid retrieval fuses two rankings via RRF, so its score is a fused rank score,
  // not a similarity. Labelling both "score" would invite reading them on one scale.
  if (source.similarity != null) {
    return { name: 'cosine similarity', value: source.similarity.toFixed(4) }
  }
  return { name: 'RRF score', value: source.score.toFixed(5) }
}

// Phase colours for the latency bar. Fixed order so the same phase is always the same
// colour across queries — a bar whose colours shuffle is unreadable at a glance.
const PHASES = [
  { key: 'query_embed', label: 'embed', hint: 'Turning the question into a vector' },
  { key: 'search', label: 'search', hint: 'Vector scan + BM25 + RRF fusion' },
  { key: 'generation', label: 'generate', hint: 'The LLM writing the answer' },
]

function Metrics({ metrics }) {
  const { cost_usd, total_tokens, seconds, total_seconds, config, retrieval_mode } = metrics
  // Widths are proportional to measured time, so the bar shows WHERE the second went.
  const total = total_seconds || 1

  return (
    <section className="metrics">
      <h2>
        Measured cost &amp; latency
        <span className="meta">this request, not an estimate</span>
      </h2>

      <div className="metric-row">
        <div className="metric">
          <strong>${cost_usd.toFixed(6)}</strong>
          <em>per query</em>
        </div>
        <div className="metric">
          <strong>{total_tokens.toLocaleString()}</strong>
          <em>tokens</em>
        </div>
        <div className="metric">
          <strong>{total_seconds.toFixed(2)}s</strong>
          <em>end to end</em>
        </div>
      </div>

      <div className="latency-bar" role="img"
           aria-label={`Latency breakdown: ${PHASES.map(
             (p) => `${p.label} ${(seconds[p.key] || 0).toFixed(2)}s`).join(', ')}`}>
        {PHASES.map((p) => {
          const v = seconds[p.key] || 0
          return v > 0 ? (
            <span key={p.key} className={`seg seg-${p.key}`}
                  style={{ width: `${(v / total) * 100}%` }}
                  title={`${p.hint} — ${v.toFixed(3)}s`} />
          ) : null
        })}
      </div>

      <ul className="phase-legend">
        {PHASES.map((p) => (
          <li key={p.key} title={p.hint}>
            <span className={`dot seg-${p.key}`} />
            {p.label} <strong>{(seconds[p.key] || 0).toFixed(2)}s</strong>
          </li>
        ))}
      </ul>

      <p className="metrics-note">
        Served by <code>{config}</code> using <strong>{retrieval_mode}</strong> retrieval
        — the configuration the experiment matrix selected. Generation dominates both
        cost and time; the embedding model is a fraction of a percent of the bill.
      </p>
    </section>
  )
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
  const [question, setQuestion] = useState(EXAMPLES[0].q)
  const [corpus, setCorpus] = useState(null)
  const [role, setRole] = useState('public')
  const [k, setK] = useState(5)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  // Ask the server what it can actually answer. On failure we leave corpus null and
  // fall back to showing every example — degrading to the old behaviour rather than
  // presenting an empty, apparently-broken interface.
  useEffect(() => {
    fetch('/api/corpus')
      .then((r) => (r.ok ? r.json() : null))
      .then(setCorpus)
      .catch(() => setCorpus(null))
  }, [])

  const examples = corpus
    ? EXAMPLES.filter((e) => corpus.sources.includes(e.source))
    : EXAMPLES

  async function ask(e) {
    e?.preventDefault()
    if (!question.trim() || loading) return

    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-User-Roles': role },
        body: JSON.stringify({ question, k: Number(k) }),
      })
      if (!res.ok) throw new Error(`API returned ${res.status}: ${await res.text()}`)
      setResult(await res.json())
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
          <code>{corpus ? corpus.config : 'recursive256__voyage-4-large'}</code>,{' '}
          {corpus ? corpus.retrieval_mode : 'hybrid'} retrieval.
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

        {examples.length > 0 && (
          <div className="examples">
            {examples.map((ex) => (
              <button key={ex.q} type="button" onClick={() => setQuestion(ex.q)}>
                {ex.q}
                {ex.role && <span className="needs-role">{ex.role} only</span>}
              </button>
            ))}
          </div>
        )}
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
          {result.metrics && <Metrics metrics={result.metrics} />}
          <Sources sources={result.sources} />
        </>
      )}
    </div>
  )
}
