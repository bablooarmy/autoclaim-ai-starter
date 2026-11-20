import { useEffect, useState } from 'react'
import { ShieldCheck, Car, FileSearch, Activity, Upload } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_BASE || ''

export default function App() {
  const [imageFile, setImageFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [description, setDescription] = useState(
    'Rear-ended at a traffic light. Deep dent on rear bumper and cracked left taillight. Plate KA-01-AB-1234.'
  )
  const [policyTier, setPolicyTier] = useState('COLLISION')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])
  const [evalReport, setEvalReport] = useState(null)

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/claims`)
      if (res.ok) setHistory(await res.json())
    } catch {
      /* ignore */
    }
  }

  useEffect(() => {
    fetchHistory()
  }, [])

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setImageFile(file)
    setPreviewUrl(URL.createObjectURL(file))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!imageFile) {
      setError('Please select a car damage image to assess.')
      return
    }
    setLoading(true)
    setError('')
    try {
      const form = new FormData()
      form.append('image', imageFile)
      form.append('description', description)
      form.append('policy_tier', policyTier)

      const res = await fetch(`${API_BASE}/api/v1/claims/assess`, {
        method: 'POST',
        body: form,
      })
      if (!res.ok) throw new Error(`Request failed (${res.status})`)
      const data = await res.json()
      setResult(data)
      fetchHistory()
    } catch (err) {
      setError(err.message || 'Assessment failed')
    } finally {
      setLoading(false)
    }
  }

  const runEvals = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/evals/run`, { method: 'POST' })
      if (res.ok) setEvalReport(await res.json())
    } catch {
      /* ignore */
    }
  }

  return (
    <div className="container">
      <header className="header">
        <div>
          <h1 style={{ margin: 0, fontSize: '1.5rem', display: 'flex', alignItems: 'center', gap: 10 }}>
            <Car size={26} color="#38bdf8" /> AutoClaim AI
          </h1>
          <p style={{ margin: '4px 0 0', color: '#94a3b8', fontSize: '0.9rem' }}>
            100% Local Multi-Agent Car Damage Severity &amp; Qdrant Policy RAG (Ollama + LangGraph)
          </p>
        </div>
        <button className="btn-secondary" onClick={runEvals}>
          Run Local Evals
        </button>
      </header>

      <div className="grid">
        <div>
          <form className="card" onSubmit={handleSubmit}>
            <h2 style={{ marginTop: 0, fontSize: '1.15rem' }}>Submit Damage Claim</h2>

            <label className="label">Car Damage Photo</label>
            <input className="input" type="file" accept="image/*" onChange={handleFileChange} />
            {previewUrl && (
              <img
                src={previewUrl}
                alt="Uploaded vehicle"
                style={{ width: '100%', maxHeight: 220, objectFit: 'cover', borderRadius: 8, marginBottom: 14 }}
              />
            )}

            <label className="label">Policy Coverage Tier</label>
            <select className="select" value={policyTier} onChange={(e) => setPolicyTier(e.target.value)}>
              <option value="COMPREHENSIVE">Comprehensive ($500 deductible)</option>
              <option value="COLLISION">Collision ($1,000 deductible)</option>
              <option value="LIABILITY_ONLY">Liability Only (No own-vehicle coverage)</option>
            </select>

            <label className="label">Incident Description (PII auto-redacted by Guardrail Agent)</label>
            <textarea
              className="textarea"
              rows={4}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />

            {error && <p style={{ color: '#f87171', fontSize: '0.9rem' }}>{error}</p>}

            <button className="btn" type="submit" disabled={loading}>
              <Upload size={16} style={{ verticalAlign: 'middle', marginRight: 6 }} />
              {loading ? 'Running Multi-Agent Workflow...' : 'Assess Damage Severity'}
            </button>
          </form>

          {evalReport && (
            <div className="card">
              <h3 style={{ marginTop: 0, fontSize: '1rem' }}>Golden Evaluation Suite</h3>
              <p style={{ margin: '4px 0 10px', color: '#94a3b8', fontSize: '0.9rem' }}>
                Accuracy: <strong>{(evalReport.accuracy * 100).toFixed(0)}%</strong> ({evalReport.passed_cases}/
                {evalReport.total_cases}) · Avg RAG Faithfulness:{' '}
                <strong>{evalReport.avg_faithfulness}</strong>
              </p>
              {evalReport.results.map((r) => (
                <div key={r.case_id} className="trace-step">
                  <span>{r.case_id}: {r.predicted_severity}</span>
                  <span style={{ color: r.passed ? '#4ade80' : '#f87171' }}>
                    {r.passed ? 'PASS' : `FAIL (expected ${r.expected_severity})`}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div>
          {result ? (
            <>
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h2 style={{ margin: 0, fontSize: '1.15rem' }}>Claim {result.claim_id}</h2>
                  <span className={`badge badge-${result.severity}`}>{result.severity}</span>
                </div>
                <p style={{ color: '#cbd5e1', marginTop: 10 }}>{result.recommended_action}</p>
                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: '0.9rem', color: '#94a3b8' }}>
                  <span>
                    Estimate: <strong style={{ color: '#f8fafc' }}>
                      ${result.estimated_cost_min_usd.toLocaleString()} – ${result.estimated_cost_max_usd.toLocaleString()}
                    </strong>
                  </span>
                  <span>
                    Deductible: <strong style={{ color: '#f8fafc' }}>${result.deductible_usd.toLocaleString()}</strong>
                  </span>
                  <span>
                    Covered: <strong style={{ color: result.coverage_eligible ? '#4ade80' : '#f87171' }}>
                      {result.coverage_eligible ? 'Yes' : 'No'}
                    </strong>
                  </span>
                </div>
              </div>

              <div className="card">
                <h3 style={{ marginTop: 0, fontSize: '1rem', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Activity size={18} color="#38bdf8" /> Multi-Agent Execution Trace
                </h3>
                {result.agent_trace.map((step, i) => (
                  <div key={i} className="trace-step">
                    <div>
                      <strong>{step.agent_name}</strong>
                      <div style={{ fontSize: '0.82rem', color: '#94a3b8' }}>{step.summary}</div>
                    </div>
                    <span style={{ fontSize: '0.8rem', color: '#38bdf8' }}>{step.duration_ms} ms</span>
                  </div>
                ))}
              </div>

              <div className="card">
                <h3 style={{ marginTop: 0, fontSize: '1rem', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <FileSearch size={18} color="#38bdf8" /> Qdrant Policy Citations
                </h3>
                {result.policy_citations.map((c, i) => (
                  <div key={i} className="citation">
                    <strong>{c.doc_id} · {c.section}</strong> (score: {c.score})
                    <div style={{ color: '#cbd5e1', marginTop: 4 }}>{c.excerpt}</div>
                  </div>
                ))}
              </div>

              <div className="card">
                <h3 style={{ marginTop: 0, fontSize: '1rem', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <ShieldCheck size={18} color="#4ade80" /> AI Governance &amp; Safety
                </h3>
                <p style={{ margin: '4px 0', fontSize: '0.9rem' }}>
                  PII Redacted: <strong>{result.governance.pii_redacted ? result.governance.redacted_fields.join(', ') : 'None detected'}</strong>
                </p>
                <p style={{ margin: '4px 0', fontSize: '0.9rem' }}>
                  RAG Faithfulness Score: <strong>{result.governance.rag_faithfulness_score}</strong> ·
                  Human Review Required: <strong>{result.governance.human_review_required ? 'Yes' : 'No'}</strong>
                </p>
                {result.governance.escalation_reasons.length > 0 && (
                  <ul style={{ margin: '6px 0 0', paddingLeft: 18, color: '#fbbf24', fontSize: '0.86rem' }}>
                    {result.governance.escalation_reasons.map((r, idx) => <li key={idx}>{r}</li>)}
                  </ul>
                )}
              </div>
            </>
          ) : (
            <div className="card">
              <h2 style={{ marginTop: 0, fontSize: '1.1rem' }}>Recent Claims ({history.length})</h2>
              {history.length === 0 ? (
                <p style={{ color: '#94a3b8' }}>Upload a car damage photo on the left to run the 4-agent workflow.</p>
              ) : (
                history.slice(0, 5).map((item) => (
                  <div key={item.claim_id} className="trace-step" onClick={() => setResult(item)} style={{ cursor: 'pointer' }}>
                    <span><strong>{item.claim_id}</strong> · {item.policy_tier}</span>
                    <span className={`badge badge-${item.severity}`}>{item.severity}</span>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
