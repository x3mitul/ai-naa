import { useEffect, useRef } from 'react'

// Face shape SVG icons
const FACE_SHAPE_ICONS = {
  Oval: (
    <svg viewBox="0 0 60 80" className="face-shape-svg">
      <ellipse cx="30" cy="40" rx="22" ry="32" fill="none" stroke="currentColor" strokeWidth="2.5" />
      <line x1="14" y1="58" x2="46" y2="58" stroke="currentColor" strokeWidth="1.5" strokeDasharray="3,2" />
    </svg>
  ),
  Round: (
    <svg viewBox="0 0 70 70" className="face-shape-svg">
      <circle cx="35" cy="35" r="28" fill="none" stroke="currentColor" strokeWidth="2.5" />
      <line x1="12" y1="52" x2="58" y2="52" stroke="currentColor" strokeWidth="1.5" strokeDasharray="3,2" />
    </svg>
  ),
  Square: (
    <svg viewBox="0 0 70 75" className="face-shape-svg">
      <rect x="8" y="8" width="54" height="60" rx="8" fill="none" stroke="currentColor" strokeWidth="2.5" />
      <line x1="8" y1="55" x2="62" y2="55" stroke="currentColor" strokeWidth="1.5" strokeDasharray="3,2" />
    </svg>
  ),
  Heart: (
    <svg viewBox="0 0 70 80" className="face-shape-svg">
      <path d="M35 70 C35 70 8 45 8 22 Q8 8 22 10 Q35 8 35 20 Q35 8 48 10 Q62 8 62 22 C62 45 35 70 35 70Z" fill="none" stroke="currentColor" strokeWidth="2.5" />
      <line x1="18" y1="22" x2="52" y2="22" stroke="currentColor" strokeWidth="1.5" strokeDasharray="3,2" />
    </svg>
  ),
  Diamond: (
    <svg viewBox="0 0 70 90" className="face-shape-svg">
      <path d="M35 5 L60 35 L35 85 L10 35 Z" fill="none" stroke="currentColor" strokeWidth="2.5" />
      <line x1="10" y1="55" x2="60" y2="55" stroke="currentColor" strokeWidth="1.5" strokeDasharray="3,2" />
    </svg>
  ),
  Oblong: (
    <svg viewBox="0 0 55 90" className="face-shape-svg">
      <ellipse cx="27" cy="45" rx="18" ry="38" fill="none" stroke="currentColor" strokeWidth="2.5" />
      <line x1="12" y1="65" x2="43" y2="65" stroke="currentColor" strokeWidth="1.5" strokeDasharray="3,2" />
    </svg>
  ),
}

const SUNGLASSES_ICONS = {
  Aviator: '🕶️',
  Wayfarer: '🕶️',
  Round: '👓',
  Square: '🕶️',
  Rectangle: '🕶️',
  'Cat-Eye': '🕶️',
  Geometric: '🕶️',
  Oval: '👓',
  Oversized: '🕶️',
  Rimless: '👓',
  Wraparound: '🕶️',
}

const HAIR_EMOJIS = {
  waves: '〰️', curls: '🌀', straight: '➖', bangs: '✂️',
  layers: '📐', bun: '🎀', pixie: '✂️', bob: '💇', Default: '💇'
}

function getHairEmoji(name) {
  const lower = name.toLowerCase()
  for (const [key, emoji] of Object.entries(HAIR_EMOJIS)) {
    if (lower.includes(key)) return emoji
  }
  return '💇'
}

export default function HairstylePanel({ data, onClose, onAnalyzeOutfit }) {
  const panelRef = useRef(null)

  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onClose])

  if (!data) return null

  const { face_shape, confidence, measurements, hairstyle_recommendations = [], hairstyles_to_avoid = [], sunglasses_recommendations = [] } = data
  const shapeIcon = FACE_SHAPE_ICONS[face_shape] || FACE_SHAPE_ICONS['Oval']
  const confPct = Math.round((confidence || 0.75) * 100)

  return (
    <div className="analysis-overlay hairstyle-overlay" onClick={onClose} id="hairstyle-overlay">
      <div className="hairstyle-panel" ref={panelRef} onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div className="hp-header">
          <div className="hp-title-row">
            <span className="hp-icon">✂️</span>
            <h2 className="hp-title">Style Advisor</h2>
          </div>
          <button className="hp-close-btn" onClick={onClose} aria-label="Close">✕</button>
        </div>

        {/* Face Shape Card */}
        <div className="face-shape-card">
          <div className="face-shape-icon-wrap" style={{ color: 'var(--accent-cyan)' }}>
            {shapeIcon}
          </div>
          <div className="face-shape-info">
            <div className="face-shape-label">Detected Face Shape</div>
            <div className="face-shape-name">{face_shape}</div>
            <div className="face-conf-bar-wrap">
              <div className="face-conf-bar" style={{ width: `${confPct}%` }} />
            </div>
            <div className="face-conf-text">{confPct}% confidence</div>
          </div>
          {measurements && (
            <div className="face-measurements">
              {measurements.aspect_ratio && (
                <div className="face-metric">
                  <span className="fm-label">Aspect</span>
                  <span className="fm-value">{measurements.aspect_ratio}</span>
                </div>
              )}
              {measurements.jaw_forehead_ratio && (
                <div className="face-metric">
                  <span className="fm-label">Jaw/Brow</span>
                  <span className="fm-value">{measurements.jaw_forehead_ratio}</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Scrollable body */}
        <div className="hp-body">

          {/* Hairstyle Recommendations */}
          {hairstyle_recommendations.length > 0 && (
            <section className="hp-section">
              <h3 className="hp-section-title"><span>💇</span> Recommended Hairstyles</h3>
              <div className="hair-cards-grid">
                {hairstyle_recommendations.map((rec, i) => (
                  <div key={i} className="hair-card" style={{ animationDelay: `${i * 0.07}s` }}>
                    <div className="hair-card-top">
                      <span className="hair-emoji">{getHairEmoji(rec.name)}</span>
                      <span className="hair-name">{rec.name}</span>
                    </div>
                    <p className="hair-desc">{rec.description}</p>
                    <div className="hair-tags">
                      {(rec.tags || []).map((t, j) => (
                        <span key={j} className="hair-tag">{t}</span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Hairstyles to Avoid */}
          {hairstyles_to_avoid.length > 0 && (
            <section className="hp-section">
              <h3 className="hp-section-title avoid-title"><span>🚫</span> Styles to Avoid</h3>
              <div className="avoid-list">
                {hairstyles_to_avoid.map((item, i) => (
                  <div key={i} className="avoid-item">
                    <span className="avoid-x">✕</span>
                    <div>
                      <span className="avoid-name">{item.name}</span>
                      {item.reason && <span className="avoid-reason"> — {item.reason}</span>}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Sunglasses */}
          {sunglasses_recommendations.length > 0 && (
            <section className="hp-section">
              <h3 className="hp-section-title sunglass-title"><span>🕶️</span> Sunglasses for You</h3>
              <div className="sunglass-cards">
                {sunglasses_recommendations.map((sg, i) => (
                  <div key={i} className="sunglass-card" style={{ animationDelay: `${i * 0.08}s` }}>
                    <div className="sg-top">
                      <span className="sg-icon">{SUNGLASSES_ICONS[sg.name] || '🕶️'}</span>
                      <div>
                        <div className="sg-name">{sg.name}</div>
                        <span className={`sg-vibe vibe-${(sg.vibe || 'classic').toLowerCase()}`}>{sg.vibe}</span>
                      </div>
                    </div>
                    <p className="sg-reason">{sg.reason}</p>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* CTA */}
          <div className="hp-cta-row">
            <button className="hp-cta-btn" onClick={() => { onClose(); onAnalyzeOutfit && onAnalyzeOutfit() }}>
              👗 Also Rate My Outfit
            </button>
            <button className="analysis-close hp-dismiss" onClick={onClose}>Dismiss</button>
          </div>
        </div>
      </div>
    </div>
  )
}
