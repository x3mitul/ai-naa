export default function AnalysisPanel({ data, onClose }) {
  if (!data) return null

  return (
    <div className="analysis-overlay" onClick={onClose} id="analysis-overlay">
      <div className="analysis-panel" onClick={(e) => e.stopPropagation()}>
        <h2>Style Analysis</h2>

        <div className="analysis-rating-container">
          <div className="analysis-rating">{data.rating}</div>
          <div className="analysis-rating-label">out of 10</div>
        </div>

        <p className="analysis-text">{data.analysis}</p>

        {/* New Styling Profile Grid */}
        {(data.skin_tone || data.face_shape) && (
          <div className="style-grid">
            <div className="style-badge">
              <span className="badge-icon">🎨</span>
              <span className="badge-label">Tone</span>
              <span className="badge-value">{data.skin_tone}</span>
            </div>
            <div className="style-badge">
              <span className="badge-icon">👤</span>
              <span className="badge-label">Face</span>
              <span className="badge-value">{data.face_shape}</span>
            </div>
            <div className="style-badge">
              <span className="badge-icon">💇</span>
              <span className="badge-label">Hair</span>
              <span className="badge-value">{data.hairstyle}</span>
            </div>
            <div className="style-badge">
              <span className="badge-icon">👁️</span>
              <span className="badge-label">Eyes</span>
              <span className="badge-value">{data.eye_color}</span>
            </div>
          </div>
        )}

        {/* Dynamic Recommendations */}
        <div className="recommendations">
          {data.color_suggestions && (
            <div className="rec-row">
              <span className="rec-label">Color Match</span>
              <div className="rec-pills">
                {data.color_suggestions.map((c, i) => (
                  <span key={i} className="color-pill">{c}</span>
                ))}
              </div>
            </div>
          )}
          {data.spectacles && (
            <div className="rec-row">
              <span className="rec-label">Spectacles</span>
              <span className="rec-value">{data.spectacles}</span>
            </div>
          )}
        </div>

        <div className="analysis-suggestion">💡 {data.suggestions}</div>

        <button className="analysis-close" onClick={onClose}>Dismiss</button>
      </div>
    </div>
  )
}
