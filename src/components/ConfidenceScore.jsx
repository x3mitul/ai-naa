export default function ConfidenceScore({ score }) {
  return (
    <div className="glass-card confidence-score" id="confidence-score">
      <div className="confidence-label">Confidence</div>
      <div className="confidence-value">{score}</div>
      <div className="confidence-unit">/ 100</div>
    </div>
  )
}
