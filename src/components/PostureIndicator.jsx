export default function PostureIndicator({ status, score, neckAngle }) {
  const statusClass = status === 'ALIGNED' ? 'good' : status === 'FAIR' ? 'fair' : 'bad'
  const dotClass = status === 'SLOUCHING' ? 'bad' : 'good'

  return (
    <div className="glass-card posture-indicator" id="posture-indicator">
      <div className={`posture-dot ${dotClass}`} />
      <div>
        <div className="posture-label">Posture</div>
        <div className={`posture-value ${statusClass}`}>{status}</div>
        <div style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.4)', marginTop: '2px', fontFamily: 'var(--font-display)' }}>
          {score}/100 · Neck {neckAngle}°
        </div>
      </div>
    </div>
  )
}
