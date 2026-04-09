export default function VoiceStatus({ isListening, isProcessing, voiceEnabled, onActivate }) {
  const active = isListening || isProcessing

  if (!voiceEnabled) {
    return (
      <button className="glass-card voice-activate-btn" id="voice-activate" onClick={onActivate}>
        <div className="voice-rings inactive">
          <div className="voice-core" />
        </div>
        <span className="voice-label inactive">TAP TO ENABLE VOICE</span>
      </button>
    )
  }

  const label = isProcessing ? 'Thinking...' : isListening ? 'Listening…' : 'Say "Hey"'

  return (
    <div className="glass-card voice-status" id="voice-status">
      <div className={`voice-rings ${active ? 'active' : 'inactive'}`}>
        <div className="ring" />
        <div className="ring" />
        <div className="ring" />
        <div className="voice-core" />
      </div>
      <div className={`voice-label ${active ? '' : 'inactive'}`}>
        {label}
      </div>
    </div>
  )
}
