import { useState, useEffect, useRef, useCallback } from 'react'
import './index.css'

import ClockWidget from './components/ClockWidget'
import PostureIndicator from './components/PostureIndicator'
import VoiceStatus from './components/VoiceStatus'
import ConfidenceScore from './components/ConfidenceScore'
import AinaaBrand from './components/AinaaBrand'
import AnalysisPanel from './components/AnalysisPanel'
import HairstylePanel from './components/HairstylePanel'

const BACKEND_URL = 'http://localhost:8000'
const WS_URL = 'ws://localhost:8000/ws/metrics'

function App() {
  const [postureData, setPostureData] = useState({
    status: 'ALIGNED', composite_score: 0, neck_angle: 0,
    shoulder_tilt: 0, torso_angle: 0, has_person: false, gesture: 'None'
  })
  const [voiceEnabled, setVoiceEnabled] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [analysisData, setAnalysisData] = useState(null)
  const [showTranscript, setShowTranscript] = useState(false)
  const [feedError, setFeedError] = useState(false)
  const [aiResponse, setAiResponse] = useState('')
  const [showAiResponse, setShowAiResponse] = useState(false)
  const [greeting, setGreeting] = useState(true)
  const [hairstyleData, setHairstyleData] = useState(null)
  const [isLoadingHairstyle, setIsLoadingHairstyle] = useState(false)

  const recognitionRef = useRef(null)
  const wsRef = useRef(null)
  const transcriptTimerRef = useRef(null)
  const aiResponseTimerRef = useRef(null)
  const videoRef = useRef(null)
  const wakeActiveRef = useRef(false)
  const commandTimerRef = useRef(null)
  const isProcessingRef = useRef(false)

  // Hide greeting after 3 seconds
  useEffect(() => {
    const t = setTimeout(() => setGreeting(false), 3500)
    return () => clearTimeout(t)
  }, [])

  // --- WebSocket ---
  useEffect(() => {
    let ws, reconnectTimer
    const connect = () => {
      ws = new WebSocket(WS_URL)
      ws.onopen = () => console.log('[Ainaa] WS connected')
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data)
          if (msg.type === 'posture' && msg.data) {
            setPostureData(msg.data)
            
            // --- Gesture Trigger Logic ---
            if (msg.data.gesture === 'Open Palm' && !isProcessingRef.current && !analysisData) {
              console.log('[Ainaa] Gesture trigger: Rate Outfit')
              handleCommand('rate my outfit')
            }
          }
        } catch {}
      }
      ws.onclose = () => { reconnectTimer = setTimeout(connect, 3000) }
      ws.onerror = () => ws.close()
      wsRef.current = ws
    }
    connect()
    return () => { clearTimeout(reconnectTimer); wsRef.current?.close() }
  }, [])

  // --- Browser Webcam Fallback ---
  useEffect(() => {
    const img = document.getElementById('mirror-feed-img')
    if (!img) return
    const handleError = async () => {
      setFeedError(true)
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: 'user' } })
        if (videoRef.current) videoRef.current.srcObject = stream
      } catch {}
    }
    img.addEventListener('error', handleError)
    return () => img.removeEventListener('error', handleError)
  }, [])

  // --- TTS ---
  const speak = useCallback((text) => {
    if (!window.speechSynthesis) return
    window.speechSynthesis.cancel()
    const u = new SpeechSynthesisUtterance(text)
    u.rate = 0.95; u.pitch = 1.0; u.volume = 1.0
    const voices = window.speechSynthesis.getVoices()
    const pref = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Google') || v.name.includes('Female')))
    if (pref) u.voice = pref
    window.speechSynthesis.speak(u)
  }, [])

  const showResponseToast = useCallback((text) => {
    setAiResponse(text)
    setShowAiResponse(true)
    clearTimeout(aiResponseTimerRef.current)
    aiResponseTimerRef.current = setTimeout(() => setShowAiResponse(false), 8000)
  }, [])

  // --- Hairstyle + Sunglasses Analysis ---
  const fetchHairstyle = useCallback(async () => {
    if (isLoadingHairstyle) return
    setIsLoadingHairstyle(true)
    speak('Analyzing your face shape and style…')
    showResponseToast('🔍 Detecting face shape…')
    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/analyze/hairstyle`, { method: 'POST' })
      const data = await res.json()
      setHairstyleData(data)
      speak(`I detected a ${data.face_shape} face shape. Check out your personalized hairstyle and sunglasses recommendations!`)
      showResponseToast(`✂️ ${data.face_shape} face — ${data.hairstyle_recommendations?.[0]?.name} suits you best!`)
    } catch (err) {
      console.error('[Ainaa] Hairstyle error:', err)
      speak('Sorry, hairstyle analysis failed. Please try again.')
    } finally {
      setIsLoadingHairstyle(false)
    }
  }, [isLoadingHairstyle, speak, showResponseToast])

  // --- Keyboard shortcut: H for hairstyle ---
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'h' || e.key === 'H') {
        if (!hairstyleData && !isLoadingHairstyle) fetchHairstyle()
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [fetchHairstyle, hairstyleData, isLoadingHairstyle])

  // --- Command Handler ---
  const handleCommand = useCallback(async (command) => {
    const cmd = command.toLowerCase().trim()
    if (cmd.length < 2 || isProcessingRef.current) return
    console.log('[Ainaa] Command:', cmd)
    isProcessingRef.current = true
    setIsProcessing(true)

    try {
      if (cmd.match(/hairstyle|haircut|hair style|suggest hair|face shape|what.*hair|hair.*suit/)) {
        fetchHairstyle()
      } else if (cmd.match(/rate|outfit|fit|style|wear|look|dress|cloth/)) {
        speak('Analyzing your outfit.')
        showResponseToast('🔍 Analyzing outfit...')
        const res = await fetch(`${BACKEND_URL}/api/v1/analyze/outfit`, { method: 'POST' })
        const data = await res.json()
        setAnalysisData(data)
        speak(`I rate this outfit ${data.rating} out of 10. ${data.analysis}`)
        showResponseToast(`⭐ ${data.rating}/10 — ${data.analysis}`)
      } else if (cmd.match(/posture|standing|slouch|back|straight|spine|sit/)) {
        const s = postureData
        if (s.status === 'SLOUCHING') {
          speak(`Posture score is ${Math.round(s.composite_score)}. You're slouching. Pull your shoulders back and lift your chin.`)
        } else if (s.status === 'FAIR') {
          speak(`Posture is fair at ${Math.round(s.composite_score)}. A small adjustment would help.`)
        } else {
          speak(`Posture is great. Score ${Math.round(s.composite_score)} out of 100.`)
        }
        showResponseToast(`📊 Posture: ${Math.round(s.composite_score)}/100 — ${s.status}`)
      } else if (cmd.match(/score|confidence|number/)) {
        speak(`Your confidence score is ${Math.round(postureData.composite_score)} out of 100.`)
      } else if (cmd.match(/time|clock/)) {
        const n = new Date(); const h = n.getHours() % 12 || 12; const m = n.getMinutes()
        speak(`It's ${h}:${m.toString().padStart(2, '0')} ${n.getHours() >= 12 ? 'PM' : 'AM'}.`)
      } else if (cmd.match(/hello|hi|hey|greet|how are/)) {
        speak("Hello! I'm Ainaa, your smart mirror. Ask me to rate your outfit or check your posture.")
      } else {
        // Send to Gemini AI for general conversation
        try {
          const res = await fetch(`${BACKEND_URL}/api/v1/chat`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: cmd })
          })
          const data = await res.json()
          speak(data.response)
          showResponseToast(data.response)
        } catch {
          speak("Sorry, I didn't catch that. Try: rate my outfit, or check my posture.")
        }
      }
    } catch (err) {
      console.error('[Ainaa] Command error:', err)
      speak("Something went wrong. Please try again.")
    } finally {
      isProcessingRef.current = false
      setIsProcessing(false)
      wakeActiveRef.current = false
      setIsListening(false)
    }
  }, [speak, showResponseToast, postureData])

  // --- Voice Recognition ---
  const startVoice = useCallback(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SR) { console.warn('[Ainaa] No SpeechRecognition'); return }
    if (recognitionRef.current) {
      recognitionRef.current.onend = null
      recognitionRef.current.abort()
    }

    const rec = new SR()
    rec.continuous = true
    rec.interimResults = true
    rec.lang = 'en-US'
    rec.maxAlternatives = 3

    rec.onresult = (event) => {
      let finalText = '', interimText = ''
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript
        if (event.results[i].isFinal) finalText += t
        else interimText += t
      }

      const display = finalText || interimText
      if (display) {
        setTranscript(display)
        setShowTranscript(true)
        clearTimeout(transcriptTimerRef.current)
        transcriptTimerRef.current = setTimeout(() => setShowTranscript(false), 4000)
      }

      const combined = (finalText + ' ' + interimText).toLowerCase()

      // Wake word: "hey"
      if (!wakeActiveRef.current && !isProcessingRef.current) {
        if (combined.includes('hey') || combined.includes('ainaa') || combined.includes('aina') || combined.includes('mirror')) {
          wakeActiveRef.current = true
          setIsListening(true)
          speak("I'm listening.")
          return
        }
      }

      // After wake, collect final transcript as command
      if (wakeActiveRef.current && finalText && !isProcessingRef.current) {
        let cmd = finalText.toLowerCase()
          .replace(/\b(hey|ainaa|aina|ayna|anna|mirror)\b/g, '').trim()

        if (cmd.length > 2) {
          clearTimeout(commandTimerRef.current)
          commandTimerRef.current = setTimeout(() => {
            handleCommand(cmd)
          }, 800)
        }
      }
    }

    rec.onerror = (e) => {
      if (e.error !== 'no-speech' && e.error !== 'aborted') {
        console.error('[Ainaa] Speech error:', e.error)
      }
    }

    rec.onend = () => {
      // Auto-restart to keep listening
      setTimeout(() => {
        try { rec.start() } catch {}
      }, 200)
    }

    try { rec.start(); console.log('[Ainaa] Voice recognition started') }
    catch (e) { console.error('[Ainaa] Could not start recognition:', e) }
    recognitionRef.current = rec
    setVoiceEnabled(true)
  }, [speak, handleCommand])

  // Cleanup
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.onend = null
        recognitionRef.current.abort()
      }
    }
  }, [])

  const score = Math.round(postureData.composite_score || 0)

  return (
    <div className="mirror-container" id="ainaa-mirror">

      {/* Greeting overlay */}
      {greeting && (
        <div className="greeting-overlay">
          <div className="greeting-logo">AINAA</div>
          <div className="greeting-tagline">Reflecting the best version of you</div>
        </div>
      )}

      {/* Video Feed */}
      {!feedError ? (
        <img id="mirror-feed-img" className="mirror-feed" src={`${BACKEND_URL}/stream/video`} alt="" />
      ) : (
        <video ref={videoRef} className="mirror-feed" autoPlay playsInline muted />
      )}

      {/* HUD */}
      <div className="hud-overlay">
        <div className="hud-top">
          <ClockWidget />
          <PostureIndicator status={postureData.status} score={score} neckAngle={postureData.neck_angle} />
        </div>
        <div className="hud-bottom">
          <ConfidenceScore score={score} />
          <button
            className={`glass-card hair-style-btn ${isLoadingHairstyle ? 'loading' : ''}`}
            id="hairstyle-btn"
            onClick={fetchHairstyle}
            disabled={isLoadingHairstyle}
            title="Press H or click to get hairstyle & sunglasses recommendations"
          >
            {isLoadingHairstyle ? (
              <span className="hair-btn-spinner" />
            ) : (
              <span className="hair-btn-icon">✂️</span>
            )}
            <span className="hair-btn-label">{isLoadingHairstyle ? 'Scanning…' : 'Hair & Sunnies'}</span>
          </button>
          <VoiceStatus isListening={isListening} isProcessing={isProcessing} voiceEnabled={voiceEnabled} onActivate={startVoice} />
          <AinaaBrand />
        </div>
      </div>

      {/* Transcript toast */}
      {showTranscript && transcript && (
        <div className="transcript-toast">
          <span className="highlight">🎤 </span>{transcript}
        </div>
      )}

      {/* AI Response toast */}
      {showAiResponse && aiResponse && (
        <div className="ai-response-toast">
          <span className="highlight">🤖 </span>{aiResponse}
        </div>
      )}

      {/* Analysis Panel */}
      <AnalysisPanel data={analysisData} onClose={() => setAnalysisData(null)} />

      {/* Hairstyle & Sunglasses Panel */}
      <HairstylePanel
        data={hairstyleData}
        onClose={() => setHairstyleData(null)}
        onAnalyzeOutfit={() => handleCommand('rate my outfit')}
      />
    </div>
  )
}

export default App
