import { useState, useEffect, useRef } from 'react'

const DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

export default function ClockWidget() {
  const [time, setTime] = useState(new Date())

  useEffect(() => {
    const interval = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(interval)
  }, [])

  const hours = time.getHours().toString().padStart(2, '0')
  const minutes = time.getMinutes().toString().padStart(2, '0')
  const seconds = time.getSeconds().toString().padStart(2, '0')
  const day = DAYS[time.getDay()]
  const month = MONTHS[time.getMonth()]
  const date = time.getDate()

  return (
    <div className="glass-card clock-widget" id="clock-widget">
      <div className="clock-time">
        {hours}:{minutes}<span style={{ fontSize: '1.5rem', opacity: 0.4 }}>:{seconds}</span>
      </div>
      <div className="clock-date">
        {day}, {month} {date}
      </div>
    </div>
  )
}
