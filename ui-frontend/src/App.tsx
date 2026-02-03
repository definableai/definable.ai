import { useEffect, useState, useRef } from 'react'
import './App.css'

interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  avatar?: string
  timestamp: number
}

interface Config {
  title?: string
  theme?: string
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [connected, setConnected] = useState(false)
  const [config, setConfig] = useState<Config>({ title: 'Definable Chat', theme: 'light' })
  const wsRef = useRef<WebSocket | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Connect to WebSocket
    const ws = new WebSocket(`ws://${window.location.host}/ws`)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('Connected to server')
      setConnected(true)
      // Notify server that UI is ready
      ws.send(JSON.stringify({
        type: 'ui_ready',
        data: {}
      }))
    }

    ws.onmessage = (event) => {
      const command = JSON.parse(event.data)
      handleCommand(command)
    }

    ws.onclose = () => {
      console.log('Disconnected from server')
      setConnected(false)
      // Attempt reconnection
      setTimeout(() => {
        window.location.reload()
      }, 2000)
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    return () => {
      ws.close()
    }
  }, [])

  useEffect(() => {
    // Auto-scroll to bottom
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleCommand = (command: any) => {
    console.log('Received command:', command)

    switch (command.type) {
      case 'add_message':
        setMessages(prev => [...prev, command.data])
        break
      case 'clear_chat':
        setMessages([])
        break
      case 'update_config':
        setConfig(prev => ({ ...prev, ...command.data }))
        if (command.data.title) {
          document.title = command.data.title
        }
        break
    }
  }

  const sendMessage = () => {
    if (!input.trim() || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return
    }

    wsRef.current.send(JSON.stringify({
      type: 'user_message',
      data: { content: input }
    }))

    setInput('')
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="chat-container">
      {/* Connection Status */}
      <div className={`status-bar ${connected ? 'connected' : 'disconnected'}`}>
        {connected ? '● Connected' : '○ Disconnected'}
      </div>

      {/* Header */}
      <div className="chat-header">
        <h1>{config.title}</h1>
      </div>

      {/* Messages */}
      <div className="messages-container">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <div className="avatar">
              {msg.avatar ? (
                <img src={msg.avatar} alt="avatar" />
              ) : (
                <span>{msg.role.charAt(0).toUpperCase()}</span>
              )}
            </div>
            <div className="message-bubble">
              <div className="message-content">{msg.content}</div>
              <div className="message-timestamp">
                {new Date(msg.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="input-area">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type a message..."
          rows={2}
          disabled={!connected}
        />
        <button onClick={sendMessage} disabled={!connected || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  )
}

export default App
