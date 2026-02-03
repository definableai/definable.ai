import { useEffect, useState, useRef } from 'react'
import './ChatWidget.css'

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

// Declare global window type for IPC bridge
declare global {
  interface Window {
    definableChat: {
      send: (channel: string, data: any) => Promise<void>
      on: (channel: string, callback: (data: any) => void) => () => void
      emit: (channel: string, data: any) => void
    }
  }
}

interface ChatWidgetProps {
  config?: Config
}

export function ChatWidget({ config = {} }: ChatWidgetProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [widgetConfig, setWidgetConfig] = useState<Config>({
    title: config.title || 'Definable Chat',
    theme: config.theme || 'light'
  })
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const assistantMsgIndexRef = useRef<number>(-1)

  useEffect(() => {
    // Update config when props change
    setWidgetConfig({
      title: config.title || widgetConfig.title,
      theme: config.theme || widgetConfig.theme
    })
  }, [config])

  useEffect(() => {
    // Setup IPC listeners (Electron-style)
    console.log('[ChatWidget] Using IPC mode')
    
    // Listen for streaming chunks
    const unsubChunk = window.definableChat.on('chat:chunk', (data: any) => {
      const { chunk } = data
      setMessages(prev => {
        const newMessages = [...prev]
        if (assistantMsgIndexRef.current >= 0 && assistantMsgIndexRef.current < newMessages.length) {
          // Append chunk to existing content
          const currentMsg = newMessages[assistantMsgIndexRef.current]
          newMessages[assistantMsgIndexRef.current] = {
            ...currentMsg,
            content: currentMsg.content + chunk
          }
        }
        return newMessages
      })
    })

    // Listen for errors
    const unsubError = window.definableChat.on('chat:error', (data: any) => {
      console.error('[ChatWidget] IPC Error:', data.error)
      setMessages(prev => {
        const newMessages = [...prev]
        if (assistantMsgIndexRef.current >= 0) {
          newMessages[assistantMsgIndexRef.current] = {
            role: 'system',
            content: `Error: ${data.error}`,
            timestamp: Date.now()
          }
        }
        return newMessages
      })
      setLoading(false)
    })

    // Listen for completion
    const unsubComplete = window.definableChat.on('chat:message-complete', () => {
      setLoading(false)
      assistantMsgIndexRef.current = -1
    })

    return () => {
      unsubChunk()
      unsubError()
      unsubComplete()
    }
  }, [])

  useEffect(() => {
    // Auto-scroll to bottom
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = () => {
    if (!input.trim() || loading) {
      return
    }

    const userMessage = input
    setInput('')
    setLoading(true)

    // Add user message and assistant placeholder together
    setMessages(prev => {
      // Set the index for the assistant message that we're about to add
      assistantMsgIndexRef.current = prev.length + 1
      
      return [
        ...prev,
        {
          role: 'user',
          content: userMessage,
          timestamp: Date.now()
        },
        {
          role: 'assistant',
          content: '',
          timestamp: Date.now()
        }
      ]
    })

    // Send via IPC (Electron-style)
    console.log('[ChatWidget] Sending via IPC:', userMessage)
    window.definableChat.send('chat:message', { content: userMessage })
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className={`chat-widget ${widgetConfig.theme}`}>
      {/* Loading Indicator */}
      {loading && (
        <div className="loading-bar">
          Processing...
        </div>
      )}

      {/* Header */}
      <div className="chat-header">
        <h1>{widgetConfig.title}</h1>
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
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading || !input.trim()}>
          {loading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  )
}
