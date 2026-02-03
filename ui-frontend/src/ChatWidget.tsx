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
  apiUrl?: string
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
    theme: config.theme || 'light',
    apiUrl: config.apiUrl || '/api'
  })
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Update config when props change
    setWidgetConfig({
      title: config.title || widgetConfig.title,
      theme: config.theme || widgetConfig.theme,
      apiUrl: config.apiUrl || widgetConfig.apiUrl
    })
  }, [config])

  useEffect(() => {
    // Auto-scroll to bottom
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async () => {
    if (!input.trim() || loading) {
      return
    }

    const userMessage = input
    setInput('')
    setLoading(true)

    // Add user message immediately
    const timestamp = Date.now()
    setMessages(prev => [...prev, {
      role: 'user',
      content: userMessage,
      timestamp
    }])

    // Add placeholder for assistant response
    const assistantMsgIndex = messages.length + 1
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: '',
      timestamp: Date.now()
    }])

    try {
      // Send message to backend and stream response
      const response = await fetch(`${widgetConfig.apiUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content: userMessage }),
      })

      if (!response.ok) {
        throw new Error('Failed to get response from server')
      }

      // Read streaming response
      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let fullContent = ''

      while (reader) {
        const { done, value } = await reader.read()
        if (done) break

        // Decode chunk and add to buffer
        buffer += decoder.decode(value, { stream: true })

        // Process complete SSE messages
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.substring(6))
            
            if (data.chunk) {
              // Append chunk to assistant message
              fullContent += data.chunk
              setMessages(prev => {
                const newMessages = [...prev]
                newMessages[assistantMsgIndex] = {
                  ...newMessages[assistantMsgIndex],
                  content: fullContent
                }
                return newMessages
              })
            }
            
            if (data.error) {
              // Show error
              setMessages(prev => {
                const newMessages = [...prev]
                newMessages[assistantMsgIndex] = {
                  role: 'system',
                  content: `Error: ${data.error}`,
                  timestamp: Date.now()
                }
                return newMessages
              })
            }
          }
        }
      }
    } catch (error) {
      console.error('Error sending message:', error)
      setMessages(prev => {
        const newMessages = [...prev]
        newMessages[assistantMsgIndex] = {
          role: 'system',
          content: `Error: ${error}`,
          timestamp: Date.now()
        }
        return newMessages
      })
    } finally {
      setLoading(false)
    }
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
