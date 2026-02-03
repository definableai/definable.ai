/**
 * Definable IPC Bridge - Electron-style IPC for web (window.postMessage-like)
 * 
 * Provides Electron-style IPC communication between frontend and backend:
 * - window.definableChat.send(channel, data) - Send to backend
 * - window.definableChat.on(channel, callback) - Listen to backend
 * 
 * Internally uses HTTP/SSE but feels like Electron IPC!
 */

class DefinableIPCBridge {
  constructor(apiUrl = '/api') {
    this.apiUrl = apiUrl;
    this.listeners = new Map();
    this.streaming = false;
    
    // Create custom event target for message handling
    this.eventTarget = new EventTarget();
    
    console.log('[IPC Bridge] Initialized');
  }

  /**
   * Send message to backend (like Electron's ipcRenderer.send)
   * @param {string} channel - Channel name (e.g., 'chat:message')
   * @param {any} data - Data to send
   */
  async send(channel, data) {
    console.log(`[IPC Bridge] Sending to channel: ${channel}`, data);
    
    // Emit 'will-send' event
    this.emit('will-send', { channel, data });
    
    try {
      // For chat messages, use streaming
      if (channel === 'chat:message') {
        await this._sendChatMessage(data);
      } else {
        // Generic channel handler
        const response = await fetch(`${this.apiUrl}/ipc`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ channel, data })
        });
        
        if (response.ok) {
          const result = await response.json();
          this.emit(channel + ':reply', result);
        }
      }
    } catch (error) {
      console.error(`[IPC Bridge] Error sending to ${channel}:`, error);
      this.emit('error', { channel, error: error.message });
    }
  }

  /**
   * Listen to messages from backend (like Electron's ipcRenderer.on)
   * @param {string} channel - Channel name
   * @param {Function} callback - Callback function
   */
  on(channel, callback) {
    console.log(`[IPC Bridge] Listening to channel: ${channel}`);
    
    if (!this.listeners.has(channel)) {
      this.listeners.set(channel, []);
    }
    
    this.listeners.get(channel).push(callback);
    
    // Add event listener
    this.eventTarget.addEventListener(channel, (event) => {
      callback(event.detail);
    });
    
    return () => this.off(channel, callback);
  }

  /**
   * Remove listener (like Electron's ipcRenderer.off)
   */
  off(channel, callback) {
    const callbacks = this.listeners.get(channel);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  /**
   * Emit event to listeners
   */
  emit(channel, data) {
    console.log(`[IPC Bridge] Emitting to channel: ${channel}`, data);
    
    const event = new CustomEvent(channel, { detail: data });
    this.eventTarget.dispatchEvent(event);
    
    // Also call registered listeners
    const callbacks = this.listeners.get(channel);
    if (callbacks) {
      callbacks.forEach(cb => cb(data));
    }
  }

  /**
   * Send chat message with streaming support
   */
  async _sendChatMessage(data) {
    const { content } = data;
    
    // Emit message-start event
    this.emit('chat:message-start', { content });
    
    try {
      const response = await fetch(`${this.apiUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content })
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      // Read streaming response
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const eventData = JSON.parse(line.substring(6));
            
            if (eventData.chunk) {
              // Emit only the chunk (not accumulated content)
              this.emit('chat:chunk', { chunk: eventData.chunk });
            }
            
            if (eventData.done) {
              // Emit done event (no content needed)
              this.emit('chat:message-complete', {});
            }
            
            if (eventData.error) {
              this.emit('chat:error', { error: eventData.error });
            }
          }
        }
      }
    } catch (error) {
      this.emit('chat:error', { error: error.message });
    }
  }

  /**
   * Invoke and wait for response (like Electron's ipcRenderer.invoke)
   */
  async invoke(channel, data) {
    return new Promise((resolve, reject) => {
      const replyChannel = channel + ':reply';
      
      // Set up one-time listener for reply
      const cleanup = this.on(replyChannel, (result) => {
        cleanup();
        resolve(result);
      });
      
      // Send message
      this.send(channel, data).catch(reject);
      
      // Timeout after 30 seconds
      setTimeout(() => {
        cleanup();
        reject(new Error(`IPC timeout on channel: ${channel}`));
      }, 30000);
    });
  }
}

// Initialize global IPC bridge (like Electron's ipcRenderer)
window.definableChat = new DefinableIPCBridge('/api');

console.log('[IPC Bridge] Ready - Use window.definableChat.send() and window.definableChat.on()');
