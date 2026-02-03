import React from 'react'
import ReactDOM from 'react-dom/client'
import { ChatWidget } from './ChatWidget'

/**
 * Definable Chat Widget - Web Component
 * 
 * Usage:
 * <definable-chat 
 *   title="My Chat"
 *   theme="dark"
 *   ws-url="ws://localhost:8000/ws"
 * ></definable-chat>
 */
class DefinableChatElement extends HTMLElement {
  private root: ReactDOM.Root | null = null
  private _shadowRoot: ShadowRoot

  static get observedAttributes() {
    return ['title', 'theme', 'api-url']
  }

  constructor() {
    super()
    this._shadowRoot = this.attachShadow({ mode: 'open' })
    
    // Create container for React app
    const container = document.createElement('div')
    container.style.width = '100%'
    container.style.height = '100%'
    this._shadowRoot.appendChild(container)
  }

  connectedCallback() {
    this.render()
  }

  disconnectedCallback() {
    if (this.root) {
      this.root.unmount()
    }
  }

  attributeChangedCallback() {
    this.render()
  }

  private getConfig() {
    return {
      title: this.getAttribute('title') || undefined,
      theme: this.getAttribute('theme') || undefined,
      apiUrl: this.getAttribute('api-url') || undefined,
    }
  }

  private render() {
    const container = this._shadowRoot.querySelector('div')
    if (!container) return

    if (!this.root) {
      this.root = ReactDOM.createRoot(container)
    }

    this.root.render(
      <React.StrictMode>
        <ChatWidget config={this.getConfig()} />
      </React.StrictMode>
    )
  }
}

// Register the custom element
if (!customElements.get('definable-chat')) {
  customElements.define('definable-chat', DefinableChatElement)
}

export { DefinableChatElement }
