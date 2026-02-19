import CoreGraphics
import Foundation

/// Simulates keyboard and mouse input via CGEvent.
enum InputSimulator {
  // MARK: - Mouse

  static func click(x: Double, y: Double, button: String = "left", clicks: Int = 1) throws {
    let point = CGPoint(x: x, y: y)
    let (downType, upType) = mouseEventTypes(for: button)

    for _ in 0..<clicks {
      guard
        let down = CGEvent(mouseEventSource: nil, mouseType: downType, mouseCursorPosition: point, mouseButton: cgMouseButton(for: button)),
        let up = CGEvent(mouseEventSource: nil, mouseType: upType, mouseCursorPosition: point, mouseButton: cgMouseButton(for: button))
      else {
        throw BridgeError.operationFailed("Failed to create mouse event")
      }
      down.post(tap: .cghidEventTap)
      Thread.sleep(forTimeInterval: 0.05)
      up.post(tap: .cghidEventTap)
      Thread.sleep(forTimeInterval: 0.05)
    }
  }

  static func mouseMove(x: Double, y: Double) throws {
    let point = CGPoint(x: x, y: y)
    guard let event = CGEvent(mouseEventSource: nil, mouseType: .mouseMoved, mouseCursorPosition: point, mouseButton: .left) else {
      throw BridgeError.operationFailed("Failed to create mouse move event")
    }
    event.post(tap: .cghidEventTap)
  }

  static func scroll(x: Double, y: Double, dx: Double, dy: Double) throws {
    // Move mouse to scroll position first
    let point = CGPoint(x: x, y: y)
    guard let moveEvent = CGEvent(mouseEventSource: nil, mouseType: .mouseMoved, mouseCursorPosition: point, mouseButton: .left) else {
      throw BridgeError.operationFailed("Failed to move mouse for scroll")
    }
    moveEvent.post(tap: .cghidEventTap)

    // Scroll: units in "scroll wheel clicks"
    let scrollEvent = CGEvent(scrollWheelEvent2Source: nil, units: .line, wheelCount: 2, wheel1: Int32(dy), wheel2: Int32(dx), wheel3: 0)
    guard let event = scrollEvent else {
      throw BridgeError.operationFailed("Failed to create scroll event")
    }
    event.post(tap: .cghidEventTap)
  }

  static func drag(fromX: Double, fromY: Double, toX: Double, toY: Double, duration: Double = 0.5) throws {
    let from = CGPoint(x: fromX, y: fromY)
    let to = CGPoint(x: toX, y: toY)

    guard let down = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: from, mouseButton: .left) else {
      throw BridgeError.operationFailed("Failed to create drag start event")
    }
    down.post(tap: .cghidEventTap)

    // Interpolate movement
    let steps = max(Int(duration * 60), 10)
    for i in 1...steps {
      let t = Double(i) / Double(steps)
      let ix = fromX + (toX - fromX) * t
      let iy = fromY + (toY - fromY) * t
      let ipoint = CGPoint(x: ix, y: iy)
      if let drag = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDragged, mouseCursorPosition: ipoint, mouseButton: .left) {
        drag.post(tap: .cghidEventTap)
      }
      Thread.sleep(forTimeInterval: duration / Double(steps))
    }

    guard let up = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: to, mouseButton: .left) else {
      throw BridgeError.operationFailed("Failed to create drag end event")
    }
    up.post(tap: .cghidEventTap)
  }

  // MARK: - Keyboard

  static func typeText(_ text: String) throws {
    let source = CGEventSource(stateID: .hidSystemState)
    for scalar in text.unicodeScalars {
      let char = scalar.value
      // Try keyCode path for ASCII; fall back to Unicode event
      if let keyCode = asciiKeyCode(char) {
        let needsShift = asciiNeedsShift(char)
        try pressVirtualKey(keyCode, modifiers: needsShift ? [.maskShift] : [], source: source)
      } else {
        // Post a Unicode character event
        if let down = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: true) {
          var c = UniChar(scalar.value & 0xFFFF)
          down.keyboardSetUnicodeString(stringLength: 1, unicodeString: &c)
          down.post(tap: .cghidEventTap)
        }
        if let up = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: false) {
          var c = UniChar(scalar.value & 0xFFFF)
          up.keyboardSetUnicodeString(stringLength: 1, unicodeString: &c)
          up.post(tap: .cghidEventTap)
        }
      }
      Thread.sleep(forTimeInterval: 0.02)
    }
  }

  static func pressKey(_ key: String, modifiers: [String] = []) throws {
    let keyCode = virtualKeyCode(for: key)
    let cgModifiers = cgEventFlags(for: modifiers)
    try pressVirtualKey(keyCode, modifiers: cgModifiers, source: nil)
  }

  // MARK: - Private helpers

  private static func pressVirtualKey(_ keyCode: CGKeyCode, modifiers: CGEventFlags, source: CGEventSource?) throws {
    guard
      let down = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: true),
      let up = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: false)
    else {
      throw BridgeError.operationFailed("Failed to create key event for code \(keyCode)")
    }
    if !modifiers.isEmpty {
      down.flags = modifiers
      up.flags = modifiers
    }
    down.post(tap: .cghidEventTap)
    Thread.sleep(forTimeInterval: 0.02)
    up.post(tap: .cghidEventTap)
  }

  private static func mouseEventTypes(for button: String) -> (CGEventType, CGEventType) {
    switch button.lowercased() {
    case "right": return (.rightMouseDown, .rightMouseUp)
    case "middle": return (.otherMouseDown, .otherMouseUp)
    default: return (.leftMouseDown, .leftMouseUp)
    }
  }

  private static func cgMouseButton(for button: String) -> CGMouseButton {
    switch button.lowercased() {
    case "right": return .right
    case "middle": return .center
    default: return .left
    }
  }

  private static func cgEventFlags(for modifiers: [String]) -> CGEventFlags {
    var flags: CGEventFlags = []
    for mod in modifiers {
      switch mod.lowercased() {
      case "cmd", "command": flags.insert(.maskCommand)
      case "shift": flags.insert(.maskShift)
      case "opt", "option", "alt": flags.insert(.maskAlternate)
      case "ctrl", "control": flags.insert(.maskControl)
      default: break
      }
    }
    return flags
  }

  // swiftlint:disable cyclomatic_complexity
  private static func virtualKeyCode(for key: String) -> CGKeyCode {
    switch key.lowercased() {
    case "return", "enter": return 36
    case "tab": return 48
    case "space": return 49
    case "delete", "backspace": return 51
    case "escape", "esc": return 53
    case "left": return 123
    case "right": return 124
    case "down": return 125
    case "up": return 126
    case "f1": return 122; case "f2": return 120; case "f3": return 99
    case "f4": return 118; case "f5": return 96; case "f6": return 97
    case "f7": return 98; case "f8": return 100; case "f9": return 101
    case "f10": return 109; case "f11": return 103; case "f12": return 111
    case "home": return 115; case "end": return 119
    case "pageup": return 116; case "pagedown": return 121
    case "a": return 0; case "s": return 1; case "d": return 2
    case "f": return 3; case "h": return 4; case "g": return 5
    case "z": return 6; case "x": return 7; case "c": return 8
    case "v": return 9; case "b": return 11; case "q": return 12
    case "w": return 13; case "e": return 14; case "r": return 15
    case "y": return 16; case "t": return 17; case "1": return 18
    case "2": return 19; case "3": return 20; case "4": return 21
    case "6": return 22; case "5": return 23; case "=": return 24
    case "9": return 25; case "7": return 26; case "-": return 27
    case "8": return 28; case "0": return 29; case "]": return 30
    case "o": return 31; case "u": return 32; case "[": return 33
    case "i": return 34; case "p": return 35; case "l": return 37
    case "j": return 38; case "'": return 39; case "k": return 40
    case ";": return 41; case "\\": return 42; case ",": return 43
    case "/": return 44; case "n": return 45; case "m": return 46
    case ".": return 47
    default: return 49  // space as fallback
    }
  }
  // swiftlint:enable cyclomatic_complexity

  private static func asciiKeyCode(_ char: UInt32) -> CGKeyCode? {
    let ascii = char & 0x7F
    if ascii >= 65 && ascii <= 90 { return virtualKeyCode(for: String(UnicodeScalar(ascii + 32)!)) }
    if ascii >= 97 && ascii <= 122 { return virtualKeyCode(for: String(UnicodeScalar(ascii)!)) }
    if ascii >= 48 && ascii <= 57 { return virtualKeyCode(for: String(UnicodeScalar(ascii)!)) }
    switch ascii {
    case 13: return 36  // return
    case 9: return 48   // tab
    case 32: return 49  // space
    case 27: return 53  // escape
    default: return nil
    }
  }

  private static func asciiNeedsShift(_ char: UInt32) -> Bool {
    let ascii = char & 0x7F
    return ascii >= 65 && ascii <= 90  // uppercase
  }
}
