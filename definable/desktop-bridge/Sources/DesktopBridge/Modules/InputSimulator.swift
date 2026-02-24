import ApplicationServices
import CoreGraphics
import Foundation

enum InputSimulator {
  // MARK: - Mouse

  static func click(x: Double, y: Double, button: String = "left", clicks: Int = 1, modifiers: [String] = []) {
    let point = CGPoint(x: x, y: y)
    let (downType, upType, cgButton) = mouseButtonTypes(button)
    let flags = modifierFlags(modifiers)

    for _ in 0..<max(1, clicks) {
      if let down = CGEvent(mouseEventSource: nil, mouseType: downType, mouseCursorPosition: point, mouseButton: cgButton) {
        down.flags = flags
        down.post(tap: .cghidEventTap)
      }
      usleep(20_000)  // 20ms
      if let up = CGEvent(mouseEventSource: nil, mouseType: upType, mouseCursorPosition: point, mouseButton: cgButton) {
        up.flags = flags
        up.post(tap: .cghidEventTap)
      }
      usleep(50_000)  // 50ms between clicks
    }
  }

  static func mouseMove(x: Double, y: Double) {
    let point = CGPoint(x: x, y: y)
    if let event = CGEvent(mouseEventSource: nil, mouseType: .mouseMoved, mouseCursorPosition: point, mouseButton: .left) {
      event.post(tap: .cghidEventTap)
    }
  }

  static func scroll(x: Double, y: Double, dx: Double = 0, dy: Double = -3) {
    // Move to position first
    mouseMove(x: x, y: y)
    usleep(20_000)

    if let event = CGEvent(scrollWheelEvent2Source: nil, units: .line, wheelCount: 2, wheel1: Int32(dy), wheel2: Int32(dx), wheel3: 0) {
      event.post(tap: .cghidEventTap)
    }
  }

  static func drag(fromX: Double, fromY: Double, toX: Double, toY: Double, duration: Double = 0.5) {
    let from = CGPoint(x: fromX, y: fromY)
    let to = CGPoint(x: toX, y: toY)
    let steps = max(10, Int(duration * 60))  // ~60fps

    // Mouse down at start
    if let down = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: from, mouseButton: .left) {
      down.post(tap: .cghidEventTap)
    }
    usleep(30_000)

    // Smooth interpolation
    for i in 1...steps {
      let t = Double(i) / Double(steps)
      let x = from.x + (to.x - from.x) * t
      let y = from.y + (to.y - from.y) * t
      let point = CGPoint(x: x, y: y)
      if let drag = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDragged, mouseCursorPosition: point, mouseButton: .left) {
        drag.post(tap: .cghidEventTap)
      }
      usleep(UInt32(duration * 1_000_000 / Double(steps)))
    }

    // Mouse up at end
    if let up = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: to, mouseButton: .left) {
      up.post(tap: .cghidEventTap)
    }
  }

  // MARK: - Keyboard

  static func typeText(_ text: String) {
    for char in text {
      let str = String(char) as NSString
      let unichars = [UniChar](unsafeUninitializedCapacity: str.length) { buf, count in
        str.getCharacters(buf.baseAddress!, range: NSRange(location: 0, length: str.length))
        count = str.length
      }

      if let down = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: true) {
        down.keyboardSetUnicodeString(stringLength: unichars.count, unicodeString: unichars)
        down.post(tap: .cghidEventTap)
      }
      if let up = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: false) {
        up.keyboardSetUnicodeString(stringLength: unichars.count, unicodeString: unichars)
        up.post(tap: .cghidEventTap)
      }
      usleep(10_000)  // 10ms between chars
    }
  }

  static func pressKey(_ key: String, modifiers: [String] = []) {
    guard let keyCode = keyCodeFor(key) else {
      // Try Unicode fallback
      typeText(key)
      return
    }
    let flags = modifierFlags(modifiers)

    if let down = CGEvent(keyboardEventSource: nil, virtualKey: CGKeyCode(keyCode), keyDown: true) {
      down.flags = flags
      down.post(tap: .cghidEventTap)
    }
    usleep(20_000)
    if let up = CGEvent(keyboardEventSource: nil, virtualKey: CGKeyCode(keyCode), keyDown: false) {
      up.flags = flags
      up.post(tap: .cghidEventTap)
    }
  }

  // MARK: - Helpers

  private static func mouseButtonTypes(_ button: String) -> (CGEventType, CGEventType, CGMouseButton) {
    switch button.lowercased() {
    case "right":
      return (.rightMouseDown, .rightMouseUp, .right)
    case "middle":
      return (.otherMouseDown, .otherMouseUp, .center)
    default:
      return (.leftMouseDown, .leftMouseUp, .left)
    }
  }

  private static func modifierFlags(_ modifiers: [String]) -> CGEventFlags {
    var flags = CGEventFlags()
    for mod in modifiers {
      switch mod.lowercased() {
      case "cmd", "command": flags.insert(.maskCommand)
      case "shift": flags.insert(.maskShift)
      case "alt", "option": flags.insert(.maskAlternate)
      case "ctrl", "control": flags.insert(.maskControl)
      case "fn": flags.insert(.maskSecondaryFn)
      default: break
      }
    }
    return flags
  }

  private static func keyCodeFor(_ key: String) -> Int? {
    let map: [String: Int] = [
      "return": 36, "enter": 36, "tab": 48, "space": 49,
      "delete": 51, "backspace": 51, "escape": 53, "esc": 53,
      "command": 55, "shift": 56, "capslock": 57, "option": 58,
      "control": 59, "fn": 63,
      "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96,
      "f6": 97, "f7": 98, "f8": 100, "f9": 101, "f10": 109,
      "f11": 103, "f12": 111,
      "up": 126, "down": 125, "left": 123, "right": 124,
      "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
      "forwarddelete": 117,
      "a": 0, "b": 11, "c": 8, "d": 2, "e": 14, "f": 3, "g": 5,
      "h": 4, "i": 34, "j": 38, "k": 40, "l": 37, "m": 46,
      "n": 45, "o": 31, "p": 35, "q": 12, "r": 15, "s": 1,
      "t": 17, "u": 32, "v": 9, "w": 13, "x": 7, "y": 16, "z": 6,
      "0": 29, "1": 18, "2": 19, "3": 20, "4": 21, "5": 23,
      "6": 22, "7": 26, "8": 28, "9": 25,
      "-": 27, "=": 24, "[": 33, "]": 30, "\\": 42,
      ";": 41, "'": 39, ",": 43, ".": 47, "/": 44, "`": 50,
    ]
    return map[key.lowercased()]
  }
}
