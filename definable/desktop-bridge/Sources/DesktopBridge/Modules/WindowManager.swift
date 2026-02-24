import ApplicationServices
import AppKit
import CoreGraphics
import Foundation

enum WindowManager {
  struct WindowInfo {
    var id: Int
    var app: String
    var title: String
    var x: Double
    var y: Double
    var width: Double
    var height: Double
    var minimized: Bool
  }

  static func listWindows() -> [WindowInfo] {
    guard let windowList = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] else {
      return []
    }

    var results: [WindowInfo] = []
    for info in windowList {
      guard let id = info[kCGWindowNumber as String] as? Int,
            let ownerName = info[kCGWindowOwnerName as String] as? String,
            let bounds = info[kCGWindowBounds as String] as? [String: Double]
      else { continue }

      let layer = info[kCGWindowLayer as String] as? Int ?? 0
      guard layer == 0 else { continue }  // Only normal windows

      let title = info[kCGWindowName as String] as? String ?? ""
      let x = bounds["X"] ?? 0
      let y = bounds["Y"] ?? 0
      let w = bounds["Width"] ?? 0
      let h = bounds["Height"] ?? 0

      results.append(WindowInfo(
        id: id, app: ownerName, title: title,
        x: x, y: y, width: w, height: h,
        minimized: false))
    }
    return results
  }

  static func focusWindow(windowId: Int? = nil, title: String? = nil) -> Bool {
    if let title {
      return focusByTitle(title)
    }
    if let windowId {
      return focusById(windowId)
    }
    return false
  }

  static func resizeWindow(windowId: Int, x: Double?, y: Double?, width: Double?, height: Double?) -> Bool {
    guard let (_, axWindow) = findAXWindow(windowId: windowId) else { return false }

    if let x, let y {
      var point = CGPoint(x: x, y: y)
      if let value = AXValueCreate(.cgPoint, &point) {
        AXUIElementSetAttributeValue(axWindow, kAXPositionAttribute as CFString, value)
      }
    }

    if let width, let height {
      var size = CGSize(width: width, height: height)
      if let value = AXValueCreate(.cgSize, &size) {
        AXUIElementSetAttributeValue(axWindow, kAXSizeAttribute as CFString, value)
      }
    }

    return true
  }

  static func closeWindow(windowId: Int) -> Bool {
    guard let (_, axWindow) = findAXWindow(windowId: windowId) else { return false }

    var closeButton: CFTypeRef?
    AXUIElementCopyAttributeValue(axWindow, kAXCloseButtonAttribute as CFString, &closeButton)
    guard let button = closeButton else { return false }
    return AXUIElementPerformAction(button as! AXUIElement, kAXPressAction as CFString) == .success
  }

  // MARK: - Private

  private static func focusByTitle(_ title: String) -> Bool {
    let lowered = title.lowercased()
    for app in NSWorkspace.shared.runningApplications where app.activationPolicy == .regular {
      let axApp = AXUIElementCreateApplication(app.processIdentifier)
      var windowsRef: CFTypeRef?
      AXUIElementCopyAttributeValue(axApp, kAXWindowsAttribute as CFString, &windowsRef)
      guard let windows = windowsRef as? [AXUIElement] else { continue }

      for window in windows {
        var titleRef: CFTypeRef?
        AXUIElementCopyAttributeValue(window, kAXTitleAttribute as CFString, &titleRef)
        if let windowTitle = titleRef as? String, windowTitle.lowercased().contains(lowered) {
          app.activate()
          AXUIElementPerformAction(window, kAXRaiseAction as CFString)
          return true
        }
      }
    }
    return false
  }

  private static func focusById(_ windowId: Int) -> Bool {
    // Find which app owns this window via CGWindowList
    guard let windowList = CGWindowListCopyWindowInfo([.optionAll], kCGNullWindowID) as? [[String: Any]] else {
      return false
    }

    var ownerPID: Int32?
    for info in windowList {
      if let id = info[kCGWindowNumber as String] as? Int, id == windowId {
        ownerPID = info[kCGWindowOwnerPID as String] as? Int32
        break
      }
    }

    guard let pid = ownerPID else { return false }
    guard let app = NSWorkspace.shared.runningApplications.first(where: { $0.processIdentifier == pid }) else {
      return false
    }

    app.activate()

    // Try to raise the specific window via AX
    let axApp = AXUIElementCreateApplication(pid)
    var windowsRef: CFTypeRef?
    AXUIElementCopyAttributeValue(axApp, kAXWindowsAttribute as CFString, &windowsRef)
    if let windows = windowsRef as? [AXUIElement] {
      for window in windows {
        AXUIElementPerformAction(window, kAXRaiseAction as CFString)
        return true
      }
    }
    return true
  }

  private static func findAXWindow(windowId: Int) -> (NSRunningApplication, AXUIElement)? {
    guard let windowList = CGWindowListCopyWindowInfo([.optionAll], kCGNullWindowID) as? [[String: Any]] else {
      return nil
    }

    var ownerPID: Int32?
    for info in windowList {
      if let id = info[kCGWindowNumber as String] as? Int, id == windowId {
        ownerPID = info[kCGWindowOwnerPID as String] as? Int32
        break
      }
    }

    guard let pid = ownerPID else { return nil }
    guard let app = NSWorkspace.shared.runningApplications.first(where: { $0.processIdentifier == pid }) else {
      return nil
    }

    let axApp = AXUIElementCreateApplication(pid)
    var windowsRef: CFTypeRef?
    AXUIElementCopyAttributeValue(axApp, kAXWindowsAttribute as CFString, &windowsRef)
    guard let windows = windowsRef as? [AXUIElement] else { return nil }

    // Return first window (CGWindowID-to-AXUIElement mapping is platform-specific)
    if let first = windows.first {
      return (app, first)
    }
    return nil
  }
}
