import AppKit
import ApplicationServices
import CoreGraphics
import Foundation

// Private SPI — available on all macOS versions we target
@_silgen_name("_AXUIElementGetWindow")
private func _AXUIElementGetWindow(_ element: AXUIElement, _ identifier: UnsafeMutablePointer<CGWindowID>) -> AXError

/// Window listing and management via CGWindowListCopyWindowInfo.
enum WindowManager {
  static func listWindows() -> WindowListResponse {
    let options: CGWindowListOption = [.excludeDesktopElements, .optionOnScreenOnly]
    guard let list = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] else {
      return WindowListResponse(windows: [])
    }

    var windows: [WindowInfoResponse] = []
    for info in list {
      guard let windowID = info[kCGWindowNumber as String] as? Int,
            let owner = info[kCGWindowOwnerName as String] as? String,
            let title = info[kCGWindowName as String] as? String,
            let boundsDict = info[kCGWindowBounds as String] as? [String: Double]
      else { continue }

      let x = boundsDict["X"] ?? 0
      let y = boundsDict["Y"] ?? 0
      let w = boundsDict["Width"] ?? 0
      let h = boundsDict["Height"] ?? 0

      // Skip very small or system windows
      if w < 10 || h < 10 { continue }

      windows.append(WindowInfoResponse(
        id: windowID,
        app: owner,
        title: title,
        bounds: BoundsResponse(x: x, y: y, width: w, height: h),
        minimized: false
      ))
    }
    return WindowListResponse(windows: windows)
  }

  static func focusWindow(id: Int?, title: String?) throws {
    let windows = listWindows().windows
    guard let target = windows.first(where: {
      (id != nil && $0.id == id!) || (title != nil && $0.title.contains(title!))
    }) else {
      throw BridgeError.notFound("Window not found: id=\(id.map(String.init) ?? "nil") title=\(title ?? "nil")")
    }
    // Activate the owning application first
    if let app = NSWorkspace.shared.runningApplications.first(where: { $0.localizedName == target.app }) {
      app.activate()
    }
  }

  static func resizeWindow(id: Int, x: Double, y: Double, width: Double, height: Double) throws {
    // Find owning app via window list
    let options: CGWindowListOption = [.excludeDesktopElements, .optionOnScreenOnly]
    guard let list = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]],
          let info = list.first(where: { ($0[kCGWindowNumber as String] as? Int) == id }),
          let owner = info[kCGWindowOwnerName as String] as? String,
          let pid = (NSWorkspace.shared.runningApplications.first { $0.localizedName == owner }?.processIdentifier)
    else {
      throw BridgeError.notFound("Window \(id) not found")
    }

    let appElement = AXUIElementCreateApplication(pid)
    var windowsValue: AnyObject?
    guard AXUIElementCopyAttributeValue(appElement, kAXWindowsAttribute as CFString, &windowsValue) == .success,
          let axWindows = windowsValue as? [AXUIElement],
          let axWindow = axWindows.first(where: { axWindowID($0) == id })
    else {
      throw BridgeError.notFound("AX window \(id) not found for app '\(owner)'")
    }

    var newPos = CGPoint(x: x, y: y)
    var newSize = CGSize(width: width, height: height)
    if let posVal = AXValueCreate(.cgPoint, &newPos) {
      AXUIElementSetAttributeValue(axWindow, kAXPositionAttribute as CFString, posVal)
    }
    if let sizeVal = AXValueCreate(.cgSize, &newSize) {
      AXUIElementSetAttributeValue(axWindow, kAXSizeAttribute as CFString, sizeVal)
    }
  }

  static func closeWindow(id: Int) throws {
    let options: CGWindowListOption = [.excludeDesktopElements, .optionOnScreenOnly]
    guard let list = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]],
          let info = list.first(where: { ($0[kCGWindowNumber as String] as? Int) == id }),
          let owner = info[kCGWindowOwnerName as String] as? String,
          let pid = (NSWorkspace.shared.runningApplications.first { $0.localizedName == owner }?.processIdentifier)
    else {
      throw BridgeError.notFound("Window \(id) not found")
    }

    let appElement = AXUIElementCreateApplication(pid)
    var windowsValue: AnyObject?
    guard AXUIElementCopyAttributeValue(appElement, kAXWindowsAttribute as CFString, &windowsValue) == .success,
          let axWindows = windowsValue as? [AXUIElement],
          let axWindow = axWindows.first(where: { axWindowID($0) == id })
    else {
      throw BridgeError.notFound("AX window \(id) not found for app '\(owner)'")
    }
    var buttonValue: AnyObject?
    if AXUIElementCopyAttributeValue(axWindow, kAXCloseButtonAttribute as CFString, &buttonValue) == .success,
       let closeButton = buttonValue
    {
      AXUIElementPerformAction(closeButton as! AXUIElement, kAXPressAction as CFString)
    }
  }

  // MARK: - Private helpers

  private static func axWindowID(_ element: AXUIElement) -> Int {
    var windowID: CGWindowID = 0
    _ = _AXUIElementGetWindow(element, &windowID)
    return Int(windowID)
  }
}
