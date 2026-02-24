import ApplicationServices
import AppKit
import Foundation

enum AccessibilityEngine {
  struct UIElement {
    var role: String?
    var title: String?
    var value: String?
    var x: Double?
    var y: Double?
    var width: Double?
    var height: Double?
    var children: [UIElement]?
  }

  static func getFocusedElement() -> UIElement? {
    let sysWide = AXUIElementCreateSystemWide()
    var focusedRef: CFTypeRef?
    let err = AXUIElementCopyAttributeValue(sysWide, kAXFocusedUIElementAttribute as CFString, &focusedRef)
    guard err == .success, let focused = focusedRef else { return nil }
    return parseElement(focused as! AXUIElement, depth: 0, maxDepth: 0)
  }

  static func getUITree(appName: String, depth: Int = 3) -> UIElement? {
    guard let app = findRunningApp(name: appName) else { return nil }
    let axApp = AXUIElementCreateApplication(app.processIdentifier)
    return parseElement(axApp, depth: 0, maxDepth: depth)
  }

  static func findElement(appName: String, role: String? = nil, title: String? = nil) -> UIElement? {
    guard let app = findRunningApp(name: appName) else { return nil }
    let axApp = AXUIElementCreateApplication(app.processIdentifier)
    return searchElement(axApp, role: role, title: title, depth: 0, maxDepth: 10)
  }

  static func performAction(appName: String, role: String? = nil, title: String? = nil, action: String = "AXPress") -> Bool {
    guard let app = findRunningApp(name: appName) else { return false }
    let axApp = AXUIElementCreateApplication(app.processIdentifier)
    guard let target = searchAXElement(axApp, role: role, title: title, depth: 0, maxDepth: 10) else { return false }
    let err = AXUIElementPerformAction(target, action as CFString)
    return err == .success
  }

  static func setValue(appName: String, role: String? = nil, title: String? = nil, value: String) -> Bool {
    guard let app = findRunningApp(name: appName) else { return false }
    let axApp = AXUIElementCreateApplication(app.processIdentifier)
    guard let target = searchAXElement(axApp, role: role, title: title, depth: 0, maxDepth: 10) else { return false }

    let cfValue = value as CFTypeRef
    let err = AXUIElementSetAttributeValue(target, kAXValueAttribute as CFString, cfValue)
    return err == .success
  }

  // MARK: - Private

  private static func parseElement(_ element: AXUIElement, depth: Int, maxDepth: Int) -> UIElement {
    let role = attribute(element, kAXRoleAttribute) as? String
    let title = attribute(element, kAXTitleAttribute) as? String
    let valueAttr = attribute(element, kAXValueAttribute)
    let value: String? = {
      if let s = valueAttr as? String { return s }
      if let n = valueAttr as? NSNumber { return n.stringValue }
      return nil
    }()

    var x: Double?
    var y: Double?
    var width: Double?
    var height: Double?

    if let posValue = attribute(element, kAXPositionAttribute) {
      var point = CGPoint.zero
      if AXValueGetValue(posValue as! AXValue, .cgPoint, &point) {
        x = Double(point.x)
        y = Double(point.y)
      }
    }

    if let sizeValue = attribute(element, kAXSizeAttribute) {
      var size = CGSize.zero
      if AXValueGetValue(sizeValue as! AXValue, .cgSize, &size) {
        width = Double(size.width)
        height = Double(size.height)
      }
    }

    var children: [UIElement]?
    if depth < maxDepth, let axChildren = attribute(element, kAXChildrenAttribute) as? [AXUIElement] {
      children = axChildren.prefix(100).map { child in
        parseElement(child, depth: depth + 1, maxDepth: maxDepth)
      }
    }

    return UIElement(role: role, title: title, value: value, x: x, y: y, width: width, height: height, children: children)
  }

  private static func searchElement(_ element: AXUIElement, role: String?, title: String?, depth: Int, maxDepth: Int) -> UIElement? {
    let axElement = searchAXElement(element, role: role, title: title, depth: depth, maxDepth: maxDepth)
    guard let found = axElement else { return nil }
    return parseElement(found, depth: 0, maxDepth: 0)
  }

  private static func searchAXElement(_ element: AXUIElement, role: String?, title: String?, depth: Int, maxDepth: Int) -> AXUIElement? {
    let currentRole = attribute(element, kAXRoleAttribute) as? String
    let currentTitle = attribute(element, kAXTitleAttribute) as? String

    let roleMatch = role == nil || currentRole?.lowercased() == role?.lowercased()
    let titleMatch = title == nil || (currentTitle?.lowercased().contains(title?.lowercased() ?? "") ?? false)

    if roleMatch, titleMatch, (role != nil || title != nil) {
      return element
    }

    if depth >= maxDepth { return nil }

    if let children = attribute(element, kAXChildrenAttribute) as? [AXUIElement] {
      for child in children {
        if let found = searchAXElement(child, role: role, title: title, depth: depth + 1, maxDepth: maxDepth) {
          return found
        }
      }
    }

    return nil
  }

  private static func attribute(_ element: AXUIElement, _ name: String) -> CFTypeRef? {
    var value: CFTypeRef?
    AXUIElementCopyAttributeValue(element, name as CFString, &value)
    return value
  }

  private static func findRunningApp(name: String) -> NSRunningApplication? {
    let workspace = NSWorkspace.shared
    let lowered = name.lowercased()
    return workspace.runningApplications.first { app in
      app.localizedName?.lowercased() == lowered
        || app.bundleIdentifier?.lowercased() == lowered
    }
  }
}
