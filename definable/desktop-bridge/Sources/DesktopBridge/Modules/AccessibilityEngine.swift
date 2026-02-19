import AppKit
import ApplicationServices
import Foundation

/// AXUIElement-based accessibility inspection and interaction.
enum AccessibilityEngine {
  // MARK: - Focused element

  static func getFocusedElement() -> FocusedElementResponse {
    let systemElement = AXUIElementCreateSystemWide()
    var value: AnyObject?
    let result = AXUIElementCopyAttributeValue(systemElement, kAXFocusedUIElementAttribute as CFString, &value)
    guard result == .success, let axElem = value else {
      return FocusedElementResponse(found: false, element: nil)
    }
    // swiftlint:disable:next force_cast
    let element = parseElement(axElem as! AXUIElement, depth: 0, maxDepth: 0)
    return FocusedElementResponse(found: true, element: element)
  }

  // MARK: - UI tree

  static func getUITree(appName: String, depth: Int) throws -> [String: Any] {
    let appElement = try findAppElement(named: appName)
    return elementToDict(appElement, depth: depth, maxDepth: depth)
  }

  // MARK: - Find element

  static func findElement(appName: String, role: String?, title: String?) throws -> FindElementResponse {
    let appElement = try findAppElement(named: appName)
    guard let found = searchElement(appElement, role: role, title: title, depth: 8) else {
      return FindElementResponse(found: false, element: nil)
    }
    let parsed = parseElement(found, depth: 0, maxDepth: 0)
    return FindElementResponse(found: true, element: parsed)
  }

  // MARK: - Perform action

  static func performAction(appName: String, role: String?, title: String?, action: String) throws {
    let appElement = try findAppElement(named: appName)
    guard let found = searchElement(appElement, role: role, title: title, depth: 8) else {
      throw BridgeError.notFound("Element role='\(role ?? "")' title='\(title ?? "")' in '\(appName)'")
    }
    let result = AXUIElementPerformAction(found, action as CFString)
    if result != .success {
      throw BridgeError.operationFailed("AX action '\(action)' failed: \(result.rawValue)")
    }
  }

  // MARK: - Set value

  static func setValue(appName: String, role: String, title: String, value: String) throws {
    let appElement = try findAppElement(named: appName)
    guard let found = searchElement(appElement, role: role, title: title, depth: 8) else {
      throw BridgeError.notFound("Element role='\(role)' title='\(title)' in '\(appName)'")
    }
    let result = AXUIElementSetAttributeValue(found, kAXValueAttribute as CFString, value as AnyObject)
    if result != .success {
      throw BridgeError.operationFailed("AXSetValue failed: \(result.rawValue)")
    }
  }

  // MARK: - Private helpers

  private static func findAppElement(named name: String) throws -> AXUIElement {
    let workspace = NSWorkspace.shared
    guard let app = workspace.runningApplications.first(where: { $0.localizedName == name || $0.bundleIdentifier == name }) else {
      throw BridgeError.notFound("App '\(name)' is not running")
    }
    return AXUIElementCreateApplication(app.processIdentifier)
  }

  private static func parseElement(_ element: AXUIElement, depth: Int, maxDepth: Int) -> UIElementResponse {
    let role = stringAttribute(element, kAXRoleAttribute) ?? ""
    let title = stringAttribute(element, kAXTitleAttribute) ?? stringAttribute(element, kAXDescriptionAttribute) ?? ""
    let value = stringAttribute(element, kAXValueAttribute) ?? ""
    let bounds = boundsAttribute(element)

    var children: [UIElementResponse]? = nil
    if depth < maxDepth {
      var childrenValue: AnyObject?
      if AXUIElementCopyAttributeValue(element, kAXChildrenAttribute as CFString, &childrenValue) == .success,
         let childArray = childrenValue as? [AXUIElement]
      {
        children = childArray.map { parseElement($0, depth: depth + 1, maxDepth: maxDepth) }
      }
    }

    return UIElementResponse(role: role, title: title, value: value, bounds: bounds, children: children)
  }

  private static func elementToDict(_ element: AXUIElement, depth: Int, maxDepth: Int) -> [String: Any] {
    var dict: [String: Any] = [
      "role": stringAttribute(element, kAXRoleAttribute) ?? "",
      "title": stringAttribute(element, kAXTitleAttribute) ?? stringAttribute(element, kAXDescriptionAttribute) ?? "",
      "value": stringAttribute(element, kAXValueAttribute) ?? "",
    ]
    let b = boundsAttribute(element)
    dict["bounds"] = ["x": b.x, "y": b.y, "width": b.width, "height": b.height]

    if depth > 0 {
      var childrenValue: AnyObject?
      if AXUIElementCopyAttributeValue(element, kAXChildrenAttribute as CFString, &childrenValue) == .success,
         let childArray = childrenValue as? [AXUIElement]
      {
        dict["children"] = childArray.map { elementToDict($0, depth: depth - 1, maxDepth: maxDepth) }
      }
    }
    return dict
  }

  private static func searchElement(_ element: AXUIElement, role: String?, title: String?, depth: Int) -> AXUIElement? {
    let elemRole = stringAttribute(element, kAXRoleAttribute) ?? ""
    let elemTitle = stringAttribute(element, kAXTitleAttribute) ?? stringAttribute(element, kAXDescriptionAttribute) ?? ""

    let roleMatch = role == nil || role!.isEmpty || elemRole == role
    let titleMatch = title == nil || title!.isEmpty || elemTitle.contains(title!)
    if roleMatch && titleMatch && (role != nil || title != nil) {
      return element
    }

    if depth > 0 {
      var childrenValue: AnyObject?
      if AXUIElementCopyAttributeValue(element, kAXChildrenAttribute as CFString, &childrenValue) == .success,
         let childArray = childrenValue as? [AXUIElement]
      {
        for child in childArray {
          if let found = searchElement(child, role: role, title: title, depth: depth - 1) {
            return found
          }
        }
      }
    }
    return nil
  }

  private static func stringAttribute(_ element: AXUIElement, _ attribute: String) -> String? {
    var value: AnyObject?
    guard AXUIElementCopyAttributeValue(element, attribute as CFString, &value) == .success else { return nil }
    return value as? String
  }

  private static func boundsAttribute(_ element: AXUIElement) -> BoundsResponse {
    var posValue: AnyObject?
    var sizeValue: AnyObject?
    var position = CGPoint.zero
    var size = CGSize.zero
    if AXUIElementCopyAttributeValue(element, kAXPositionAttribute as CFString, &posValue) == .success,
       let axPos = posValue
    {
      AXValueGetValue(axPos as! AXValue, AXValueType.cgPoint, &position)
    }
    if AXUIElementCopyAttributeValue(element, kAXSizeAttribute as CFString, &sizeValue) == .success,
       let axSz = sizeValue
    {
      AXValueGetValue(axSz as! AXValue, AXValueType.cgSize, &size)
    }
    return BoundsResponse(x: Double(position.x), y: Double(position.y), width: Double(size.width), height: Double(size.height))
  }
}
