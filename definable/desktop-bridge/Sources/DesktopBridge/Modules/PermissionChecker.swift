import ApplicationServices
import AVFoundation
import CoreGraphics
import Foundation

enum PermissionChecker {
  static func checkAccessibility() -> Bool {
    // AXIsProcessTrusted() can return false on macOS Sequoia even when
    // accessibility is actually functional (stale TCC cache for self-signed
    // CLI binaries). Fall back to a practical test: try to query the
    // system-wide element. .cannotComplete or .success means we have access;
    // .apiDisabled means we truly don't.
    if AXIsProcessTrusted() { return true }
    let systemWide = AXUIElementCreateSystemWide()
    var value: CFTypeRef?
    let result = AXUIElementCopyAttributeValue(systemWide, kAXFocusedApplicationAttribute as CFString, &value)
    // .apiDisabled = no permission. Anything else = permission granted.
    return result != .apiDisabled
  }

  static func checkScreenRecording() -> Bool {
    CGPreflightScreenCaptureAccess()
  }

  static func checkFullDiskAccess() -> Bool {
    FileManager.default.isReadableFile(atPath: "/Library/Application Support")
  }

  static func requestAccessibility() {
    let opts = [kAXTrustedCheckOptionPrompt.takeRetainedValue(): true] as NSDictionary
    AXIsProcessTrustedWithOptions(opts)
  }

  static func requestScreenRecording() {
    CGRequestScreenCaptureAccess()
  }
}
