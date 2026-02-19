import ApplicationServices
import Foundation

/// Checks macOS privacy permission status.
enum PermissionChecker {
  static func checkAccessibility() -> Bool {
    // AXIsProcessTrustedWithOptions with prompt=true properly registers unsigned
    // binaries in the TCC database and triggers the system permission dialog when needed.
    let options = [kAXTrustedCheckOptionPrompt.takeRetainedValue(): false] as CFDictionary
    return AXIsProcessTrustedWithOptions(options)
  }

  /// Call once at startup to trigger the macOS permission dialog if not yet granted.
  static func requestAccessibilityIfNeeded() {
    let options = [kAXTrustedCheckOptionPrompt.takeRetainedValue(): true] as CFDictionary
    _ = AXIsProcessTrustedWithOptions(options)
  }

  static func checkScreenRecording() -> Bool {
    return CGPreflightScreenCaptureAccess()
  }

  static func checkFullDiskAccess() -> Bool {
    let testPath = "/Library/Application Support"
    return FileManager.default.isReadableFile(atPath: testPath)
  }

  static func currentPermissions() -> PermissionsResponse {
    return PermissionsResponse(
      accessibility: checkAccessibility(),
      screenRecording: checkScreenRecording(),
      fullDiskAccess: checkFullDiskAccess()
    )
  }
}
