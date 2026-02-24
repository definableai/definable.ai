import Foundation
import Vapor

@main
struct DesktopBridgeApp {
  static func main() async throws {
    let configDir = FileManager.default.homeDirectoryForCurrentUser
      .appendingPathComponent(".definable")
    try FileManager.default.createDirectory(at: configDir, withIntermediateDirectories: true)

    let tokenPath = configDir.appendingPathComponent("bridge-token").path
    let token: String
    if let existing = try? String(contentsOfFile: tokenPath, encoding: .utf8)
      .trimmingCharacters(in: .whitespacesAndNewlines), !existing.isEmpty
    {
      token = existing
    } else {
      token = UUID().uuidString
      try token.write(toFile: tokenPath, atomically: true, encoding: .utf8)
      chmod(tokenPath, 0o600)
    }

    // Check permissions
    let accessibility = PermissionChecker.checkAccessibility()
    let screenRecording = PermissionChecker.checkScreenRecording()

    print("╔══════════════════════════════════════════╗")
    print("║        Definable Desktop Bridge          ║")
    print("╠══════════════════════════════════════════╣")
    print("║  Token: \(String(token.prefix(8)))...                        ║")
    print("║  Accessibility: \(accessibility ? "✓" : "✗")                        ║")
    print("║  Screen Recording: \(screenRecording ? "✓" : "✗")                     ║")
    print("╚══════════════════════════════════════════╝")
    print("")
    print("Listening on http://127.0.0.1:7777")

    let binaryPath = ProcessInfo.processInfo.arguments.first ?? "~/.definable/bin/desktop-bridge"

    if !accessibility || !screenRecording {
      print("⚠️  Missing permissions detected.")
      print("   Binary: \(binaryPath)")
      print("")
      print("   macOS ties permissions to the binary's code signature (cdhash).")
      print("   After a rebuild, old permission entries become stale even if")
      print("   they still appear toggled ON in System Settings.")
      print("")
      print("   To fix: open System Settings → Privacy & Security, then for")
      print("   each permission below:")
      print("     1. REMOVE the existing entry (click −), not just toggle off")
      print("     2. Click + and browse to: \(binaryPath)")
      print("     3. Toggle ON, then restart the bridge")
    }
    if !accessibility {
      print("")
      print("   ✗ Accessibility — needed for input simulation and UI inspection")
    }
    if !screenRecording {
      print("")
      print("   ✗ Screen Recording — needed for screenshots and OCR")
    }

    let env = try Environment.detect()
    let app = try await Application.make(env)
    app.http.server.configuration.hostname = "127.0.0.1"
    app.http.server.configuration.port = 7777

    // Auth middleware
    let authMiddleware = BearerAuthMiddleware(validToken: token)
    let protected = app.grouped(authMiddleware)

    // Register routes
    registerHealthRoutes(protected)
    registerScreenRoutes(protected)
    registerInputRoutes(protected)
    registerAppRoutes(protected)
    registerWindowRoutes(protected)
    registerAccessibilityRoutes(protected)
    registerAppleScriptRoutes(protected)
    registerFileRoutes(protected)
    registerClipboardRoutes(protected)
    registerSystemRoutes(protected)
    registerShellRoutes(protected)
    registerCameraRoutes(protected)
    registerScreenRecordRoutes(protected)
    registerNotificationRoutes(protected)

    try await app.execute()
  }
}
