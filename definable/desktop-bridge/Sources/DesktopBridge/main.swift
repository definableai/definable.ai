import AppKit
import Foundation
import Vapor

// MARK: - Token setup

let tokenDir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".definable")
try? FileManager.default.createDirectory(at: tokenDir, withIntermediateDirectories: true)
let tokenFile = tokenDir.appendingPathComponent("bridge-token")
let token: String

if let existing = try? String(contentsOf: tokenFile, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
   !existing.isEmpty
{
  token = existing
} else {
  token = UUID().uuidString
  try token.write(to: tokenFile, atomically: true, encoding: .utf8)
  // chmod 600
  try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: tokenFile.path)
}

// MARK: - Permission warnings

// Request accessibility permission on first run (triggers system dialog if not yet granted)
PermissionChecker.requestAccessibilityIfNeeded()
let perms = PermissionChecker.currentPermissions()

print("""
╔══════════════════════════════════════════════════════╗
║       Definable Desktop Bridge  v1.0.0               ║
╚══════════════════════════════════════════════════════╝
  URL:    http://127.0.0.1:7777
  Token:  \(token)

  Permissions:
    Accessibility:     \(perms.accessibility ? "✓" : "✗  → System Settings > Privacy > Accessibility")
    Screen Recording:  \(perms.screenRecording ? "✓" : "✗  → System Settings > Privacy > Screen Recording")
    Full Disk Access:  \(perms.fullDiskAccess ? "✓" : "✗  → System Settings > Privacy > Full Disk Access")

  Log:    ~/.definable/bridge.log
""")

if !perms.accessibility {
  print("⚠️  Accessibility not granted — input simulation and AX APIs will fail.")
}
if !perms.screenRecording {
  print("⚠️  Screen Recording not granted — screenshot and OCR will fail.")
}

// MARK: - Vapor app

var env = try Environment.detect()
let app = Application(env)
defer { app.shutdown() }

// Bind to localhost only
app.http.server.configuration.hostname = "127.0.0.1"
app.http.server.configuration.port = 7777

// Request logging to ~/.definable/bridge.log
let logFile = tokenDir.appendingPathComponent("bridge.log")
app.logger = Logger(label: "DesktopBridge") { _ in
  return MultiplexLogHandler([
    StreamLogHandler.standardOutput(label: "DesktopBridge"),
  ])
}

// Routes
try registerRoutes(app: app, token: token)

// Run
try app.run()
