import AppKit
import Foundation

enum AppManager {
  struct AppInfo {
    var name: String
    var bundleId: String?
    var pid: Int
    var active: Bool
  }

  static func listRunningApps() -> [AppInfo] {
    NSWorkspace.shared.runningApplications
      .filter { $0.activationPolicy == .regular }
      .map { app in
        AppInfo(
          name: app.localizedName ?? "Unknown",
          bundleId: app.bundleIdentifier,
          pid: Int(app.processIdentifier),
          active: app.isActive)
      }
  }

  static func openApp(name: String? = nil, bundleId: String? = nil, path: String? = nil) async throws -> Int {
    if let bundleId {
      if let running = NSWorkspace.shared.runningApplications.first(where: { $0.bundleIdentifier == bundleId }) {
        running.activate()
        return Int(running.processIdentifier)
      }
      if let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleId) {
        let config = NSWorkspace.OpenConfiguration()
        let app = try await NSWorkspace.shared.openApplication(at: url, configuration: config)
        return Int(app.processIdentifier)
      }
    }

    if let path {
      let url = URL(fileURLWithPath: path)
      let config = NSWorkspace.OpenConfiguration()
      let app = try await NSWorkspace.shared.openApplication(at: url, configuration: config)
      return Int(app.processIdentifier)
    }

    if let name {
      // Try to find by name in running apps first
      if let running = NSWorkspace.shared.runningApplications.first(where: {
        $0.localizedName?.lowercased() == name.lowercased()
      }) {
        running.activate()
        return Int(running.processIdentifier)
      }

      // Try common locations
      let paths = [
        "/Applications/\(name).app",
        "/System/Applications/\(name).app",
        "/Applications/Utilities/\(name).app",
        "/System/Applications/Utilities/\(name).app",
      ]

      for p in paths {
        let url = URL(fileURLWithPath: p)
        if FileManager.default.fileExists(atPath: p) {
          let config = NSWorkspace.OpenConfiguration()
          let app = try await NSWorkspace.shared.openApplication(at: url, configuration: config)
          return Int(app.processIdentifier)
        }
      }

      // Try open command as fallback
      let process = Process()
      process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
      process.arguments = ["-a", name]
      try process.run()
      process.waitUntilExit()

      // Find the newly launched app
      try? await Task.sleep(nanoseconds: 500_000_000)
      if let app = NSWorkspace.shared.runningApplications.first(where: {
        $0.localizedName?.lowercased() == name.lowercased()
      }) {
        return Int(app.processIdentifier)
      }
    }

    throw NSError(domain: "AppManager", code: 1, userInfo: [NSLocalizedDescriptionKey: "App not found"])
  }

  static func quitApp(name: String, force: Bool = false) -> Bool {
    guard let app = NSWorkspace.shared.runningApplications.first(where: {
      $0.localizedName?.lowercased() == name.lowercased()
        || $0.bundleIdentifier?.lowercased() == name.lowercased()
    }) else { return false }

    if force {
      return app.forceTerminate()
    } else {
      return app.terminate()
    }
  }

  static func activateApp(name: String) -> Bool {
    guard let app = NSWorkspace.shared.runningApplications.first(where: {
      $0.localizedName?.lowercased() == name.lowercased()
        || $0.bundleIdentifier?.lowercased() == name.lowercased()
    }) else { return false }

    return app.activate()
  }

  static func openURL(_ urlString: String) -> Bool {
    guard let url = URL(string: urlString) else { return false }
    return NSWorkspace.shared.open(url)
  }

  static func openFile(_ path: String) -> Bool {
    let url = URL(fileURLWithPath: path)
    return NSWorkspace.shared.open(url)
  }
}
