import AppKit
import Foundation

/// Application lifecycle via NSWorkspace.
enum AppManager {
  static func listApps() -> AppListResponse {
    let running = NSWorkspace.shared.runningApplications
    let apps = running.map { app in
      AppInfoResponse(
        name: app.localizedName ?? app.bundleIdentifier ?? "Unknown",
        bundleId: app.bundleIdentifier ?? "",
        pid: app.processIdentifier,
        active: app.isActive
      )
    }
    return AppListResponse(apps: apps)
  }

  static func openApp(_ name: String) throws -> OpenAppResponse {
    // Try bundle ID first, then name, then path
    let workspace = NSWorkspace.shared

    if name.hasPrefix("/") {
      // Treat as path
      let url = URL(fileURLWithPath: name)
      let config = NSWorkspace.OpenConfiguration()
      var pid: pid_t = -1
      let sem = DispatchSemaphore(value: 0)
      workspace.openApplication(at: url, configuration: config) { app, err in
        pid = app?.processIdentifier ?? -1
        sem.signal()
      }
      sem.wait()
      if pid > 0 { return OpenAppResponse(pid: pid) }
      throw BridgeError.operationFailed("Failed to open app at path: \(name)")
    }

    // Try by bundle ID
    if name.contains("."), let url = workspace.urlForApplication(withBundleIdentifier: name) {
      let config = NSWorkspace.OpenConfiguration()
      var pid: pid_t = -1
      let sem = DispatchSemaphore(value: 0)
      workspace.openApplication(at: url, configuration: config) { app, err in
        pid = app?.processIdentifier ?? -1
        sem.signal()
      }
      sem.wait()
      if pid > 0 { return OpenAppResponse(pid: pid) }
    }

    // Try to find running app by name and activate it
    if let running = workspace.runningApplications.first(where: { $0.localizedName == name }) {
      running.activate(options: [.activateIgnoringOtherApps])
      return OpenAppResponse(pid: running.processIdentifier)
    }

    // Try NSWorkspace launch by name
    guard
      let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "")
        ?? findAppURL(name: name)
    else {
      throw BridgeError.notFound("App not found: '\(name)'")
    }

    let config = NSWorkspace.OpenConfiguration()
    var resultPid: pid_t = -1
    let sem = DispatchSemaphore(value: 0)
    workspace.openApplication(at: url, configuration: config) { app, err in
      resultPid = app?.processIdentifier ?? -1
      sem.signal()
    }
    sem.wait()

    if resultPid > 0 { return OpenAppResponse(pid: resultPid) }
    throw BridgeError.operationFailed("Failed to open '\(name)'")
  }

  static func quitApp(_ name: String, force: Bool) throws {
    guard let app = findRunning(name: name) else {
      throw BridgeError.notFound("App not running: '\(name)'")
    }
    if force {
      app.forceTerminate()
    } else {
      app.terminate()
    }
  }

  static func activateApp(_ name: String) throws {
    guard let app = findRunning(name: name) else {
      throw BridgeError.notFound("App not running: '\(name)'")
    }
    app.activate(options: [.activateIgnoringOtherApps])
  }

  static func openURL(_ urlString: String) throws {
    guard let url = URL(string: urlString) else {
      throw BridgeError.invalidInput("Invalid URL: \(urlString)")
    }
    NSWorkspace.shared.open(url)
  }

  static func openFile(_ path: String) {
    let url = URL(fileURLWithPath: path)
    NSWorkspace.shared.open(url)
  }

  // MARK: - Private helpers

  private static func findRunning(name: String) -> NSRunningApplication? {
    return NSWorkspace.shared.runningApplications.first {
      $0.localizedName == name || $0.bundleIdentifier == name
    }
  }

  private static func findAppURL(name: String) -> URL? {
    let paths = ["/Applications", "/System/Applications", "/Applications/Utilities"]
    for dir in paths {
      let url = URL(fileURLWithPath: dir).appendingPathComponent("\(name).app")
      if FileManager.default.fileExists(atPath: url.path) { return url }
    }
    return nil
  }
}
