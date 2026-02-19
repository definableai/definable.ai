import Foundation
import Vapor

// MARK: - Route registration

func registerRoutes(app: Application, token: String) throws {
  // All routes require bearer auth
  let protected = app.grouped(BearerAuthMiddleware(expectedToken: token))

  // MARK: Health

  protected.post("health") { req async throws -> HealthResponse in
    let perms = PermissionChecker.currentPermissions()
    return HealthResponse(status: "ok", version: "1.0.0", permissions: perms)
  }

  // MARK: Screen

  protected.post("screen", "capture") { req async throws -> CaptureResponse in
    let body = try? req.content.decode(ScreenCaptureRequest.self)
    let jpeg = try ScreenCapture.captureDisplay(body?.display ?? 0, region: body?.cgRect, maxWidth: body?.max_width ?? 512)
    return CaptureResponse(image: jpeg.base64EncodedString(), format: "jpeg")
  }

  protected.post("screen", "ocr") { req async throws -> OCRResponse in
    let body = try? req.content.decode(ScreenOCRRequest.self)
    return try ScreenCapture.ocrDisplay(0, region: body?.cgRect)
  }

  protected.post("screen", "find_text") { req async throws -> FindTextResponse in
    let body = try req.content.decode(FindTextRequest.self)
    return try ScreenCapture.findText(body.text, nth: body.nth ?? 0)
  }

  // MARK: Input

  protected.post("input", "click") { req async throws -> OKResponse in
    let body = try req.content.decode(ClickRequest.self)
    try await Task.detached(priority: .userInitiated) {
      try InputSimulator.click(x: body.x, y: body.y, button: body.button ?? "left", clicks: body.clicks ?? 1)
    }.value
    return OKResponse()
  }

  protected.post("input", "type") { req async throws -> OKResponse in
    let body = try req.content.decode(TypeRequest.self)
    try await Task.detached(priority: .userInitiated) {
      try InputSimulator.typeText(body.text)
    }.value
    return OKResponse()
  }

  protected.post("input", "key") { req async throws -> OKResponse in
    let body = try req.content.decode(KeyRequest.self)
    try await Task.detached(priority: .userInitiated) {
      try InputSimulator.pressKey(body.key, modifiers: body.modifiers ?? [])
    }.value
    return OKResponse()
  }

  protected.post("input", "mouse_move") { req async throws -> OKResponse in
    let body = try req.content.decode(MouseMoveRequest.self)
    try await Task.detached(priority: .userInitiated) {
      try InputSimulator.mouseMove(x: body.x, y: body.y)
    }.value
    return OKResponse()
  }

  protected.post("input", "scroll") { req async throws -> OKResponse in
    let body = try req.content.decode(ScrollRequest.self)
    try await Task.detached(priority: .userInitiated) {
      try InputSimulator.scroll(x: body.x, y: body.y, dx: body.dx ?? 0, dy: body.dy ?? -3)
    }.value
    return OKResponse()
  }

  protected.post("input", "drag") { req async throws -> OKResponse in
    let body = try req.content.decode(DragRequest.self)
    try await Task.detached(priority: .userInitiated) {
      try InputSimulator.drag(fromX: body.from_x, fromY: body.from_y, toX: body.to_x, toY: body.to_y, duration: body.duration ?? 0.5)
    }.value
    return OKResponse()
  }

  // MARK: Apps

  protected.post("apps", "list") { req async throws -> AppListResponse in
    return AppManager.listApps()
  }

  protected.post("apps", "open") { req async throws -> OpenAppResponse in
    let body = try req.content.decode(AppNameRequest.self)
    return try AppManager.openApp(body.name)
  }

  protected.post("apps", "quit") { req async throws -> OKResponse in
    let body = try req.content.decode(QuitAppRequest.self)
    try AppManager.quitApp(body.name, force: body.force ?? false)
    return OKResponse()
  }

  protected.post("apps", "activate") { req async throws -> OKResponse in
    let body = try req.content.decode(AppNameRequest.self)
    try AppManager.activateApp(body.name)
    return OKResponse()
  }

  protected.post("apps", "open_url") { req async throws -> OKResponse in
    let body = try req.content.decode(OpenURLRequest.self)
    try AppManager.openURL(body.url)
    return OKResponse()
  }

  protected.post("apps", "open_file") { req async throws -> OKResponse in
    let body = try req.content.decode(OpenFileRequest.self)
    AppManager.openFile(body.path)
    return OKResponse()
  }

  // MARK: Windows

  protected.post("windows", "list") { req async throws -> WindowListResponse in
    return WindowManager.listWindows()
  }

  protected.post("windows", "focus") { req async throws -> OKResponse in
    let body = try req.content.decode(FocusWindowRequest.self)
    try WindowManager.focusWindow(id: body.id, title: body.title)
    return OKResponse()
  }

  protected.post("windows", "resize") { req async throws -> OKResponse in
    let body = try req.content.decode(ResizeWindowRequest.self)
    try WindowManager.resizeWindow(id: body.id, x: body.x, y: body.y, width: body.width, height: body.height)
    return OKResponse()
  }

  protected.post("windows", "close") { req async throws -> OKResponse in
    let body = try req.content.decode(WindowIDRequest.self)
    try WindowManager.closeWindow(id: body.id)
    return OKResponse()
  }

  // MARK: Accessibility

  protected.post("ax", "get_focused_element") { req async throws -> FocusedElementResponse in
    return AccessibilityEngine.getFocusedElement()
  }

  protected.post("ax", "get_ui_tree") { req async throws -> Response in
    let body = try req.content.decode(UITreeRequest.self)
    let tree = try AccessibilityEngine.getUITree(appName: body.app, depth: body.depth ?? 3)
    let data = try JSONSerialization.data(withJSONObject: tree)
    return Response(status: .ok, headers: ["content-type": "application/json"], body: .init(data: data))
  }

  protected.post("ax", "find_element") { req async throws -> FindElementResponse in
    let body = try req.content.decode(FindElementRequest.self)
    return try AccessibilityEngine.findElement(appName: body.app, role: body.role, title: body.title)
  }

  protected.post("ax", "perform_action") { req async throws -> OKResponse in
    let body = try req.content.decode(PerformActionRequest.self)
    try AccessibilityEngine.performAction(appName: body.app, role: body.role, title: body.title, action: body.action)
    return OKResponse()
  }

  protected.post("ax", "set_value") { req async throws -> OKResponse in
    let body = try req.content.decode(SetValueRequest.self)
    try AccessibilityEngine.setValue(appName: body.app, role: body.role, title: body.title, value: body.value)
    return OKResponse()
  }

  // MARK: AppleScript

  protected.post("applescript", "run") { req async throws -> AppleScriptResponse in
    let body = try req.content.decode(AppleScriptRequest.self)
    return AppleScriptEngine.run(script: body.script)
  }

  // MARK: Files

  protected.post("files", "list") { req async throws -> FileListResponse in
    let body = try req.content.decode(FileListRequest.self)
    return try FileBridge.listFiles(path: body.path, recursive: body.recursive ?? false)
  }

  protected.post("files", "read") { req async throws -> FileContentResponse in
    let body = try req.content.decode(FilePathRequest.self)
    return try FileBridge.readFile(path: body.path)
  }

  protected.post("files", "write") { req async throws -> OKResponse in
    let body = try req.content.decode(FileWriteRequest.self)
    try FileBridge.writeFile(path: body.path, content: body.content)
    return OKResponse()
  }

  protected.post("files", "move") { req async throws -> OKResponse in
    let body = try req.content.decode(FileMoveRequest.self)
    try FileBridge.moveFile(from: body.from, to: body.to)
    return OKResponse()
  }

  protected.post("files", "delete") { req async throws -> OKResponse in
    let body = try req.content.decode(FileDeleteRequest.self)
    try FileBridge.deleteFile(path: body.path, toTrash: body.toTrash ?? true)
    return OKResponse()
  }

  protected.post("files", "info") { req async throws -> FileInfoResponse in
    let body = try req.content.decode(FilePathRequest.self)
    return try FileBridge.fileInfo(path: body.path)
  }

  // MARK: Clipboard

  protected.post("clipboard", "get") { req async throws -> ClipboardGetResponse in
    return ClipboardManager.getText()
  }

  protected.post("clipboard", "set") { req async throws -> OKResponse in
    let body = try req.content.decode(ClipboardSetRequest.self)
    ClipboardManager.setText(body.text)
    return OKResponse()
  }

  // MARK: System

  protected.post("system", "info") { req async throws -> SystemInfoResponse in
    return SystemInfo.getSystemInfo()
  }

  protected.post("system", "volume") { req async throws -> VolumeResponse in
    return SystemInfo.getVolume()
  }

  protected.post("system", "set_volume") { req async throws -> OKResponse in
    let body = try req.content.decode(SetVolumeRequest.self)
    SystemInfo.setVolume(body.volume)
    return OKResponse()
  }

  protected.post("system", "battery") { req async throws -> BatteryResponse in
    return SystemInfo.getBattery()
  }

  protected.post("system", "dark_mode") { req async throws -> DarkModeResponse in
    return SystemInfo.getDarkMode()
  }

  protected.post("system", "set_dark_mode") { req async throws -> OKResponse in
    let body = try req.content.decode(SetDarkModeRequest.self)
    SystemInfo.setDarkMode(enabled: body.enabled)
    return OKResponse()
  }

  protected.post("system", "lock") { req async throws -> OKResponse in
    SystemInfo.lockScreen()
    return OKResponse()
  }

  // MARK: Notifications

  protected.post("notifications", "send") { req async throws -> OKResponse in
    let body = try req.content.decode(NotificationRequest.self)
    NotificationManager.send(title: body.title, message: body.message)
    return OKResponse()
  }
}

// MARK: - Convenience OK response

struct OKResponse: Content {
  let status = "ok"
}

// MARK: - Request body types

struct ScreenCaptureRequest: Decodable {
  let display: Int?
  let region: [String: Double]?
  let max_width: Int?
  var cgRect: CGRect? {
    guard let r = region, let x = r["x"], let y = r["y"], let w = r["width"], let h = r["height"] else { return nil }
    return CGRect(x: x, y: y, width: w, height: h)
  }
}

struct ScreenOCRRequest: Decodable {
  let region: [String: Double]?
  var cgRect: CGRect? {
    guard let r = region, let x = r["x"], let y = r["y"], let w = r["width"], let h = r["height"] else { return nil }
    return CGRect(x: x, y: y, width: w, height: h)
  }
}

struct FindTextRequest: Decodable { let text: String; let nth: Int? }
struct ClickRequest: Decodable { let x: Double; let y: Double; let button: String?; let clicks: Int?; let modifiers: [String]? }
struct TypeRequest: Decodable { let text: String }
struct KeyRequest: Decodable { let key: String; let modifiers: [String]? }
struct MouseMoveRequest: Decodable { let x: Double; let y: Double }
struct ScrollRequest: Decodable { let x: Double; let y: Double; let dx: Double?; let dy: Double? }
struct DragRequest: Decodable { let from_x: Double; let from_y: Double; let to_x: Double; let to_y: Double; let duration: Double? }
struct AppNameRequest: Decodable { let name: String }
struct QuitAppRequest: Decodable { let name: String; let force: Bool? }
struct OpenURLRequest: Decodable { let url: String }
struct OpenFileRequest: Decodable { let path: String }
struct FocusWindowRequest: Decodable { let id: Int?; let title: String? }
struct ResizeWindowRequest: Decodable { let id: Int; let x: Double; let y: Double; let width: Double; let height: Double }
struct WindowIDRequest: Decodable { let id: Int }
struct UITreeRequest: Decodable { let app: String; let depth: Int? }
struct FindElementRequest: Decodable { let app: String; let role: String?; let title: String? }
struct PerformActionRequest: Decodable { let app: String; let role: String?; let title: String?; let action: String }
struct SetValueRequest: Decodable { let app: String; let role: String; let title: String; let value: String }
struct AppleScriptRequest: Decodable { let script: String }
struct FileListRequest: Decodable { let path: String; let recursive: Bool? }
struct FilePathRequest: Decodable { let path: String }
struct FileWriteRequest: Decodable { let path: String; let content: String }
struct FileMoveRequest: Decodable { let from: String; let to: String }
struct FileDeleteRequest: Decodable { let path: String; let toTrash: Bool? }
struct ClipboardSetRequest: Decodable { let text: String }
struct SetVolumeRequest: Decodable { let volume: Int }
struct SetDarkModeRequest: Decodable { let enabled: Bool }
struct NotificationRequest: Decodable { let title: String; let message: String }
