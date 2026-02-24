import Foundation
import Vapor

// Shared instances for actor-based modules
private let screenCapture = ScreenCapture()
private let cameraCapture = CameraCapture()

// MARK: - Health

func registerHealthRoutes(_ app: RoutesBuilder) {
  app.post("health") { _ async throws -> BridgeResponse<HealthData> in
    .success(HealthData(
      status: "ok",
      permissions: PermissionStatus(
        accessibility: PermissionChecker.checkAccessibility(),
        screenRecording: PermissionChecker.checkScreenRecording(),
        fullDiskAccess: PermissionChecker.checkFullDiskAccess())))
  }
}

// MARK: - Screen

func registerScreenRoutes(_ app: RoutesBuilder) {
  let screen = app.grouped("screen")

  screen.post("capture") { req async throws -> BridgeResponse<CaptureData> in
    let body = try req.content.decode(CaptureRequest.self)
    let region: CGRect? = body.region.map {
      CGRect(x: $0.x, y: $0.y, width: $0.width, height: $0.height)
    }
    let result = try await screenCapture.captureScreen(
      display: body.display ?? 0,
      maxWidth: body.maxWidth ?? 512,
      region: region)
    return .success(CaptureData(
      image: result.data.base64EncodedString(),
      width: result.width,
      height: result.height))
  }

  screen.post("ocr") { req async throws -> BridgeResponse<OCRData> in
    let body = try req.content.decode(OCRRequest.self)
    let region: CGRect? = body.region.map {
      CGRect(x: $0.x, y: $0.y, width: $0.width, height: $0.height)
    }
    let result = try await screenCapture.ocrScreen(region: region)
    return .success(OCRData(
      text: result.text,
      elements: result.elements.map { e in
        OCRElement(text: e.text, x: e.x, y: e.y, width: e.width, height: e.height, confidence: e.confidence)
      }))
  }

  screen.post("find_text") { req async throws -> BridgeResponse<TextLocation> in
    let body = try req.content.decode(FindTextRequest.self)
    guard let loc = try await screenCapture.findText(body.text, nth: body.nth ?? 0) else {
      throw Abort(.notFound, reason: "Text '\(body.text)' not found on screen")
    }
    return .success(TextLocation(
      x: loc.x, y: loc.y, width: loc.width, height: loc.height,
      centerX: loc.centerX, centerY: loc.centerY))
  }
}

// MARK: - Input

func registerInputRoutes(_ app: RoutesBuilder) {
  let input = app.grouped("input")

  input.post("click") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(ClickRequest.self)
    InputSimulator.click(
      x: body.x, y: body.y,
      button: body.button ?? "left",
      clicks: body.clicks ?? 1,
      modifiers: body.modifiers ?? [])
    return .success(EmptyData())
  }

  input.post("type") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(TypeTextRequest.self)
    InputSimulator.typeText(body.text)
    return .success(EmptyData())
  }

  input.post("key") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(KeyRequest.self)
    InputSimulator.pressKey(body.key, modifiers: body.modifiers ?? [])
    return .success(EmptyData())
  }

  input.post("mouse_move") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(MouseMoveRequest.self)
    InputSimulator.mouseMove(x: body.x, y: body.y)
    return .success(EmptyData())
  }

  input.post("scroll") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(ScrollRequest.self)
    InputSimulator.scroll(x: body.x, y: body.y, dx: body.dx ?? 0, dy: body.dy ?? -3)
    return .success(EmptyData())
  }

  input.post("drag") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(DragRequest.self)
    InputSimulator.drag(
      fromX: body.fromX, fromY: body.fromY,
      toX: body.toX, toY: body.toY,
      duration: body.duration ?? 0.5)
    return .success(EmptyData())
  }
}

// MARK: - Apps

func registerAppRoutes(_ app: RoutesBuilder) {
  let apps = app.grouped("apps")

  apps.post("list") { _ async throws -> BridgeResponse<[AppInfoData]> in
    let list = AppManager.listRunningApps()
    return .success(list.map { app in
      AppInfoData(name: app.name, bundleId: app.bundleId, pid: app.pid, active: app.active)
    })
  }

  apps.post("open") { req async throws -> BridgeResponse<OpenAppData> in
    let body = try req.content.decode(OpenAppRequest.self)
    let pid = try await AppManager.openApp(name: body.name, bundleId: body.bundleId, path: body.path)
    return .success(OpenAppData(pid: pid))
  }

  apps.post("quit") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(QuitAppRequest.self)
    let ok = AppManager.quitApp(name: body.name, force: body.force ?? false)
    if !ok { throw Abort(.notFound, reason: "App '\(body.name)' not found") }
    return .success(EmptyData())
  }

  apps.post("activate") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(ActivateAppRequest.self)
    let ok = AppManager.activateApp(name: body.name)
    if !ok { throw Abort(.notFound, reason: "App '\(body.name)' not found") }
    return .success(EmptyData())
  }

  apps.post("open_url") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(OpenURLRequest.self)
    let ok = AppManager.openURL(body.url)
    if !ok { throw Abort(.badRequest, reason: "Failed to open URL") }
    return .success(EmptyData())
  }

  apps.post("open_file") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(OpenFileRequest.self)
    let ok = AppManager.openFile(body.path)
    if !ok { throw Abort(.notFound, reason: "File not found or no app to open it") }
    return .success(EmptyData())
  }
}

// MARK: - Windows

func registerWindowRoutes(_ app: RoutesBuilder) {
  let windows = app.grouped("windows")

  windows.post("list") { _ async throws -> BridgeResponse<[WindowInfoData]> in
    let list = WindowManager.listWindows()
    return .success(list.map { w in
      WindowInfoData(id: w.id, app: w.app, title: w.title,
                     x: w.x, y: w.y, width: w.width, height: w.height,
                     minimized: w.minimized)
    })
  }

  windows.post("focus") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(FocusWindowRequest.self)
    let ok = WindowManager.focusWindow(windowId: body.windowId, title: body.title)
    if !ok { throw Abort(.notFound, reason: "Window not found") }
    return .success(EmptyData())
  }

  windows.post("resize") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(ResizeWindowRequest.self)
    let ok = WindowManager.resizeWindow(
      windowId: body.windowId,
      x: body.x, y: body.y, width: body.width, height: body.height)
    if !ok { throw Abort(.notFound, reason: "Window not found") }
    return .success(EmptyData())
  }

  windows.post("close") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(CloseWindowRequest.self)
    let ok = WindowManager.closeWindow(windowId: body.windowId)
    if !ok { throw Abort(.notFound, reason: "Window not found") }
    return .success(EmptyData())
  }
}

// MARK: - Accessibility

func registerAccessibilityRoutes(_ app: RoutesBuilder) {
  let ax = app.grouped("ax")

  ax.post("get_focused_element") { _ async throws -> BridgeResponse<UIElementData> in
    guard let el = AccessibilityEngine.getFocusedElement() else {
      throw Abort(.notFound, reason: "No focused element")
    }
    return .success(convertUIElement(el))
  }

  ax.post("get_ui_tree") { req async throws -> BridgeResponse<UIElementData> in
    let body = try req.content.decode(UITreeRequest.self)
    guard let tree = AccessibilityEngine.getUITree(appName: body.app, depth: body.depth ?? 3) else {
      throw Abort(.notFound, reason: "App '\(body.app)' not found")
    }
    return .success(convertUIElement(tree))
  }

  ax.post("find_element") { req async throws -> BridgeResponse<UIElementData> in
    let body = try req.content.decode(FindElementRequest.self)
    guard let el = AccessibilityEngine.findElement(appName: body.app, role: body.role, title: body.title) else {
      throw Abort(.notFound, reason: "Element not found")
    }
    return .success(convertUIElement(el))
  }

  ax.post("perform_action") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(PerformActionRequest.self)
    let ok = AccessibilityEngine.performAction(
      appName: body.app, role: body.role, title: body.title, action: body.action ?? "AXPress")
    if !ok { throw Abort(.notFound, reason: "Element not found or action failed") }
    return .success(EmptyData())
  }

  ax.post("set_value") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(SetValueRequest.self)
    let ok = AccessibilityEngine.setValue(appName: body.app, role: body.role, title: body.title, value: body.value)
    if !ok { throw Abort(.notFound, reason: "Element not found or value set failed") }
    return .success(EmptyData())
  }
}

// MARK: - AppleScript

func registerAppleScriptRoutes(_ app: RoutesBuilder) {
  app.post("applescript", "run") { req async throws -> BridgeResponse<AppleScriptData> in
    let body = try req.content.decode(AppleScriptRequest.self)
    let result = await MainActor.run { AppleScriptEngine.execute(script: body.script) }
    return .success(AppleScriptData(output: result.output, error: result.error))
  }
}

// MARK: - Files

func registerFileRoutes(_ app: RoutesBuilder) {
  let files = app.grouped("files")

  files.post("list") { req async throws -> BridgeResponse<[FileEntryData]> in
    let body = try req.content.decode(ListFilesRequest.self)
    let entries = try FileBridge.listFiles(path: body.path, recursive: body.recursive ?? false)
    return .success(entries.map { e in
      FileEntryData(name: e.name, path: e.path, isDirectory: e.isDirectory, size: e.size)
    })
  }

  files.post("read") { req async throws -> BridgeResponse<FileContentData> in
    let body = try req.content.decode(ReadFileRequest.self)
    let content = try FileBridge.readFile(path: body.path)
    return .success(FileContentData(content: content))
  }

  files.post("write") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(WriteFileRequest.self)
    try FileBridge.writeFile(path: body.path, content: body.content)
    return .success(EmptyData())
  }

  files.post("move") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(MoveFileRequest.self)
    try FileBridge.moveFile(from: body.from, to: body.to)
    return .success(EmptyData())
  }

  files.post("delete") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(DeleteFileRequest.self)
    try FileBridge.deleteFile(path: body.path, toTrash: body.toTrash ?? true)
    return .success(EmptyData())
  }

  files.post("info") { req async throws -> BridgeResponse<FileInfoData> in
    let body = try req.content.decode(FileInfoRequest.self)
    let info = try FileBridge.fileInfo(path: body.path)
    return .success(FileInfoData(size: info.size, created: info.created, modified: info.modified, kind: info.kind))
  }
}

// MARK: - Clipboard

func registerClipboardRoutes(_ app: RoutesBuilder) {
  let clipboard = app.grouped("clipboard")

  clipboard.post("get") { _ async throws -> BridgeResponse<ClipboardData> in
    .success(ClipboardData(text: ClipboardManager.getText(), hasImage: ClipboardManager.hasImage()))
  }

  clipboard.post("set") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(SetClipboardRequest.self)
    ClipboardManager.setText(body.text)
    return .success(EmptyData())
  }
}

// MARK: - System

func registerSystemRoutes(_ app: RoutesBuilder) {
  let system = app.grouped("system")

  system.post("info") { _ async throws -> BridgeResponse<SystemInfoData> in
    .success(SystemInfoData(
      hostname: SystemInfo.hostname(),
      osVersion: SystemInfo.osVersion(),
      cpu: SystemInfo.cpu(),
      memoryGb: SystemInfo.memoryGB()))
  }

  system.post("volume") { _ async throws -> BridgeResponse<VolumeData> in
    .success(VolumeData(volume: SystemInfo.volume()))
  }

  system.post("set_volume") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(SetVolumeRequest.self)
    SystemInfo.setVolume(body.volume)
    return .success(EmptyData())
  }

  system.post("battery") { _ async throws -> BridgeResponse<BatteryData> in
    let b = SystemInfo.battery()
    return .success(BatteryData(level: b.level, charging: b.charging, timeRemaining: b.timeRemaining))
  }

  system.post("dark_mode") { _ async throws -> BridgeResponse<DarkModeData> in
    .success(DarkModeData(enabled: SystemInfo.isDarkMode()))
  }

  system.post("set_dark_mode") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(SetDarkModeRequest.self)
    SystemInfo.setDarkMode(body.enabled)
    return .success(EmptyData())
  }

  system.post("lock") { _ async throws -> BridgeResponse<EmptyData> in
    SystemInfo.lockScreen()
    return .success(EmptyData())
  }
}

// MARK: - Shell

func registerShellRoutes(_ app: RoutesBuilder) {
  app.post("shell", "run") { req async throws -> BridgeResponse<ShellResultData> in
    let body = try req.content.decode(ShellRequest.self)
    let result = await ShellExecutor.run(
      command: body.command,
      cwd: body.cwd,
      env: body.env,
      timeout: body.timeout)
    return .success(ShellResultData(
      stdout: result.stdout,
      stderr: result.stderr,
      exitCode: result.exitCode,
      timedOut: result.timedOut,
      success: result.success))
  }
}

// MARK: - Camera

func registerCameraRoutes(_ app: RoutesBuilder) {
  let camera = app.grouped("camera")

  camera.post("list") { _ async throws -> BridgeResponse<[CameraDeviceData]> in
    let devices = await cameraCapture.listDevices()
    return .success(devices.map { d in
      CameraDeviceData(id: d.id, name: d.name, position: d.position)
    })
  }

  camera.post("snap") { req async throws -> BridgeResponse<CameraSnapData> in
    let body = try req.content.decode(CameraSnapRequest.self)
    let result = try await cameraCapture.snap(
      facing: body.facing ?? "front",
      maxWidth: body.maxWidth ?? 1600,
      quality: body.quality ?? 0.9,
      outPath: body.outPath)
    return .success(CameraSnapData(
      image: result.data.base64EncodedString(),
      width: result.width,
      height: result.height))
  }

  camera.post("clip") { req async throws -> BridgeResponse<CameraClipData> in
    let body = try req.content.decode(CameraClipRequest.self)
    let result = try await cameraCapture.clip(
      facing: body.facing ?? "front",
      durationMs: body.durationMs ?? 3000,
      includeAudio: body.includeAudio ?? false,
      outPath: body.outPath)
    return .success(CameraClipData(
      path: result.path,
      durationMs: result.durationMs,
      hasAudio: result.hasAudio))
  }
}

// MARK: - Screen Record

func registerScreenRecordRoutes(_ app: RoutesBuilder) {
  app.post("screen", "record") { req async throws -> BridgeResponse<ScreenRecordData> in
    let body = try req.content.decode(ScreenRecordRequest.self)
    let recorder = await MainActor.run { ScreenRecorder() }
    let result = try await recorder.record(
      screenIndex: body.screenIndex,
      durationMs: body.durationMs,
      fps: body.fps,
      includeAudio: body.includeAudio ?? false,
      outPath: body.outPath)
    return .success(ScreenRecordData(path: result.path, hasAudio: result.hasAudio))
  }
}

// MARK: - Notifications

func registerNotificationRoutes(_ app: RoutesBuilder) {
  app.post("notifications", "send") { req async throws -> BridgeResponse<EmptyData> in
    let body = try req.content.decode(NotificationRequest.self)
    let ok = await NotificationService.send(title: body.title, message: body.message, sound: body.sound)
    if !ok { throw Abort(.forbidden, reason: "Notification permission denied") }
    return .success(EmptyData())
  }
}

// MARK: - Helpers

private func convertUIElement(_ el: AccessibilityEngine.UIElement) -> UIElementData {
  UIElementData(
    role: el.role,
    title: el.title,
    value: el.value,
    x: el.x, y: el.y,
    width: el.width, height: el.height,
    children: el.children?.map { convertUIElement($0) })
}

