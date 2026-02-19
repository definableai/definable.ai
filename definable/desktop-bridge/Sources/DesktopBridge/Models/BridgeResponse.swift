import Foundation
import Vapor

// ---------------------------------------------------------------------------
// Generic response envelope
// ---------------------------------------------------------------------------

struct BridgeResponse<T: Content>: Content {
  let success: Bool
  let data: T?
  let error: String?

  init(_ data: T) {
    self.success = true
    self.data = data
    self.error = nil
  }
}

struct EmptyData: Content {}

extension BridgeResponse where T == EmptyData {
  static func ok() -> BridgeResponse<EmptyData> {
    return BridgeResponse(EmptyData())
  }
}

// ---------------------------------------------------------------------------
// Shared geometry
// ---------------------------------------------------------------------------

struct BoundsResponse: Content {
  let x: Double
  let y: Double
  let width: Double
  let height: Double
}

// ---------------------------------------------------------------------------
// Screen
// ---------------------------------------------------------------------------

struct CaptureResponse: Content {
  let image: String   // base64-encoded image
  let format: String  // "jpeg" or "png"
}

struct OCRElement: Content {
  let text: String
  let bounds: BoundsResponse
}

struct OCRResponse: Content {
  let text: String
  let elements: [OCRElement]
}

struct FindTextResponse: Content {
  let found: Bool
  let bounds: BoundsResponse?
}

// ---------------------------------------------------------------------------
// Apps
// ---------------------------------------------------------------------------

struct AppInfoResponse: Content {
  let name: String
  let bundleId: String
  let pid: Int32
  let active: Bool
}

struct AppListResponse: Content {
  let apps: [AppInfoResponse]
}

struct OpenAppResponse: Content {
  let pid: Int32
}

// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------

struct WindowInfoResponse: Content {
  let id: Int
  let app: String
  let title: String
  let bounds: BoundsResponse
  let minimized: Bool
}

struct WindowListResponse: Content {
  let windows: [WindowInfoResponse]
}

// ---------------------------------------------------------------------------
// Accessibility
// ---------------------------------------------------------------------------

struct UIElementResponse: Content {
  let role: String
  let title: String
  let value: String
  let bounds: BoundsResponse
  let children: [UIElementResponse]?
}

struct FindElementResponse: Content {
  let found: Bool
  let element: UIElementResponse?
}

struct FocusedElementResponse: Content {
  let found: Bool
  let element: UIElementResponse?
}

// ---------------------------------------------------------------------------
// Permissions / health
// ---------------------------------------------------------------------------

struct PermissionsResponse: Content {
  let accessibility: Bool
  let screenRecording: Bool
  let fullDiskAccess: Bool
}

struct HealthResponse: Content {
  let status: String
  let version: String
  let permissions: PermissionsResponse
}

// ---------------------------------------------------------------------------
// AppleScript
// ---------------------------------------------------------------------------

struct AppleScriptResponse: Content {
  let output: String
  let error: String?
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

struct FileEntry: Content {
  let name: String
  let path: String
  let kind: String
  let size: Int64
}

struct FileListResponse: Content {
  let files: [FileEntry]
}

struct FileContentResponse: Content {
  let content: String
}

struct FileInfoResponse: Content {
  let size: Int64
  let created: String
  let modified: String
  let kind: String
}

// ---------------------------------------------------------------------------
// Clipboard
// ---------------------------------------------------------------------------

struct ClipboardGetResponse: Content {
  let text: String
  let hasImage: Bool
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

struct SystemInfoResponse: Content {
  let hostname: String
  let osVersion: String
  let cpu: String
  let memoryGb: Double
}

struct VolumeResponse: Content {
  let volume: Int
}

struct BatteryResponse: Content {
  let level: Int
  let charging: Bool
  let timeRemaining: Int
}

struct DarkModeResponse: Content {
  let enabled: Bool
}
