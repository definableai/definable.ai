import Vapor

struct BridgeResponse<T: Content>: Content {
  var ok: Bool
  var data: T?
  var error: String?

  static func success(_ data: T) -> BridgeResponse {
    BridgeResponse(ok: true, data: data, error: nil)
  }

  static func failure(_ message: String) -> BridgeResponse<EmptyData> {
    BridgeResponse<EmptyData>(ok: false, data: nil, error: message)
  }
}

struct EmptyData: Content {}

struct HealthData: Content {
  var status: String
  var permissions: PermissionStatus
}

struct PermissionStatus: Content {
  var accessibility: Bool
  var screenRecording: Bool
  var fullDiskAccess: Bool

  enum CodingKeys: String, CodingKey {
    case accessibility
    case screenRecording = "screen_recording"
    case fullDiskAccess = "full_disk_access"
  }
}

struct CaptureData: Content {
  var image: String  // base64 JPEG
  var width: Int
  var height: Int
}

struct OCRData: Content {
  var text: String
  var elements: [OCRElement]
}

struct OCRElement: Content {
  var text: String
  var x: Double
  var y: Double
  var width: Double
  var height: Double
  var confidence: Double
}

struct TextLocation: Content {
  var x: Double
  var y: Double
  var width: Double
  var height: Double
  var centerX: Double
  var centerY: Double

  enum CodingKeys: String, CodingKey {
    case x, y, width, height
    case centerX = "center_x"
    case centerY = "center_y"
  }
}

struct AppInfoData: Content {
  var name: String
  var bundleId: String?
  var pid: Int
  var active: Bool

  enum CodingKeys: String, CodingKey {
    case name
    case bundleId = "bundle_id"
    case pid
    case active
  }
}

struct OpenAppData: Content {
  var pid: Int
}

struct WindowInfoData: Content {
  var id: Int
  var app: String
  var title: String
  var x: Double
  var y: Double
  var width: Double
  var height: Double
  var minimized: Bool
}

struct UIElementData: Content {
  var role: String?
  var title: String?
  var value: String?
  var x: Double?
  var y: Double?
  var width: Double?
  var height: Double?
  var children: [UIElementData]?
}

struct AppleScriptData: Content {
  var output: String?
  var error: String?
}

struct FileEntryData: Content {
  var name: String
  var path: String
  var isDirectory: Bool
  var size: Int?

  enum CodingKeys: String, CodingKey {
    case name, path
    case isDirectory = "is_directory"
    case size
  }
}

struct FileContentData: Content {
  var content: String
}

struct FileInfoData: Content {
  var size: Int
  var created: String?
  var modified: String?
  var kind: String
}

struct ClipboardData: Content {
  var text: String?
  var hasImage: Bool

  enum CodingKeys: String, CodingKey {
    case text
    case hasImage = "has_image"
  }
}

struct SystemInfoData: Content {
  var hostname: String
  var osVersion: String
  var cpu: String
  var memoryGb: Double

  enum CodingKeys: String, CodingKey {
    case hostname
    case osVersion = "os_version"
    case cpu
    case memoryGb = "memory_gb"
  }
}

struct VolumeData: Content {
  var volume: Int
}

struct BatteryData: Content {
  var level: Int
  var charging: Bool
  var timeRemaining: Int?

  enum CodingKeys: String, CodingKey {
    case level, charging
    case timeRemaining = "time_remaining"
  }
}

struct DarkModeData: Content {
  var enabled: Bool
}

struct ShellResultData: Content {
  var stdout: String
  var stderr: String
  var exitCode: Int?
  var timedOut: Bool
  var success: Bool

  enum CodingKeys: String, CodingKey {
    case stdout, stderr
    case exitCode = "exit_code"
    case timedOut = "timed_out"
    case success
  }
}

struct CameraSnapData: Content {
  var image: String  // base64 JPEG
  var width: Int
  var height: Int
}

struct CameraClipData: Content {
  var path: String
  var durationMs: Int
  var hasAudio: Bool

  enum CodingKeys: String, CodingKey {
    case path
    case durationMs = "duration_ms"
    case hasAudio = "has_audio"
  }
}

struct ScreenRecordData: Content {
  var path: String
  var hasAudio: Bool

  enum CodingKeys: String, CodingKey {
    case path
    case hasAudio = "has_audio"
  }
}

struct CameraDeviceData: Content {
  var id: String
  var name: String
  var position: String
}
