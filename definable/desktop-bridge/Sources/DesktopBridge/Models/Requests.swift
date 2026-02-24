import Vapor

struct CaptureRequest: Content {
  var display: Int?
  var maxWidth: Int?
  var region: ScreenRegion?
}

struct ScreenRegion: Content {
  var x: Double
  var y: Double
  var width: Double
  var height: Double
}

struct OCRRequest: Content {
  var region: ScreenRegion?
}

struct FindTextRequest: Content {
  var text: String
  var nth: Int?
}

struct ClickRequest: Content {
  var x: Double
  var y: Double
  var button: String?
  var clicks: Int?
  var modifiers: [String]?
}

struct TypeTextRequest: Content {
  var text: String
}

struct KeyRequest: Content {
  var key: String
  var modifiers: [String]?
}

struct MouseMoveRequest: Content {
  var x: Double
  var y: Double
}

struct ScrollRequest: Content {
  var x: Double
  var y: Double
  var dx: Double?
  var dy: Double?
}

struct DragRequest: Content {
  var fromX: Double
  var fromY: Double
  var toX: Double
  var toY: Double
  var duration: Double?

  enum CodingKeys: String, CodingKey {
    case fromX = "from_x"
    case fromY = "from_y"
    case toX = "to_x"
    case toY = "to_y"
    case duration
  }
}

struct OpenAppRequest: Content {
  var name: String?
  var bundleId: String?
  var path: String?

  enum CodingKeys: String, CodingKey {
    case name
    case bundleId = "bundle_id"
    case path
  }
}

struct QuitAppRequest: Content {
  var name: String
  var force: Bool?
}

struct ActivateAppRequest: Content {
  var name: String
}

struct OpenURLRequest: Content {
  var url: String
}

struct OpenFileRequest: Content {
  var path: String
}

struct FocusWindowRequest: Content {
  var windowId: Int?
  var title: String?

  enum CodingKeys: String, CodingKey {
    case windowId = "window_id"
    case title
  }
}

struct ResizeWindowRequest: Content {
  var windowId: Int
  var x: Double?
  var y: Double?
  var width: Double?
  var height: Double?

  enum CodingKeys: String, CodingKey {
    case windowId = "window_id"
    case x, y, width, height
  }
}

struct CloseWindowRequest: Content {
  var windowId: Int

  enum CodingKeys: String, CodingKey {
    case windowId = "window_id"
  }
}

struct UITreeRequest: Content {
  var app: String
  var depth: Int?
}

struct FindElementRequest: Content {
  var app: String
  var role: String?
  var title: String?
}

struct PerformActionRequest: Content {
  var app: String
  var role: String?
  var title: String?
  var action: String?
}

struct SetValueRequest: Content {
  var app: String
  var role: String?
  var title: String?
  var value: String
}

struct AppleScriptRequest: Content {
  var script: String
}

struct ListFilesRequest: Content {
  var path: String
  var recursive: Bool?
}

struct ReadFileRequest: Content {
  var path: String
}

struct WriteFileRequest: Content {
  var path: String
  var content: String
}

struct MoveFileRequest: Content {
  var from: String
  var to: String
}

struct DeleteFileRequest: Content {
  var path: String
  var toTrash: Bool?

  enum CodingKeys: String, CodingKey {
    case path
    case toTrash = "to_trash"
  }
}

struct FileInfoRequest: Content {
  var path: String
}

struct SetClipboardRequest: Content {
  var text: String
}

struct SetVolumeRequest: Content {
  var volume: Int
}

struct SetDarkModeRequest: Content {
  var enabled: Bool
}

struct NotificationRequest: Content {
  var title: String
  var message: String
  var sound: String?
}

struct ShellRequest: Content {
  var command: [String]
  var cwd: String?
  var env: [String: String]?
  var timeout: Double?
}

struct CameraSnapRequest: Content {
  var facing: String?
  var maxWidth: Int?
  var quality: Double?
  var outPath: String?

  enum CodingKeys: String, CodingKey {
    case facing
    case maxWidth = "max_width"
    case quality
    case outPath = "out_path"
  }
}

struct CameraClipRequest: Content {
  var facing: String?
  var durationMs: Int?
  var includeAudio: Bool?
  var outPath: String?

  enum CodingKeys: String, CodingKey {
    case facing
    case durationMs = "duration_ms"
    case includeAudio = "include_audio"
    case outPath = "out_path"
  }
}

struct ScreenRecordRequest: Content {
  var screenIndex: Int?
  var durationMs: Int?
  var fps: Double?
  var includeAudio: Bool?
  var outPath: String?

  enum CodingKeys: String, CodingKey {
    case screenIndex = "screen_index"
    case durationMs = "duration_ms"
    case fps
    case includeAudio = "include_audio"
    case outPath = "out_path"
  }
}
