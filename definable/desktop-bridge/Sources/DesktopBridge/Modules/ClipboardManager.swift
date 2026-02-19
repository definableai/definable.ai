import AppKit
import Foundation

/// Clipboard read/write via NSPasteboard.
enum ClipboardManager {
  static func getText() -> ClipboardGetResponse {
    let pb = NSPasteboard.general
    let text = pb.string(forType: .string) ?? ""
    let hasImage = pb.data(forType: .tiff) != nil || pb.data(forType: .png) != nil
    return ClipboardGetResponse(text: text, hasImage: hasImage)
  }

  static func setText(_ text: String) {
    let pb = NSPasteboard.general
    pb.clearContents()
    pb.setString(text, forType: .string)
  }
}
