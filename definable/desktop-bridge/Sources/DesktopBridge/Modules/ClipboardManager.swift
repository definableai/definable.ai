import AppKit
import Foundation

enum ClipboardManager {
  static func getText() -> String? {
    NSPasteboard.general.string(forType: .string)
  }

  static func setText(_ text: String) {
    let pasteboard = NSPasteboard.general
    pasteboard.clearContents()
    pasteboard.setString(text, forType: .string)
  }

  static func hasImage() -> Bool {
    let types: [NSPasteboard.PasteboardType] = [.tiff, .png]
    return NSPasteboard.general.availableType(from: types) != nil
  }
}
