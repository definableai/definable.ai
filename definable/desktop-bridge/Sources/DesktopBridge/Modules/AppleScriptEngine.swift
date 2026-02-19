import AppKit
import Foundation

/// Executes AppleScript via NSAppleScript.
enum AppleScriptEngine {
  static func run(script: String) -> AppleScriptResponse {
    var error: NSDictionary?
    let appleScript = NSAppleScript(source: script)
    guard let result = appleScript?.executeAndReturnError(&error) else {
      let errMsg = (error?["NSAppleScriptErrorMessage"] as? String) ?? "Unknown error"
      return AppleScriptResponse(output: "", error: errMsg)
    }
    return AppleScriptResponse(output: result.stringValue ?? "", error: nil)
  }
}
