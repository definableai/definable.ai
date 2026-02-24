import Foundation

enum AppleScriptEngine {
  struct ScriptResult {
    var output: String?
    var error: String?
  }

  @MainActor
  static func execute(script: String) -> ScriptResult {
    var errorDict: NSDictionary?
    let appleScript = NSAppleScript(source: script)
    let result = appleScript?.executeAndReturnError(&errorDict)

    if let errorDict {
      let message = errorDict["NSAppleScriptErrorMessage"] as? String
        ?? errorDict["NSAppleScriptErrorBriefMessage"] as? String
        ?? "Unknown AppleScript error"
      return ScriptResult(output: nil, error: message)
    }

    return ScriptResult(output: result?.stringValue, error: nil)
  }
}
