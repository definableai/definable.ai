import AppKit
import CoreAudio
import Foundation
import IOKit.ps

enum SystemInfo {
  static func hostname() -> String {
    ProcessInfo.processInfo.hostName
  }

  static func osVersion() -> String {
    let v = ProcessInfo.processInfo.operatingSystemVersion
    return "macOS \(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
  }

  static func cpu() -> String {
    var size = 0
    sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
    var buffer = [CChar](repeating: 0, count: size)
    sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0)
    return String(cString: buffer)
  }

  static func memoryGB() -> Double {
    Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
  }

  static func volume() -> Int {
    var defaultOutputID = AudioObjectID(kAudioObjectSystemObject)
    var address = AudioObjectPropertyAddress(
      mSelector: kAudioHardwarePropertyDefaultOutputDevice,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &defaultOutputID)

    var vol: Float32 = 0
    address.mSelector = kAudioDevicePropertyVolumeScalar
    address.mScope = kAudioDevicePropertyScopeOutput
    address.mElement = 1  // Channel 1 (main)
    size = UInt32(MemoryLayout<Float32>.size)
    AudioObjectGetPropertyData(defaultOutputID, &address, 0, nil, &size, &vol)
    return Int(vol * 100)
  }

  static func setVolume(_ level: Int) {
    let clamped = Float32(min(100, max(0, level))) / 100.0

    var defaultOutputID = AudioObjectID(kAudioObjectSystemObject)
    var address = AudioObjectPropertyAddress(
      mSelector: kAudioHardwarePropertyDefaultOutputDevice,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &defaultOutputID)

    var vol = clamped
    address.mSelector = kAudioDevicePropertyVolumeScalar
    address.mScope = kAudioDevicePropertyScopeOutput
    address.mElement = 1
    size = UInt32(MemoryLayout<Float32>.size)
    AudioObjectSetPropertyData(defaultOutputID, &address, 0, nil, size, &vol)
  }

  static func battery() -> (level: Int, charging: Bool, timeRemaining: Int?) {
    let snapshot = IOPSCopyPowerSourcesInfo().takeRetainedValue()
    let sources = IOPSCopyPowerSourcesList(snapshot).takeRetainedValue() as [CFTypeRef]

    for source in sources {
      guard let info = IOPSGetPowerSourceDescription(snapshot, source)?.takeUnretainedValue() as? [String: Any] else {
        continue
      }

      let capacity = info[kIOPSCurrentCapacityKey] as? Int ?? 0
      let maxCapacity = info[kIOPSMaxCapacityKey] as? Int ?? 100
      let isCharging = (info[kIOPSIsChargingKey] as? Bool) ?? false
      let timeToEmpty = info[kIOPSTimeToEmptyKey] as? Int

      let level = maxCapacity > 0 ? (capacity * 100 / maxCapacity) : 0
      return (level: level, charging: isCharging, timeRemaining: timeToEmpty)
    }

    return (level: -1, charging: false, timeRemaining: nil)
  }

  static func isDarkMode() -> Bool {
    let appearance = NSApp?.effectiveAppearance ?? NSAppearance.currentDrawing()
    return appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
  }

  static func setDarkMode(_ enabled: Bool) {
    let script = """
    tell application "System Events"
      tell appearance preferences
        set dark mode to \(enabled ? "true" : "false")
      end tell
    end tell
    """
    var error: NSDictionary?
    NSAppleScript(source: script)?.executeAndReturnError(&error)
  }

  static func lockScreen() {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/pmset")
    process.arguments = ["displaysleepnow"]
    try? process.run()
    process.waitUntilExit()
  }
}
