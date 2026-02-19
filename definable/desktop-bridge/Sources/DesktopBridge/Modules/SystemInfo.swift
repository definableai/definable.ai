import AppKit
import AudioToolbox
import CoreAudio
import CoreFoundation
import Foundation
import IOKit
import IOKit.ps

/// System information: battery, volume, hostname, OS version.
enum SystemInfo {
  // MARK: - Info

  static func getSystemInfo() -> SystemInfoResponse {
    let hostname = ProcessInfo.processInfo.hostName
    let osVersion = ProcessInfo.processInfo.operatingSystemVersionString
    let cpu = cpuModel()
    let memoryGb = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824.0

    return SystemInfoResponse(
      hostname: hostname,
      osVersion: osVersion,
      cpu: cpu,
      memoryGb: round(memoryGb * 10) / 10
    )
  }

  // MARK: - Battery

  static func getBattery() -> BatteryResponse {
    guard let powerSources = IOPSCopyPowerSourcesList(IOPSCopyPowerSourcesInfo().takeRetainedValue()).takeRetainedValue() as? [CFTypeRef],
          let source = powerSources.first,
          let description = IOPSGetPowerSourceDescription(IOPSCopyPowerSourcesInfo().takeRetainedValue(), source)?.takeUnretainedValue() as? [String: Any]
    else {
      return BatteryResponse(level: -1, charging: false, timeRemaining: -1)
    }

    let level = description[kIOPSCurrentCapacityKey] as? Int ?? -1
    let isCharging = (description[kIOPSIsChargingKey] as? Bool) ?? false
    let timeRemaining = isCharging
      ? (description[kIOPSTimeToFullChargeKey] as? Int ?? -1)
      : (description[kIOPSTimeToEmptyKey] as? Int ?? -1)

    return BatteryResponse(level: level, charging: isCharging, timeRemaining: timeRemaining)
  }

  // MARK: - Volume

  static func getVolume() -> VolumeResponse {
    var defaultOutputDeviceID = AudioDeviceID(0)
    var propertySize = UInt32(MemoryLayout<AudioDeviceID>.size)
    var address = AudioObjectPropertyAddress(
      mSelector: kAudioHardwarePropertyDefaultOutputDevice,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain
    )
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &propertySize, &defaultOutputDeviceID)

    var volume: Float32 = 0
    var volumeSize = UInt32(MemoryLayout<Float32>.size)
    var volumeAddress = AudioObjectPropertyAddress(
      mSelector: kAudioHardwareServiceDeviceProperty_VirtualMainVolume,
      mScope: kAudioDevicePropertyScopeOutput,
      mElement: kAudioObjectPropertyElementMain
    )
    AudioObjectGetPropertyData(defaultOutputDeviceID, &volumeAddress, 0, nil, &volumeSize, &volume)

    return VolumeResponse(volume: Int(volume * 100))
  }

  static func setVolume(_ level: Int) {
    var defaultOutputDeviceID = AudioDeviceID(0)
    var propertySize = UInt32(MemoryLayout<AudioDeviceID>.size)
    var address = AudioObjectPropertyAddress(
      mSelector: kAudioHardwarePropertyDefaultOutputDevice,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain
    )
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &propertySize, &defaultOutputDeviceID)

    var volume = Float32(max(0, min(100, level))) / 100.0
    var volumeSize = UInt32(MemoryLayout<Float32>.size)
    var volumeAddress = AudioObjectPropertyAddress(
      mSelector: kAudioHardwareServiceDeviceProperty_VirtualMainVolume,
      mScope: kAudioDevicePropertyScopeOutput,
      mElement: kAudioObjectPropertyElementMain
    )
    AudioObjectSetPropertyData(defaultOutputDeviceID, &volumeAddress, 0, nil, volumeSize, &volume)
  }

  // MARK: - Dark Mode

  static func getDarkMode() -> DarkModeResponse {
    let names: [NSAppearance.Name] = [.darkAqua, .aqua]
    let isDark = NSAppearance.current?.bestMatch(from: names) == NSAppearance.Name.darkAqua
    return DarkModeResponse(enabled: isDark)
  }

  static func setDarkMode(enabled: Bool) {
    let mode = enabled ? "true" : "false"
    let script = "tell application \"System Events\" to tell appearance preferences to set dark mode to \(mode)"
    _ = AppleScriptEngine.run(script: script)
  }

  static func lockScreen() {
    // Lock via screensaver (requires password-on-wake in System Settings)
    NSWorkspace.shared.open(URL(fileURLWithPath: "/System/Library/CoreServices/ScreenSaverEngine.app"))
  }

  // MARK: - Private helpers

  private static func cpuModel() -> String {
    var size = 0
    sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
    var model = [CChar](repeating: 0, count: size)
    sysctlbyname("machdep.cpu.brand_string", &model, &size, nil, 0)
    return String(cString: model)
  }
}
