import Foundation
import UserNotifications

enum NotificationService {
  static func send(title: String, message: String, sound: String? = nil) async -> Bool {
    let center = UNUserNotificationCenter.current()
    let settings = await center.notificationSettings()

    if settings.authorizationStatus == .notDetermined {
      let granted = (try? await center.requestAuthorization(options: [.alert, .sound, .badge])) ?? false
      if !granted { return false }
    } else if settings.authorizationStatus != .authorized {
      return false
    }

    let content = UNMutableNotificationContent()
    content.title = title
    content.body = message
    if let sound, !sound.isEmpty {
      content.sound = UNNotificationSound(named: UNNotificationSoundName(sound))
    } else {
      content.sound = .default
    }

    let request = UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)
    do {
      try await center.add(request)
      return true
    } catch {
      return false
    }
  }
}
