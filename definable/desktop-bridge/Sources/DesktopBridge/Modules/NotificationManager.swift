import Foundation
import UserNotifications

/// Sends macOS notifications via UNUserNotificationCenter.
enum NotificationManager {
  static func send(title: String, message: String) {
    let center = UNUserNotificationCenter.current()

    // Request permission (best-effort; bridge may already have it)
    center.requestAuthorization(options: [.alert, .sound]) { _, _ in }

    let content = UNMutableNotificationContent()
    content.title = title
    content.body = message
    content.sound = .default

    let id = UUID().uuidString
    let request = UNNotificationRequest(identifier: id, content: content, trigger: nil)
    center.add(request) { _ in }
  }
}
