import Foundation

/// File operations bridged to the agent.
enum FileBridge {
  static func listFiles(path: String, recursive: Bool) throws -> FileListResponse {
    let fm = FileManager.default
    guard fm.fileExists(atPath: path) else {
      throw BridgeError.notFound("Path does not exist: '\(path)'")
    }

    if recursive {
      guard let enumerator = fm.enumerator(atPath: path) else {
        throw BridgeError.operationFailed("Cannot enumerate '\(path)'")
      }
      var files: [FileEntry] = []
      while let relative = enumerator.nextObject() as? String {
        let full = (path as NSString).appendingPathComponent(relative)
        files.append(makeEntry(path: full, name: relative))
      }
      return FileListResponse(files: files)
    }

    let names = try fm.contentsOfDirectory(atPath: path)
    let files = names.map { name -> FileEntry in
      let full = (path as NSString).appendingPathComponent(name)
      return makeEntry(path: full, name: name)
    }
    return FileListResponse(files: files)
  }

  static func readFile(path: String) throws -> FileContentResponse {
    guard let content = try? String(contentsOfFile: path, encoding: .utf8) else {
      throw BridgeError.operationFailed("Cannot read file '\(path)'")
    }
    return FileContentResponse(content: content)
  }

  static func writeFile(path: String, content: String) throws {
    do {
      try content.write(toFile: path, atomically: true, encoding: .utf8)
    } catch {
      throw BridgeError.operationFailed("Cannot write file '\(path)': \(error)")
    }
  }

  static func moveFile(from: String, to: String) throws {
    do {
      try FileManager.default.moveItem(atPath: from, toPath: to)
    } catch {
      throw BridgeError.operationFailed("Cannot move '\(from)' → '\(to)': \(error)")
    }
  }

  static func deleteFile(path: String, toTrash: Bool) throws {
    let fm = FileManager.default
    if toTrash {
      var url = URL(fileURLWithPath: path)
      do {
        var trashURL: NSURL?
        try fm.trashItem(at: url, resultingItemURL: &trashURL)
      } catch {
        throw BridgeError.operationFailed("Cannot move to trash '\(path)': \(error)")
      }
    } else {
      do {
        try fm.removeItem(atPath: path)
      } catch {
        throw BridgeError.operationFailed("Cannot delete '\(path)': \(error)")
      }
    }
  }

  static func fileInfo(path: String) throws -> FileInfoResponse {
    let fm = FileManager.default
    guard let attrs = try? fm.attributesOfItem(atPath: path) else {
      throw BridgeError.notFound("File not found: '\(path)'")
    }
    let size = (attrs[.size] as? Int64) ?? 0
    let created = isoDate(attrs[.creationDate] as? Date)
    let modified = isoDate(attrs[.modificationDate] as? Date)
    let kind = (attrs[.type] as? FileAttributeType) == .typeDirectory ? "folder" : "file"
    return FileInfoResponse(size: size, created: created, modified: modified, kind: kind)
  }

  // MARK: - Private helpers

  private static func makeEntry(path: String, name: String) -> FileEntry {
    let fm = FileManager.default
    let attrs = try? fm.attributesOfItem(atPath: path)
    let size = (attrs?[.size] as? Int64) ?? 0
    let isDir = (attrs?[.type] as? FileAttributeType) == .typeDirectory
    return FileEntry(name: name, path: path, kind: isDir ? "folder" : "file", size: size)
  }

  private static func isoDate(_ date: Date?) -> String {
    guard let date = date else { return "" }
    let formatter = ISO8601DateFormatter()
    return formatter.string(from: date)
  }
}
