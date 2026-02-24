import Foundation

enum FileBridge {
  struct FileEntry {
    var name: String
    var path: String
    var isDirectory: Bool
    var size: Int?
  }

  struct FileInfo {
    var size: Int
    var created: String?
    var modified: String?
    var kind: String
  }

  private static let dateFormatter: ISO8601DateFormatter = {
    let f = ISO8601DateFormatter()
    f.formatOptions = [.withInternetDateTime]
    return f
  }()

  static func listFiles(path: String, recursive: Bool = false) throws -> [FileEntry] {
    let url = URL(fileURLWithPath: path)
    let fm = FileManager.default
    let keys: [URLResourceKey] = [.isDirectoryKey, .fileSizeKey]

    if recursive {
      guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: keys) else {
        return []
      }
      var results: [FileEntry] = []
      for case let fileURL as URL in enumerator {
        let values = try fileURL.resourceValues(forKeys: Set(keys))
        results.append(FileEntry(
          name: fileURL.lastPathComponent,
          path: fileURL.path,
          isDirectory: values.isDirectory ?? false,
          size: values.fileSize))
      }
      return results
    } else {
      let contents = try fm.contentsOfDirectory(at: url, includingPropertiesForKeys: keys)
      return try contents.map { fileURL in
        let values = try fileURL.resourceValues(forKeys: Set(keys))
        return FileEntry(
          name: fileURL.lastPathComponent,
          path: fileURL.path,
          isDirectory: values.isDirectory ?? false,
          size: values.fileSize)
      }
    }
  }

  static func readFile(path: String) throws -> String {
    try String(contentsOfFile: path, encoding: .utf8)
  }

  static func writeFile(path: String, content: String) throws {
    try content.write(toFile: path, atomically: true, encoding: .utf8)
  }

  static func moveFile(from: String, to: String) throws {
    try FileManager.default.moveItem(atPath: from, toPath: to)
  }

  static func deleteFile(path: String, toTrash: Bool = true) throws {
    let url = URL(fileURLWithPath: path)
    if toTrash {
      try FileManager.default.trashItem(at: url, resultingItemURL: nil)
    } else {
      try FileManager.default.removeItem(at: url)
    }
  }

  static func fileInfo(path: String) throws -> FileInfo {
    let attrs = try FileManager.default.attributesOfItem(atPath: path)
    let size = (attrs[.size] as? Int) ?? 0
    let created = (attrs[.creationDate] as? Date).map { dateFormatter.string(from: $0) }
    let modified = (attrs[.modificationDate] as? Date).map { dateFormatter.string(from: $0) }
    let isDir = (attrs[.type] as? FileAttributeType) == .typeDirectory
    let kind = isDir ? "directory" : "file"
    return FileInfo(size: size, created: created, modified: modified, kind: kind)
  }
}
