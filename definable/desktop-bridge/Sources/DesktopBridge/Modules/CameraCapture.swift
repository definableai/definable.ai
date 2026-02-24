import AVFoundation
import CoreGraphics
import Foundation

actor CameraCapture {
  enum CameraError: LocalizedError, Sendable {
    case unavailable
    case microphoneUnavailable
    case permissionDenied(String)
    case captureFailed(String)
    case exportFailed(String)

    var errorDescription: String? {
      switch self {
      case .unavailable: "Camera unavailable"
      case .microphoneUnavailable: "Microphone unavailable"
      case let .permissionDenied(kind): "\(kind) permission denied"
      case let .captureFailed(msg): msg
      case let .exportFailed(msg): msg
      }
    }
  }

  struct DeviceInfo: Sendable {
    var id: String
    var name: String
    var position: String
  }

  func listDevices() -> [DeviceInfo] {
    Self.availableCameras().map { device in
      DeviceInfo(
        id: device.uniqueID,
        name: device.localizedName,
        position: Self.positionLabel(device.position))
    }
  }

  func snap(
    facing: String = "front",
    maxWidth: Int = 1600,
    quality: Double = 0.9,
    outPath: String? = nil) async throws -> (data: Data, width: Int, height: Int)
  {
    try await ensureAccess(for: .video)

    let session = AVCaptureSession()
    session.sessionPreset = .photo

    let position: AVCaptureDevice.Position = facing == "back" ? .back : .front
    guard let device = Self.pickCamera(position: position) else {
      throw CameraError.unavailable
    }

    let input = try AVCaptureDeviceInput(device: device)
    guard session.canAddInput(input) else {
      throw CameraError.captureFailed("Failed to add camera input")
    }
    session.addInput(input)

    let output = AVCapturePhotoOutput()
    guard session.canAddOutput(output) else {
      throw CameraError.captureFailed("Failed to add photo output")
    }
    session.addOutput(output)
    output.maxPhotoQualityPrioritization = .quality

    session.startRunning()
    defer { session.stopRunning() }
    try? await Task.sleep(nanoseconds: 200_000_000)  // 200ms warmup
    await waitForStabilization(device: device)

    let settings: AVCapturePhotoSettings = {
      if output.availablePhotoCodecTypes.contains(.jpeg) {
        return AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])
      }
      return AVCapturePhotoSettings()
    }()
    settings.photoQualityPrioritization = .quality

    var delegate: PhotoDelegate?
    let rawData: Data = try await withCheckedThrowingContinuation { cont in
      let d = PhotoDelegate(cont)
      delegate = d
      output.capturePhoto(with: settings, delegate: d)
    }
    withExtendedLifetime(delegate) {}

    // Transcode to JPEG at desired quality and max width
    guard let source = CGImageSourceCreateWithData(rawData as CFData, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
      throw CameraError.captureFailed("Failed to decode photo")
    }

    let scaledImage = Self.scaleImage(cgImage, maxWidth: maxWidth)
    guard let mutableData = CFDataCreateMutable(nil, 0),
          let dest = CGImageDestinationCreateWithData(mutableData, "public.jpeg" as CFString, 1, nil)
    else {
      throw CameraError.captureFailed("Failed to create JPEG destination")
    }

    let opts = [kCGImageDestinationLossyCompressionQuality: min(1.0, max(0.05, quality))] as CFDictionary
    CGImageDestinationAddImage(dest, scaledImage, opts)
    CGImageDestinationFinalize(dest)

    let jpegData = mutableData as Data

    // Write to disk if outPath was requested
    if let outPath, !outPath.isEmpty {
      let url = URL(fileURLWithPath: outPath)
      try? FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
      try jpegData.write(to: url)
    }

    return (data: jpegData, width: scaledImage.width, height: scaledImage.height)
  }

  func clip(
    facing: String = "front",
    durationMs: Int = 3000,
    includeAudio: Bool = false,
    outPath: String? = nil) async throws -> (path: String, durationMs: Int, hasAudio: Bool)
  {
    try await ensureAccess(for: .video)
    if includeAudio {
      try await ensureAccess(for: .audio)
    }

    let session = AVCaptureSession()
    session.sessionPreset = .high

    let position: AVCaptureDevice.Position = facing == "back" ? .back : .front
    guard let camera = Self.pickCamera(position: position) else {
      throw CameraError.unavailable
    }
    let cameraInput = try AVCaptureDeviceInput(device: camera)
    guard session.canAddInput(cameraInput) else {
      throw CameraError.captureFailed("Failed to add camera input")
    }
    session.addInput(cameraInput)

    if includeAudio {
      guard let mic = AVCaptureDevice.default(for: .audio) else {
        throw CameraError.microphoneUnavailable
      }
      let micInput = try AVCaptureDeviceInput(device: mic)
      guard session.canAddInput(micInput) else {
        throw CameraError.captureFailed("Failed to add microphone input")
      }
      session.addInput(micInput)
    }

    let output = AVCaptureMovieFileOutput()
    guard session.canAddOutput(output) else {
      throw CameraError.captureFailed("Failed to add movie output")
    }
    session.addOutput(output)

    let clampedDuration = min(60000, max(250, durationMs))
    output.maxRecordedDuration = CMTime(value: Int64(clampedDuration), timescale: 1000)

    session.startRunning()
    defer { session.stopRunning() }
    try? await Task.sleep(nanoseconds: 200_000_000)

    let tmpURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("definable-camera-\(UUID().uuidString).mov")
    defer { try? FileManager.default.removeItem(at: tmpURL) }

    let outputURL: URL = {
      if let outPath, !outPath.isEmpty {
        return URL(fileURLWithPath: outPath)
      }
      return FileManager.default.temporaryDirectory
        .appendingPathComponent("definable-camera-\(UUID().uuidString).mp4")
    }()
    try? FileManager.default.removeItem(at: outputURL)

    var delegate: MovieDelegate?
    let recordedURL: URL = try await withCheckedThrowingContinuation { cont in
      let d = MovieDelegate(cont)
      delegate = d
      output.startRecording(to: tmpURL, recordingDelegate: d)
    }
    withExtendedLifetime(delegate) {}

    // Export to MP4
    let asset = AVURLAsset(url: recordedURL)
    guard let export = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetMediumQuality) else {
      throw CameraError.exportFailed("Failed to create export session")
    }
    export.shouldOptimizeForNetworkUse = true
    try await export.export(to: outputURL, as: .mp4)

    return (path: outputURL.path, durationMs: clampedDuration, hasAudio: includeAudio)
  }

  // MARK: - Private

  private func ensureAccess(for mediaType: AVMediaType) async throws {
    let status = AVCaptureDevice.authorizationStatus(for: mediaType)
    switch status {
    case .authorized: return
    case .notDetermined:
      let ok = await AVCaptureDevice.requestAccess(for: mediaType)
      if !ok { throw CameraError.permissionDenied(mediaType == .video ? "Camera" : "Microphone") }
    case .denied, .restricted:
      throw CameraError.permissionDenied(mediaType == .video ? "Camera" : "Microphone")
    @unknown default:
      throw CameraError.permissionDenied(mediaType == .video ? "Camera" : "Microphone")
    }
  }

  private nonisolated static func availableCameras() -> [AVCaptureDevice] {
    var types: [AVCaptureDevice.DeviceType] = [.builtInWideAngleCamera, .continuityCamera]
    if #available(macOS 14.0, *) {
      types.append(.external)
    }
    let session = AVCaptureDevice.DiscoverySession(
      deviceTypes: types, mediaType: .video, position: .unspecified)
    return session.devices
  }

  private nonisolated static func pickCamera(position: AVCaptureDevice.Position) -> AVCaptureDevice? {
    if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) {
      return device
    }
    return AVCaptureDevice.default(for: .video)
  }

  private nonisolated static func scaleImage(_ image: CGImage, maxWidth: Int) -> CGImage {
    guard image.width > maxWidth else { return image }
    let scale = Double(maxWidth) / Double(image.width)
    let newWidth = maxWidth
    let newHeight = Int(Double(image.height) * scale)

    guard let context = CGContext(
      data: nil, width: newWidth, height: newHeight,
      bitsPerComponent: 8, bytesPerRow: 0,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { return image }

    context.interpolationQuality = .high
    context.draw(image, in: CGRect(x: 0, y: 0, width: newWidth, height: newHeight))
    return context.makeImage() ?? image
  }

  private func waitForStabilization(device: AVCaptureDevice) async {
    for _ in 0..<30 {
      if !(device.isAdjustingExposure || device.isAdjustingWhiteBalance) { return }
      try? await Task.sleep(nanoseconds: 50_000_000)
    }
  }

  private nonisolated static func positionLabel(_ position: AVCaptureDevice.Position) -> String {
    switch position {
    case .front: "front"
    case .back: "back"
    default: "unspecified"
    }
  }
}

private final class PhotoDelegate: NSObject, AVCapturePhotoCaptureDelegate {
  private var cont: CheckedContinuation<Data, Error>?
  private var didResume = false

  init(_ cont: CheckedContinuation<Data, Error>) { self.cont = cont }

  func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
    guard !didResume, let cont else { return }
    didResume = true
    self.cont = nil
    if let error { cont.resume(throwing: error); return }
    guard let data = photo.fileDataRepresentation(), !data.isEmpty else {
      cont.resume(throwing: CameraCapture.CameraError.captureFailed("No photo data"))
      return
    }
    cont.resume(returning: data)
  }
}

private final class MovieDelegate: NSObject, AVCaptureFileOutputRecordingDelegate {
  private var cont: CheckedContinuation<URL, Error>?

  init(_ cont: CheckedContinuation<URL, Error>) { self.cont = cont }

  func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL,
                  from connections: [AVCaptureConnection], error: Error?)
  {
    guard let cont else { return }
    self.cont = nil
    if let error {
      let ns = error as NSError
      if ns.domain == AVFoundationErrorDomain, ns.code == AVError.maximumDurationReached.rawValue {
        cont.resume(returning: outputFileURL)
        return
      }
      cont.resume(throwing: error)
      return
    }
    cont.resume(returning: outputFileURL)
  }
}
