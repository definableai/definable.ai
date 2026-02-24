import AVFoundation
import Foundation
import ScreenCaptureKit

@MainActor
final class ScreenRecorder {
  enum RecordError: LocalizedError {
    case noDisplays
    case invalidScreenIndex(Int)
    case noFramesCaptured
    case writeFailed(String)

    var errorDescription: String? {
      switch self {
      case .noDisplays: "No displays available"
      case let .invalidScreenIndex(idx): "Invalid screen index \(idx)"
      case .noFramesCaptured: "No frames captured"
      case let .writeFailed(msg): msg
      }
    }
  }

  func record(
    screenIndex: Int? = nil,
    durationMs: Int? = nil,
    fps: Double? = nil,
    includeAudio: Bool = false,
    outPath: String? = nil) async throws -> (path: String, hasAudio: Bool)
  {
    let durationMs = Self.clampDuration(durationMs)
    let fps = Self.clampFps(fps)

    let outURL: URL = {
      if let outPath, !outPath.isEmpty {
        return URL(fileURLWithPath: outPath)
      }
      return FileManager.default.temporaryDirectory
        .appendingPathComponent("definable-screen-\(UUID().uuidString).mp4")
    }()
    try? FileManager.default.removeItem(at: outURL)

    let content = try await SCShareableContent.current
    let displays = content.displays.sorted { $0.displayID < $1.displayID }
    guard !displays.isEmpty else { throw RecordError.noDisplays }

    let idx = screenIndex ?? 0
    guard idx >= 0, idx < displays.count else { throw RecordError.invalidScreenIndex(idx) }
    let display = displays[idx]

    let filter = SCContentFilter(display: display, excludingWindows: [])
    let config = SCStreamConfiguration()
    config.width = display.width
    config.height = display.height
    config.queueDepth = 8
    config.showsCursor = true
    config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(max(1, Int32(fps.rounded()))))
    if includeAudio {
      config.capturesAudio = true
    }

    let recorder = try VideoWriter(
      outputURL: outURL,
      width: display.width,
      height: display.height,
      includeAudio: includeAudio)

    let stream = SCStream(filter: filter, configuration: config, delegate: recorder)
    try stream.addStreamOutput(recorder, type: .screen, sampleHandlerQueue: recorder.queue)
    if includeAudio {
      try stream.addStreamOutput(recorder, type: .audio, sampleHandlerQueue: recorder.queue)
    }

    var started = false
    do {
      try await stream.startCapture()
      started = true
      try await Task.sleep(nanoseconds: UInt64(durationMs) * 1_000_000)
      try await stream.stopCapture()
    } catch {
      if started { try? await stream.stopCapture() }
      throw error
    }

    try await recorder.finish()
    return (path: outURL.path, hasAudio: recorder.hasAudio)
  }

  private nonisolated static func clampDuration(_ ms: Int?) -> Int {
    min(60000, max(250, ms ?? 10000))
  }

  private nonisolated static func clampFps(_ fps: Double?) -> Double {
    let v = fps ?? 10
    return v.isFinite ? min(60, max(1, v)) : 10
  }
}

private final class VideoWriter: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
  let queue = DispatchQueue(label: "ai.definable.screenRecord.writer")
  let hasAudio: Bool

  private let writer: AVAssetWriter
  private let videoInput: AVAssetWriterInput
  private let audioInput: AVAssetWriterInput?
  private var started = false
  private var sawFrame = false
  private var didFinish = false
  private var pendingError: String?

  init(outputURL: URL, width: Int, height: Int, includeAudio: Bool) throws {
    self.writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)

    let videoSettings: [String: Any] = [
      AVVideoCodecKey: AVVideoCodecType.h264,
      AVVideoWidthKey: width,
      AVVideoHeightKey: height,
    ]
    self.videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
    self.videoInput.expectsMediaDataInRealTime = true
    guard self.writer.canAdd(self.videoInput) else {
      throw ScreenRecorder.RecordError.writeFailed("Cannot add video input")
    }
    self.writer.add(self.videoInput)

    if includeAudio {
      let audioSettings: [String: Any] = [
        AVFormatIDKey: kAudioFormatMPEG4AAC,
        AVNumberOfChannelsKey: 1,
        AVSampleRateKey: 44100,
        AVEncoderBitRateKey: 96000,
      ]
      let input = AVAssetWriterInput(mediaType: .audio, outputSettings: audioSettings)
      input.expectsMediaDataInRealTime = true
      if self.writer.canAdd(input) {
        self.writer.add(input)
        self.audioInput = input
        self.hasAudio = true
      } else {
        self.audioInput = nil
        self.hasAudio = false
      }
    } else {
      self.audioInput = nil
      self.hasAudio = false
    }
    super.init()
  }

  func stream(_ stream: SCStream, didStopWithError error: Error) {
    queue.async { self.pendingError = error.localizedDescription }
  }

  func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
    guard CMSampleBufferDataIsReady(sampleBuffer) else { return }
    switch type {
    case .screen: handleVideo(sampleBuffer)
    case .audio: handleAudio(sampleBuffer)
    default: break
    }
  }

  private func handleVideo(_ buffer: CMSampleBuffer) {
    if pendingError != nil || didFinish { return }
    if !started {
      guard writer.startWriting() else {
        pendingError = writer.error?.localizedDescription ?? "Failed to start writer"
        return
      }
      writer.startSession(atSourceTime: CMSampleBufferGetPresentationTimeStamp(buffer))
      started = true
    }
    sawFrame = true
    if videoInput.isReadyForMoreMediaData { _ = videoInput.append(buffer) }
  }

  private func handleAudio(_ buffer: CMSampleBuffer) {
    guard let audioInput, pendingError == nil, !didFinish, started else { return }
    if audioInput.isReadyForMoreMediaData { _ = audioInput.append(buffer) }
  }

  func finish() async throws {
    try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
      queue.async {
        if let msg = self.pendingError {
          cont.resume(throwing: ScreenRecorder.RecordError.writeFailed(msg)); return
        }
        guard self.started, self.sawFrame else {
          cont.resume(throwing: ScreenRecorder.RecordError.noFramesCaptured); return
        }
        if self.didFinish { cont.resume(); return }
        self.didFinish = true
        self.videoInput.markAsFinished()
        self.audioInput?.markAsFinished()
        self.writer.finishWriting {
          if let err = self.writer.error {
            cont.resume(throwing: ScreenRecorder.RecordError.writeFailed(err.localizedDescription))
          } else {
            cont.resume()
          }
        }
      }
    }
  }
}
