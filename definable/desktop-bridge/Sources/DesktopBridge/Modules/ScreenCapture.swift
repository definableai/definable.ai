import CoreGraphics
import Foundation
import ScreenCaptureKit
import Vision
import AppKit

actor ScreenCapture {
  enum CaptureError: LocalizedError {
    case noDisplays
    case captureFailed(String)
    case ocrFailed(String)

    var errorDescription: String? {
      switch self {
      case .noDisplays: "No displays available"
      case let .captureFailed(msg): msg
      case let .ocrFailed(msg): msg
      }
    }
  }

  struct OCRResult {
    var text: String
    var elements: [OCRElement]
  }

  struct OCRElement {
    var text: String
    var x: Double
    var y: Double
    var width: Double
    var height: Double
    var confidence: Double
  }

  struct TextLocation {
    var x: Double
    var y: Double
    var width: Double
    var height: Double
    var centerX: Double
    var centerY: Double
  }

  func captureScreen(display: Int = 0, maxWidth: Int = 512, region: CGRect? = nil) async throws -> (data: Data, width: Int, height: Int) {
    let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
    let displays = content.displays.sorted { $0.displayID < $1.displayID }
    guard !displays.isEmpty else { throw CaptureError.noDisplays }
    guard display >= 0, display < displays.count else {
      throw CaptureError.captureFailed("Invalid display index \(display)")
    }

    let scDisplay = displays[display]
    let filter = SCContentFilter(display: scDisplay, excludingWindows: [])

    let config = SCStreamConfiguration()
    let scale = Double(maxWidth) / Double(scDisplay.width)
    config.width = maxWidth
    config.height = Int(Double(scDisplay.height) * scale)
    config.showsCursor = true
    config.captureResolution = .nominal

    let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)

    let cgImage: CGImage
    if let region = region {
      guard let cropped = image.cropping(to: region) else {
        throw CaptureError.captureFailed("Failed to crop image")
      }
      cgImage = cropped
    } else {
      cgImage = image
    }

    let rep = NSBitmapImageRep(cgImage: cgImage)
    guard let jpegData = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.85]) else {
      throw CaptureError.captureFailed("Failed to encode JPEG")
    }

    return (data: jpegData, width: cgImage.width, height: cgImage.height)
  }

  func ocrScreen(region: CGRect? = nil) async throws -> OCRResult {
    // Capture at full resolution for OCR
    let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
    guard let display = content.displays.first else { throw CaptureError.noDisplays }

    let filter = SCContentFilter(display: display, excludingWindows: [])
    let config = SCStreamConfiguration()
    config.width = display.width * 2  // Retina
    config.height = display.height * 2
    config.showsCursor = false
    config.captureResolution = .best

    let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)

    let cgImage: CGImage
    if let region = region {
      // Scale region to retina
      let scale = Double(image.width) / Double(display.width)
      let scaledRegion = CGRect(
        x: region.origin.x * scale,
        y: region.origin.y * scale,
        width: region.size.width * scale,
        height: region.size.height * scale)
      guard let cropped = image.cropping(to: scaledRegion) else {
        throw CaptureError.ocrFailed("Failed to crop for OCR")
      }
      cgImage = cropped
    } else {
      cgImage = image
    }

    let displayWidth = Double(display.width)
    let displayHeight = Double(display.height)

    return try await withCheckedThrowingContinuation { cont in
      let request = VNRecognizeTextRequest { req, error in
        if let error {
          cont.resume(throwing: CaptureError.ocrFailed(error.localizedDescription))
          return
        }

        guard let observations = req.results as? [VNRecognizedTextObservation] else {
          cont.resume(returning: OCRResult(text: "", elements: []))
          return
        }

        var fullText = ""
        var elements: [OCRElement] = []

        for obs in observations {
          guard let candidate = obs.topCandidates(1).first else { continue }
          let text = candidate.string
          fullText += text + "\n"

          let box = obs.boundingBox
          // Convert Vision coords (bottom-left origin, normalized) to screen coords (top-left origin)
          let x = box.origin.x * displayWidth
          let y = (1.0 - box.origin.y - box.size.height) * displayHeight
          let w = box.size.width * displayWidth
          let h = box.size.height * displayHeight

          elements.append(OCRElement(
            text: text,
            x: x, y: y, width: w, height: h,
            confidence: Double(candidate.confidence)))
        }

        cont.resume(returning: OCRResult(
          text: fullText.trimmingCharacters(in: .whitespacesAndNewlines),
          elements: elements))
      }
      request.recognitionLevel = .accurate
      request.usesLanguageCorrection = true

      let handler = VNImageRequestHandler(cgImage: cgImage)
      do {
        try handler.perform([request])
      } catch {
        cont.resume(throwing: CaptureError.ocrFailed(error.localizedDescription))
      }
    }
  }

  func findText(_ searchText: String, nth: Int = 0) async throws -> TextLocation? {
    let result = try await ocrScreen()
    let lowered = searchText.lowercased()
    var matches: [TextLocation] = []

    for element in result.elements {
      if element.text.lowercased().contains(lowered) {
        matches.append(TextLocation(
          x: element.x,
          y: element.y,
          width: element.width,
          height: element.height,
          centerX: element.x + element.width / 2,
          centerY: element.y + element.height / 2))
      }
    }

    guard nth < matches.count else { return nil }
    return matches[nth]
  }
}
