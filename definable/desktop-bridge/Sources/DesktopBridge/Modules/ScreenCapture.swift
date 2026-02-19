import CoreGraphics
import Foundation
import ScreenCaptureKit
import Vision

/// Screen capture and OCR via CGWindowListCreateImage and Vision framework.
enum ScreenCapture {
  // MARK: - Capture

  /// Capture a display, scale down to ``maxWidth`` if needed, and encode as JPEG.
  ///
  /// Full-retina screenshots can be 5–15 MB as PNG. JPEG at 85 % quality is
  /// 10–20× smaller, which keeps tool results within LLM context budgets.
  static func captureDisplay(_ displayIndex: Int = 0, region: CGRect? = nil, maxWidth: Int = 512) throws -> Data {
    let displays = DisplayList.all()
    guard displayIndex < displays.count else {
      throw BridgeError.invalidInput("Display index \(displayIndex) out of range (\(displays.count) displays)")
    }
    let displayID = displays[displayIndex]

    let captureRect = region ?? CGDisplayBounds(displayID)
    let cgImage = CGDisplayCreateImage(displayID, rect: captureRect)
    guard var image = cgImage else {
      throw BridgeError.operationFailed("Failed to capture display \(displayIndex)")
    }

    if image.width > maxWidth {
      image = scaled(image: image, maxWidth: maxWidth)
    }

    return try jpegData(from: image)
  }

  // MARK: - OCR

  static func ocrDisplay(_ displayIndex: Int = 0, region: CGRect? = nil) throws -> OCRResponse {
    // OCR uses full native resolution — do NOT downscale before recognition.
    guard let rawImage = captureRaw(displayIndex, region: region) else {
      throw BridgeError.operationFailed("Failed to capture display \(displayIndex)")
    }
    return try performOCR(on: rawImage, offset: region?.origin ?? .zero)
  }

  static func findText(_ query: String, displayIndex: Int = 0, nth: Int = 0) throws -> FindTextResponse {
    let result = try ocrDisplay(displayIndex)
    let matches = result.elements.filter { $0.text.lowercased().contains(query.lowercased()) }
    if matches.count > nth {
      return FindTextResponse(found: true, bounds: matches[nth].bounds)
    }
    return FindTextResponse(found: false, bounds: nil)
  }

  // MARK: - Private helpers

  private static func performOCR(on image: CGImage, offset: CGPoint) throws -> OCRResponse {
    var fullText = ""
    var elements: [OCRElement] = []
    let semaphore = DispatchSemaphore(value: 0)
    var ocrError: Error?

    let request = VNRecognizeTextRequest { req, err in
      defer { semaphore.signal() }
      if let err = err { ocrError = err; return }
      guard let observations = req.results as? [VNRecognizedTextObservation] else { return }
      let imageW = Double(image.width)
      let imageH = Double(image.height)
      var lines: [String] = []
      for obs in observations {
        guard let top = obs.topCandidates(1).first else { continue }
        lines.append(top.string)
        // Vision uses bottom-left origin; convert to top-left screen coords
        let box = obs.boundingBox
        let x = box.minX * imageW + offset.x
        let y = (1 - box.maxY) * imageH + offset.y
        let w = box.width * imageW
        let h = box.height * imageH
        elements.append(OCRElement(text: top.string, bounds: BoundsResponse(x: x, y: y, width: w, height: h)))
      }
      fullText = lines.joined(separator: "\n")
    }
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true

    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    do {
      try handler.perform([request])
    } catch {
      throw BridgeError.operationFailed("OCR failed: \(error)")
    }
    semaphore.wait()
    if let err = ocrError { throw BridgeError.operationFailed("OCR error: \(err)") }

    return OCRResponse(text: fullText, elements: elements)
  }

  /// Encode a CGImage as JPEG at 85 % quality (~10–20× smaller than PNG for screen content).
  private static func jpegData(from image: CGImage, quality: CGFloat = 0.85) throws -> Data {
    let mutableData = NSMutableData()
    guard
      let dest = CGImageDestinationCreateWithData(
        mutableData as CFMutableData, "public.jpeg" as CFString, 1, nil)
    else {
      throw BridgeError.operationFailed("Failed to create JPEG destination")
    }
    let options = [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary
    CGImageDestinationAddImage(dest, image, options)
    guard CGImageDestinationFinalize(dest) else {
      throw BridgeError.operationFailed("Failed to encode JPEG")
    }
    return mutableData as Data
  }

  private static func pngData(from image: CGImage) throws -> Data {
    let mutableData = NSMutableData()
    guard
      let dest = CGImageDestinationCreateWithData(mutableData as CFMutableData, "public.png" as CFString, 1, nil)
    else {
      throw BridgeError.operationFailed("Failed to create PNG destination")
    }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else {
      throw BridgeError.operationFailed("Failed to encode PNG")
    }
    return mutableData as Data
  }

  /// Downscale a CGImage proportionally so its width is at most ``maxWidth``.
  private static func scaled(image: CGImage, maxWidth: Int) -> CGImage {
    let origW = image.width
    let origH = image.height
    let scale = Double(maxWidth) / Double(origW)
    let newW = maxWidth
    let newH = Int(Double(origH) * scale)

    let colorSpace = image.colorSpace ?? CGColorSpaceCreateDeviceRGB()
    guard
      let ctx = CGContext(
        data: nil,
        width: newW,
        height: newH,
        bitsPerComponent: 8,
        bytesPerRow: 0,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
      )
    else {
      return image  // fallback: return original if context creation fails
    }
    ctx.interpolationQuality = .high
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: newW, height: newH))
    return ctx.makeImage() ?? image
  }

  /// Capture a display at native resolution without any scaling or encoding.
  private static func captureRaw(_ displayIndex: Int, region: CGRect?) -> CGImage? {
    let displays = DisplayList.all()
    guard displayIndex < displays.count else { return nil }
    let displayID = displays[displayIndex]
    let captureRect = region ?? CGDisplayBounds(displayID)
    return CGDisplayCreateImage(displayID, rect: captureRect)
  }
}

// MARK: - Display enumeration helper

enum DisplayList {
  static func all() -> [CGDirectDisplayID] {
    var displayCount: UInt32 = 0
    CGGetActiveDisplayList(0, nil, &displayCount)
    var displays = [CGDirectDisplayID](repeating: kCGNullDirectDisplay, count: Int(displayCount))
    CGGetActiveDisplayList(displayCount, &displays, &displayCount)
    return displays
  }
}
