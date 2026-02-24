// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "DesktopBridge",
  platforms: [.macOS(.v14)],
  dependencies: [
    .package(url: "https://github.com/vapor/vapor.git", from: "4.99.0"),
  ],
  targets: [
    .executableTarget(
      name: "DesktopBridge",
      dependencies: [
        .product(name: "Vapor", package: "vapor"),
      ],
      linkerSettings: [
        .linkedFramework("ApplicationServices"),
        .linkedFramework("AVFoundation"),
        .linkedFramework("CoreGraphics"),
        .linkedFramework("CoreLocation"),
        .linkedFramework("ScreenCaptureKit"),
        .linkedFramework("Speech"),
        .linkedFramework("Vision"),
      ]
    ),
  ]
)
