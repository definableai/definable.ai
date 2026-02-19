// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "DesktopBridge",
  platforms: [.macOS(.v13)],
  dependencies: [
    .package(url: "https://github.com/vapor/vapor.git", from: "4.89.0"),
  ],
  targets: [
    .executableTarget(
      name: "DesktopBridge",
      dependencies: [
        .product(name: "Vapor", package: "vapor"),
      ],
      path: "Sources/DesktopBridge"
    ),
  ]
)
