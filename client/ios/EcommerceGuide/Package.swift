// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "EcommerceGuide",
    platforms: [
        .iOS(.v17),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "EcommerceGuide",
            targets: ["EcommerceGuide"]
        )
    ],
    targets: [
        .target(
            name: "EcommerceGuide",
            resources: [
                .process("Audio"),
                .process("Fixtures")
            ]
        ),
        .testTarget(
            name: "EcommerceGuideTests",
            dependencies: ["EcommerceGuide"]
        )
    ]
)
