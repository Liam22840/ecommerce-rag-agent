import SwiftUI
import PhotosUI
#if canImport(UIKit)
import UIKit
#endif

@available(iOS 17.0, macOS 13.0, *)
public struct ShoppingConciergeRootView: View {
    @StateObject private var viewModel: ChatViewModel
    @State private var screen: ShoppingFlowScreen = .onboarding
    @State private var forward = true

    @MainActor
    public init(service: any ChatService = MockChatService()) {
        _viewModel = StateObject(wrappedValue: ChatViewModel(service: service))
    }

    public var body: some View {
        ZStack {
            switch screen {
            case .onboarding:
                OnboardingScreen {
                    navigate(to: .chat, forward: true)
                }
                .transition(pushTransition)
            case .chat:
                ChatScreen(
                    viewModel: viewModel,
                    cameraAction: { navigate(to: .photoSearch, forward: true) },
                    checkoutAction: { navigate(to: .orderReview, forward: true) }
                )
                .transition(pushTransition)
            case .photoSearch:
                PhotoSearchScreen(
                    backAction: { navigate(to: .chat, forward: false) },
                    captureAction: { imageData, caption in
                        viewModel.sendPhoto(imageData: imageData, caption: caption)
                        navigate(to: .chat, forward: false)
                    }
                )
                .transition(pushTransition)
            case .orderReview:
                OrderReviewScreen(
                    items: viewModel.cartItems,
                    backAction: { navigate(to: .chat, forward: false) },
                    confirmAction: { navigate(to: .orderSuccess, forward: true) }
                )
                .transition(pushTransition)
            case .orderSuccess:
                OrderSuccessScreen {
                    navigate(to: .chat, forward: false)
                }
                .transition(pushTransition)
            }
        }
    }

    private var pushTransition: AnyTransition {
        .push(from: forward ? .trailing : .leading)
    }

    private func navigate(to target: ShoppingFlowScreen, forward isForward: Bool) {
        // Both mutations belong to one transaction so an in-flight transition
        // can't observe a direction from a different navigation event.
        withAnimation(.spring(response: 0.38, dampingFraction: 0.86)) {
            forward = isForward
            screen = target
        }
    }
}

@available(iOS 17.0, macOS 13.0, *)
private enum ShoppingFlowScreen {
    case onboarding
    case chat
    case photoSearch
    case orderReview
    case orderSuccess
}

@available(iOS 17.0, macOS 13.0, *)
private struct OnboardingScreen: View {
    let startAction: () -> Void

    @State private var appeared = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 48)

            ZStack {
                RoundedRectangle(cornerRadius: 32, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [
                                Color(red: 0.969, green: 0.914, blue: 0.886),
                                Color(red: 0.941, green: 0.890, blue: 0.863)
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .shadow(color: GuideTheme.accentShadow.opacity(0.42), radius: 28, y: 10)

                Image(systemName: "sparkles")
                    .font(.system(size: 52, weight: .bold))
                    .foregroundStyle(GuideTheme.accent)
            }
            .frame(width: 160, height: 160)
            .padding(.bottom, 32)
            .scaleEffect(appeared ? 1 : 0.7)
            .opacity(appeared ? 1 : 0)
            .animation(GuideMotion.entrance(reduceMotion: reduceMotion), value: appeared)

            Text("你的AI购物管家")
                .font(.system(size: 26, weight: .bold))
                .foregroundStyle(GuideTheme.inkStrong)
                .multilineTextAlignment(.center)
                .opacity(appeared ? 1 : 0)
                .offset(y: appeared ? 0 : 14)
                .animation(GuideMotion.entrance(reduceMotion: reduceMotion).delay(reduceMotion ? 0 : 0.12), value: appeared)

            Text("智能推荐 · 拍照找货 · 对比决策 · 一键下单")
                .font(.subheadline)
                .foregroundStyle(GuideTheme.secondaryInk)
                .multilineTextAlignment(.center)
                .padding(.top, 10)
                .opacity(appeared ? 1 : 0)
                .offset(y: appeared ? 0 : 14)
                .animation(GuideMotion.entrance(reduceMotion: reduceMotion).delay(reduceMotion ? 0 : 0.2), value: appeared)

            Text("告诉我你想买什么，我来帮你找到最合适的商品")
                .font(.footnote)
                .foregroundStyle(GuideTheme.tertiaryInk)
                .multilineTextAlignment(.center)
                .lineSpacing(2)
                .padding(.top, 6)
                .padding(.horizontal, 44)
                .opacity(appeared ? 1 : 0)
                .offset(y: appeared ? 0 : 14)
                .animation(GuideMotion.entrance(reduceMotion: reduceMotion).delay(reduceMotion ? 0 : 0.28), value: appeared)

            Button(action: startAction) {
                Text("开始对话")
                    .font(.headline.weight(.semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 56)
                    .frame(height: 48)
                    .background(GuideTheme.accent)
                    .clipShape(Capsule())
                    .shadow(color: GuideTheme.accentShadow, radius: 14, y: 4)
            }
            .buttonStyle(PressableButtonStyle())
            .padding(.top, 36)
            .opacity(appeared ? 1 : 0)
            .offset(y: appeared ? 0 : 14)
            .animation(GuideMotion.entrance(reduceMotion: reduceMotion).delay(reduceMotion ? 0 : 0.36), value: appeared)

            Text("已有 128万+ 用户正在使用")
                .font(.caption2)
                .foregroundStyle(GuideTheme.tertiaryInk)
                .padding(.top, 16)

            Spacer(minLength: 48)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background {
            LinearGradient(
                colors: [GuideTheme.accentSoft, GuideTheme.panelBackground],
                startPoint: .top,
                endPoint: .center
            )
            .ignoresSafeArea()
        }
        .onAppear {
            appeared = true
        }
    }
}

@available(iOS 17.0, macOS 13.0, *)
private struct PhotoSearchScreen: View {
    let backAction: () -> Void
    let captureAction: (Data, String) -> Void

    @State private var pickerItem: PhotosPickerItem?
    @State private var isCameraPresented = false
    @State private var isSearching = false
    // Once an image is picked it's staged here (not sent) so the shopper can add a caption first.
    // The preview is decoded once at stage time, not in `body`, so typing the caption doesn't re-decode it.
    @State private var stagedImage: Data?
    @State private var stagedPreview: Image?
    @State private var caption: String = ""

    // The Simulator has no camera, so the shutter is only live on a real device.
    private var cameraAvailable: Bool {
        #if os(iOS)
        return UIImagePickerController.isSourceTypeAvailable(.camera)
        #else
        return false
        #endif
    }

    var body: some View {
        VStack(spacing: 0) {
            header

            if let stagedImage {
                reviewView(imageData: stagedImage)
            } else {
                ZStack {
                    viewfinder

                    if isSearching {
                        searchingOverlay
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                controls
            }
        }
        .background(Color.black.ignoresSafeArea())
        .foregroundStyle(.white)
        .keyboardDismissToolbar()
        .cameraCover(isPresented: $isCameraPresented) { data in
            stage(data)
        }
    }

    // Stage a picked/captured photo: normalise to JPEG and decode the preview once (not on each render).
    private func stage(_ data: Data) {
        let jpeg = normalizedJPEG(data)
        stagedImage = jpeg
        stagedPreview = platformImage(data: jpeg)
    }

    // After a photo is chosen: preview it, let the shopper type an optional caption ("我想要同款外套"),
    // and only send on tap so the image and the words go together.
    private func reviewView(imageData: Data) -> some View {
        VStack(spacing: 18) {
            if let image = stagedPreview {
                image
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 360)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            }

            TextField("想找什么？例如：我想要同款外套", text: $caption)
                .textFieldStyle(.plain)
                .foregroundStyle(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 11)
                .background(Color.white.opacity(0.12), in: RoundedRectangle(cornerRadius: 12, style: .continuous))

            HStack(spacing: 14) {
                Button {
                    stagedImage = nil
                    stagedPreview = nil
                    caption = ""
                    pickerItem = nil
                } label: {
                    Text("重选")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .frame(height: 48)
                        .background(Color.white.opacity(0.12), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                }
                .buttonStyle(.plain)

                Button {
                    dismissKeyboard()
                    captureAction(imageData, caption)
                } label: {
                    Text("发送")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .frame(height: 48)
                        .background(GuideTheme.accent, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                }
                .buttonStyle(.plain)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
    }

    private var header: some View {
        HStack {
            Button(action: backAction) {
                Image(systemName: "chevron.left")
                    .font(.system(size: 21, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 36, height: 36)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("返回")

            Spacer()

            Text("拍照找同款")
                .font(.headline.weight(.semibold))

            Spacer()

            Color.clear
                .frame(width: 36, height: 36)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var viewfinder: some View {
        VStack(spacing: 0) {
            ZStack {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(Color.white.opacity(0.5), lineWidth: 2)

                ForEach(PhotoCorner.allCases, id: \.self) { corner in
                    PhotoSearchCorner(corner: corner)
                        .stroke(GuideTheme.accent, style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
                }
            }
            .aspectRatio(3.0 / 4.0, contentMode: .fit)
            .frame(maxWidth: 280)

            Text("将商品对准框内")
                .font(.footnote)
                .foregroundStyle(.white.opacity(0.62))
                .padding(.top, 14)
        }
        .padding(.horizontal, 48)
    }

    private var searchingOverlay: some View {
        VStack(spacing: 16) {
            ProgressView()
                .controlSize(.large)
                .tint(GuideTheme.accent)

            Text("正在识别商品...")
                .font(.subheadline.weight(.medium))
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black.opacity(0.62))
    }

    private var controls: some View {
        HStack(spacing: 36) {
            // Pick an existing photo (works on Simulator and device).
            PhotosPicker(selection: $pickerItem, matching: .images) {
                Image(systemName: "photo.on.rectangle")
                    .font(.system(size: 22, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 54, height: 54)
                    .background(Color.white.opacity(0.12), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            }
            .disabled(isSearching)
            .accessibilityLabel("从相册选择商品图片")

            // Shutter: opens the live camera (device only; dimmed where there's no camera).
            Button {
                guard cameraAvailable, !isSearching else { return }
                isCameraPresented = true
            } label: {
                Circle()
                    .fill(GuideTheme.accent)
                    .frame(width: 56, height: 56)
                    .padding(6)
                    .background(Color.white, in: Circle())
                    .overlay { Circle().stroke(Color.white.opacity(0.82), lineWidth: 4) }
                    .opacity(cameraAvailable ? 1 : 0.35)
            }
            .buttonStyle(.plain)
            .disabled(!cameraAvailable || isSearching)
            .accessibilityLabel("拍照")

            // Keeps the shutter centred opposite the album button.
            Color.clear.frame(width: 54, height: 54)
        }
        .padding(.top, 20)
        .padding(.bottom, 32)
        .onChange(of: pickerItem) { newItem in
            guard let newItem else { return }
            isSearching = true
            Task {
                let data = try? await newItem.loadTransferable(type: Data.self)
                await MainActor.run {
                    isSearching = false
                    if let data {
                        stage(data)
                    }
                }
            }
        }
    }
}

/// Normalises any picked/captured image to a small JPEG: downscale the longest side to 1024px before
/// encoding. A raw camera photo is several MB, which both slows the upload (so the instant opener is
/// delayed until the request lands) and makes the server-side embed + VLM slower. 1024px keeps plenty
/// of detail for visual search. Falls through to the raw bytes if decoding isn't available.
private func normalizedJPEG(_ data: Data) -> Data {
    #if canImport(UIKit)
    guard let image = UIImage(data: data) else { return data }
    return downscaled(image, maxDimension: 1024).jpegData(compressionQuality: 0.85) ?? data
    #else
    return data
    #endif
}

#if canImport(UIKit)
private func downscaled(_ image: UIImage, maxDimension: CGFloat) -> UIImage {
    let longest = max(image.size.width, image.size.height)
    guard longest > maxDimension else { return image }
    let scale = maxDimension / longest
    let size = CGSize(width: image.size.width * scale, height: image.size.height * scale)
    // Force a 1x render so the output is exactly `size` pixels, not size × screen-scale.
    let format = UIGraphicsImageRendererFormat.default()
    format.scale = 1
    return UIGraphicsImageRenderer(size: size, format: format).image { _ in
        image.draw(in: CGRect(origin: .zero, size: size))
    }
}
#endif

@available(iOS 17.0, macOS 13.0, *)
private extension View {
    /// Presents the live-camera capture sheet on iOS; a no-op where there's no camera support.
    @ViewBuilder
    func cameraCover(isPresented: Binding<Bool>, onCapture: @escaping (Data) -> Void) -> some View {
        #if os(iOS)
        fullScreenCover(isPresented: isPresented) {
            CameraPicker(onCapture: onCapture).ignoresSafeArea()
        }
        #else
        self
        #endif
    }
}

#if os(iOS)
/// Thin SwiftUI wrapper over the system camera. Returns JPEG bytes via `onCapture`.
@available(iOS 17.0, *)
private struct CameraPicker: UIViewControllerRepresentable {
    let onCapture: (Data) -> Void
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ controller: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraPicker

        init(_ parent: CameraPicker) { self.parent = parent }

        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            if let image = info[.originalImage] as? UIImage, let data = image.jpegData(compressionQuality: 0.85) {
                parent.onCapture(data)
            }
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}
#endif

@available(iOS 17.0, macOS 13.0, *)
private enum PhotoCorner: CaseIterable {
    case topLeft
    case topRight
    case bottomLeft
    case bottomRight
}

@available(iOS 17.0, macOS 13.0, *)
private struct PhotoSearchCorner: Shape {
    let corner: PhotoCorner

    func path(in rect: CGRect) -> Path {
        let length: CGFloat = 24
        var path = Path()

        switch corner {
        case .topLeft:
            path.move(to: CGPoint(x: rect.minX, y: rect.minY + length))
            path.addLine(to: CGPoint(x: rect.minX, y: rect.minY))
            path.addLine(to: CGPoint(x: rect.minX + length, y: rect.minY))
        case .topRight:
            path.move(to: CGPoint(x: rect.maxX - length, y: rect.minY))
            path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
            path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + length))
        case .bottomLeft:
            path.move(to: CGPoint(x: rect.minX, y: rect.maxY - length))
            path.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))
            path.addLine(to: CGPoint(x: rect.minX + length, y: rect.maxY))
        case .bottomRight:
            path.move(to: CGPoint(x: rect.maxX - length, y: rect.maxY))
            path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
            path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY - length))
        }

        return path
    }
}

@available(iOS 17.0, macOS 13.0, *)
private struct OrderReviewScreen: View {
    let items: [CartItem]
    let backAction: () -> Void
    let confirmAction: () -> Void

    @State private var recipientName = "张三"
    @State private var phoneNumber = "13812341234"
    @State private var shippingAddress = "北京市朝阳区望京SOHO T1 12层"

    private var total: Decimal {
        items.reduce(Decimal.zero) { partialResult, item in
            partialResult + (item.product.basePrice * Decimal(item.quantity))
        }
    }

    private var canSubmit: Bool {
        !items.isEmpty
            && !recipientName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !phoneNumber.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !shippingAddress.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollView {
                    VStack(spacing: 10) {
                        addressCard
                        itemList
                        priceBreakdown
                    }
                    .padding(12)
                }
                .dismissesKeyboardOnScroll()

                Button {
                    dismissKeyboard()
                    confirmAction()
                } label: {
                    Text("提交订单 \(formattedTotal)")
                        .font(.headline.weight(.semibold))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .frame(height: 48)
                        .background(GuideTheme.accent)
                        .clipShape(Capsule())
                        .shadow(color: GuideTheme.accentShadow, radius: 14, y: 4)
                }
                .buttonStyle(.plain)
                .disabled(!canSubmit)
                .opacity(canSubmit ? 1 : 0.5)
                .padding(.horizontal, 16)
                .padding(.vertical, 14)
                .background(GuideTheme.panelBackground)
                .overlay(alignment: .top) {
                    Rectangle()
                        .fill(GuideTheme.line)
                        .frame(height: 1)
                }
            }
            .background(GuideTheme.pageBackground)
            .navigationTitle("确认订单")
            .keyboardDismissToolbar()
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(action: backAction) {
                        Image(systemName: "chevron.left")
                    }
                    .accessibilityLabel("返回聊天")
                }
            }
        }
    }

    private var addressCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("收货信息", systemImage: "mappin.circle.fill")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(GuideTheme.inkStrong)

            EditableOrderField(title: "联系人", text: $recipientName)
            EditableOrderField(title: "手机号", text: $phoneNumber)
            EditableOrderField(title: "详细地址", text: $shippingAddress, axis: .vertical)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(GuideTheme.panelBackground)
        .clipShape(RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous)
                .stroke(GuideTheme.line)
        }
    }

    private var itemList: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("商品清单")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(GuideTheme.inkStrong)
                .padding(.bottom, 8)

            if items.isEmpty {
                Text("购物车为空")
                    .font(.subheadline)
                    .foregroundStyle(GuideTheme.tertiaryInk)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 20)
            } else {
                ForEach(items) { item in
                    OrderItemRow(item: item)
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(GuideTheme.panelBackground)
        .clipShape(RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous)
                .stroke(GuideTheme.line)
        }
    }

    private var priceBreakdown: some View {
        VStack(spacing: 8) {
            PriceLine(label: "商品金额", value: formattedTotal, valueColor: GuideTheme.inkStrong)
            PriceLine(label: "运费", value: "免运费", valueColor: GuideTheme.success)

            Divider()
                .overlay(GuideTheme.line)
                .padding(.vertical, 2)

            HStack {
                Text("合计")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(GuideTheme.inkStrong)

                Spacer()

                Text(formattedTotal)
                    .font(.title3.weight(.bold))
                    .foregroundStyle(GuideTheme.accent)
            }
        }
        .padding(14)
        .background(GuideTheme.panelBackground)
        .clipShape(RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous)
                .stroke(GuideTheme.line)
        }
    }

    private var formattedTotal: String {
        let value = NSDecimalNumber(decimal: total)
        return GuideTheme.currencyFormatter.string(from: value) ?? "\(total)"
    }
}

@available(iOS 17.0, macOS 13.0, *)
struct EditableOrderField: View {
    let title: String
    @Binding var text: String
    var axis: Axis = .horizontal

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption2.weight(.medium))
                .foregroundStyle(GuideTheme.tertiaryInk)

            TextField(title, text: $text, axis: axis)
                .font(.subheadline.weight(.medium))
                .foregroundStyle(GuideTheme.inkStrong)
                .lineLimit(axis == .vertical ? 3 : 1)
                .padding(.horizontal, 10)
                .padding(.vertical, 9)
                .background(GuideTheme.pageBackground)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(GuideTheme.line)
                }
        }
    }
}

@available(iOS 17.0, macOS 13.0, *)
private struct OrderItemRow: View {
    let item: CartItem

    private var lineTotal: Decimal {
        item.product.basePrice * Decimal(item.quantity)
    }

    var body: some View {
        HStack(spacing: 10) {
            ProductImageView(product: item.product)
                .frame(width: 44, height: 44)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))

            VStack(alignment: .leading, spacing: 3) {
                Text(item.product.title)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(GuideTheme.inkStrong)
                    .lineLimit(1)

                Text("x\(item.quantity)")
                    .font(.caption2)
                    .foregroundStyle(GuideTheme.tertiaryInk)
            }

            Spacer(minLength: 8)

            Text(formattedLineTotal)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(GuideTheme.inkStrong)
        }
        .padding(.vertical, 8)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(GuideTheme.line)
                .frame(height: 1)
        }
    }

    private var formattedLineTotal: String {
        let value = NSDecimalNumber(decimal: lineTotal)
        return GuideTheme.currencyFormatter.string(from: value) ?? "\(lineTotal)"
    }
}

@available(iOS 17.0, macOS 13.0, *)
private struct PriceLine: View {
    let label: String
    let value: String
    let valueColor: Color

    var body: some View {
        HStack {
            Text(label)
                .font(.footnote)
                .foregroundStyle(GuideTheme.secondaryInk)

            Spacer()

            Text(value)
                .font(.footnote)
                .foregroundStyle(valueColor)
        }
    }
}

@available(iOS 17.0, macOS 13.0, *)
private struct OrderSuccessScreen: View {
    let returnAction: () -> Void
    private let orderNumber = String(UUID().uuidString.prefix(8)).uppercased()

    @State private var appeared = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        VStack(spacing: 0) {
            Spacer()

            ZStack {
                Circle()
                    .fill(GuideTheme.successSoft)
                    .frame(width: 72, height: 72)
                    .scaleEffect(appeared ? 1 : 0.4)
                    .animation(GuideMotion.entrance(reduceMotion: reduceMotion), value: appeared)

                CheckmarkShape()
                    .trim(from: 0, to: appeared ? 1 : 0)
                    .stroke(GuideTheme.success, style: StrokeStyle(lineWidth: 4, lineCap: .round, lineJoin: .round))
                    .frame(width: 30, height: 24)
                    .animation(.easeOut(duration: 0.5).delay(reduceMotion ? 0 : 0.2), value: appeared)
            }
            .padding(.bottom, 20)

            Group {
                Text("下单成功")
                    .font(.title3.weight(.bold))
                    .foregroundStyle(GuideTheme.inkStrong)

                Text("订单已提交，预计3-5个工作日送达")
                    .font(.subheadline)
                    .foregroundStyle(GuideTheme.secondaryInk)
                    .multilineTextAlignment(.center)
                    .padding(.top, 8)

                Text("订单号: \(orderNumber)")
                    .font(.caption)
                    .foregroundStyle(GuideTheme.tertiaryInk)
                    .padding(.top, 4)
            }
            .opacity(appeared ? 1 : 0)
            .offset(y: appeared ? 0 : 14)
            .animation(GuideMotion.entrance(reduceMotion: reduceMotion).delay(reduceMotion ? 0 : 0.25), value: appeared)

            Button(action: returnAction) {
                Text("返回聊天")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 40)
                    .frame(height: 44)
                    .background(GuideTheme.accent)
                    .clipShape(Capsule())
            }
            .buttonStyle(PressableButtonStyle())
            .padding(.top, 28)
            .opacity(appeared ? 1 : 0)
            .animation(GuideMotion.entrance(reduceMotion: reduceMotion).delay(reduceMotion ? 0 : 0.45), value: appeared)

            Spacer()
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(GuideTheme.panelBackground)
        .overlay {
            ConfettiView()
        }
        .sensoryFeedback(.success, trigger: appeared)
        .onAppear {
            appeared = true
        }
    }
}

@available(iOS 17.0, macOS 13.0, *)
private struct CheckmarkShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.minX + rect.width * 0.05, y: rect.midY + rect.height * 0.1))
        path.addLine(to: CGPoint(x: rect.minX + rect.width * 0.36, y: rect.maxY - rect.height * 0.08))
        path.addLine(to: CGPoint(x: rect.maxX - rect.width * 0.02, y: rect.minY + rect.height * 0.1))
        return path
    }
}
