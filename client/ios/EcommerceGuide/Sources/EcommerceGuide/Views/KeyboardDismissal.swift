import SwiftUI

#if canImport(UIKit)
import UIKit
#endif

@available(iOS 17.0, macOS 13.0, *)
func dismissKeyboard() {
    #if canImport(UIKit)
    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    #endif
}

@available(iOS 17.0, macOS 13.0, *)
extension View {
    @ViewBuilder
    func keyboardDismissToolbar() -> some View {
        #if os(iOS)
        toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()

                Button(action: dismissKeyboard) {
                    Image(systemName: "keyboard.chevron.compact.down")
                }
                .accessibilityLabel("收起键盘")
            }
        }
        #else
        self
        #endif
    }

    @ViewBuilder
    func dismissesKeyboardOnScroll() -> some View {
        #if os(iOS)
        scrollDismissesKeyboard(.interactively)
        #else
        self
        #endif
    }
}
