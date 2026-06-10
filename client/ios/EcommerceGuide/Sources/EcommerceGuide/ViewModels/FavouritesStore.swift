import Foundation

@MainActor
@available(iOS 17.0, macOS 14.0, *)
public final class FavouritesStore: ObservableObject {
    @Published public private(set) var items: [Product]

    private let defaults: UserDefaults
    private static let storageKey = "EcommerceGuideFavourites"

    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        if let data = defaults.data(forKey: Self.storageKey),
           let decoded = try? JSONDecoder().decode([Product].self, from: data) {
            self.items = decoded
        } else {
            self.items = []
        }
    }

    public func isFavourite(_ product: Product) -> Bool {
        items.contains { $0.id == product.id }
    }

    public func toggle(_ product: Product) {
        if let index = items.firstIndex(where: { $0.id == product.id }) {
            items.remove(at: index)
        } else {
            items.append(product)
        }
        persist()
    }

    public func remove(productID: String) {
        items.removeAll { $0.id == productID }
        persist()
    }

    private func persist() {
        guard let data = try? JSONEncoder().encode(items) else {
            return
        }
        defaults.set(data, forKey: Self.storageKey)
    }
}
