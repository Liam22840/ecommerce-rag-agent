import Foundation

public struct CartItem: Identifiable, Equatable, Sendable {
    public var id: String { product.id }
    public let product: Product
    public var quantity: Int

    public init(product: Product, quantity: Int = 1) {
        self.product = product
        self.quantity = quantity
    }
}
