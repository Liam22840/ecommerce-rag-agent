import Foundation

public struct CartItem: Identifiable, Equatable, Sendable {
    public var id: String { product.id }
    public let product: Product
    public var quantity: Int
    // The variant the shopper chose ("50g标准装"), so it survives the cart round-trip and the server
    // keeps pricing this line for that SKU instead of reverting to the cheapest one.
    public var skuID: String?
    // The server's authoritative price for the chosen SKU: a display label ("268元（50g 标准装）") and
    // the per-unit price for the line total. Both fall back to the product's base price when absent
    // (e.g. a local optimistic add before the server replies). unitPrice is per-unit so the local
    // quantity stepper recomputes the line correctly.
    public var priceLabel: String?
    public var unitPrice: Double?

    public init(
        product: Product,
        quantity: Int = 1,
        skuID: String? = nil,
        priceLabel: String? = nil,
        unitPrice: Double? = nil
    ) {
        self.product = product
        self.quantity = quantity
        self.skuID = skuID
        self.priceLabel = priceLabel
        self.unitPrice = unitPrice
    }
}
