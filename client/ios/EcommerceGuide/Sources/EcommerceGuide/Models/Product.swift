import Foundation

public struct Product: Identifiable, Codable, Equatable, Sendable {
    public let id: String
    public let title: String
    public let brand: String
    public let category: String
    public let subCategory: String
    public let basePrice: Decimal
    public let imagePath: String
    public let reason: String?

    enum CodingKeys: String, CodingKey {
        case id = "product_id"
        case title
        case brand
        case category
        case subCategory = "sub_category"
        case basePrice = "base_price"
        case imagePath = "image_path"
        case reason
    }

    public init(
        id: String,
        title: String,
        brand: String,
        category: String,
        subCategory: String,
        basePrice: Decimal,
        imagePath: String,
        reason: String? = nil
    ) {
        self.id = id
        self.title = title
        self.brand = brand
        self.category = category
        self.subCategory = subCategory
        self.basePrice = basePrice
        self.imagePath = imagePath
        self.reason = reason
    }
}
