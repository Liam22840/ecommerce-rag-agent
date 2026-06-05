import Foundation

public struct Product: Identifiable, Codable, Equatable, Sendable {
    public let id: String
    public let title: String
    public let brand: String
    public let category: String
    public let subCategory: String
    public let basePrice: Decimal
    public let priceLabel: String?
    public let priceSummary: String?
    public let imagePath: String
    public let reason: String?
    public let spec: String?
    public let rating: Double?
    public let sales: String?
    public let pros: [String]
    public let cons: [String]

    enum CodingKeys: String, CodingKey {
        case id = "product_id"
        case title
        case brand
        case category
        case subCategory = "sub_category"
        case basePrice = "base_price"
        case price
        case priceLabel = "price_label"
        case priceSummary = "price_summary"
        case imagePath = "image_path"
        case reason
        case matchedReason = "matched_reason"
        case spec
        case rating
        case sales
        case pros
        case cons
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decode(String.self, forKey: .id)
        self.title = try container.decode(String.self, forKey: .title)
        self.brand = try container.decode(String.self, forKey: .brand)
        self.category = try container.decode(String.self, forKey: .category)
        self.subCategory = try container.decode(String.self, forKey: .subCategory)
        self.basePrice = try Self.decodePrice(from: container)
        self.priceLabel = try container.decodeIfPresent(String.self, forKey: .priceLabel)
        self.priceSummary = try container.decodeIfPresent(String.self, forKey: .priceSummary)
        self.imagePath = try container.decode(String.self, forKey: .imagePath)
        self.reason = try container.decodeIfPresent(String.self, forKey: .reason)
            ?? container.decodeIfPresent(String.self, forKey: .matchedReason)
        self.spec = try container.decodeIfPresent(String.self, forKey: .spec)
        self.rating = try container.decodeIfPresent(Double.self, forKey: .rating)
        self.sales = try container.decodeIfPresent(String.self, forKey: .sales)
        self.pros = try container.decodeIfPresent([String].self, forKey: .pros) ?? []
        self.cons = try container.decodeIfPresent([String].self, forKey: .cons) ?? []
    }

    private static func decodePrice(from container: KeyedDecodingContainer<CodingKeys>) throws -> Decimal {
        if let price = try decodeDecimal(forKey: .basePrice, from: container) {
            return price
        }
        if let price = try decodeDecimal(forKey: .price, from: container) {
            return price
        }
        throw DecodingError.keyNotFound(
            CodingKeys.basePrice,
            DecodingError.Context(
                codingPath: container.codingPath,
                debugDescription: "Expected base_price or price"
            )
        )
    }

    private static func decodeDecimal(
        forKey key: CodingKeys,
        from container: KeyedDecodingContainer<CodingKeys>
    ) throws -> Decimal? {
        if let value = try? container.decode(Double.self, forKey: key) {
            return Decimal(value)
        }
        if let value = try? container.decode(Int.self, forKey: key) {
            return Decimal(value)
        }
        if let value = try? container.decode(String.self, forKey: key),
           let decimal = Decimal(string: value) {
            return decimal
        }
        if let value = try? container.decode(Decimal.self, forKey: key) {
            return value
        }
        return nil
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(title, forKey: .title)
        try container.encode(brand, forKey: .brand)
        try container.encode(category, forKey: .category)
        try container.encode(subCategory, forKey: .subCategory)
        try container.encode(basePrice, forKey: .basePrice)
        try container.encodeIfPresent(priceLabel, forKey: .priceLabel)
        try container.encodeIfPresent(priceSummary, forKey: .priceSummary)
        try container.encode(imagePath, forKey: .imagePath)
        try container.encodeIfPresent(reason, forKey: .reason)
        try container.encodeIfPresent(spec, forKey: .spec)
        try container.encodeIfPresent(rating, forKey: .rating)
        try container.encodeIfPresent(sales, forKey: .sales)
        if !pros.isEmpty {
            try container.encode(pros, forKey: .pros)
        }
        if !cons.isEmpty {
            try container.encode(cons, forKey: .cons)
        }
    }

    public init(
        id: String,
        title: String,
        brand: String,
        category: String,
        subCategory: String,
        basePrice: Decimal,
        priceLabel: String? = nil,
        priceSummary: String? = nil,
        imagePath: String,
        reason: String? = nil,
        spec: String? = nil,
        rating: Double? = nil,
        sales: String? = nil,
        pros: [String] = [],
        cons: [String] = []
    ) {
        self.id = id
        self.title = title
        self.brand = brand
        self.category = category
        self.subCategory = subCategory
        self.basePrice = basePrice
        self.priceLabel = priceLabel
        self.priceSummary = priceSummary
        self.imagePath = imagePath
        self.reason = reason
        self.spec = spec
        self.rating = rating
        self.sales = sales
        self.pros = pros
        self.cons = cons
    }
}
