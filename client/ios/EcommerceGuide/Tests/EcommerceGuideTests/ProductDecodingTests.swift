import XCTest
@testable import EcommerceGuide

final class ProductDecodingTests: XCTestCase {
    func testDecodesDatasetStyleProductJSON() throws {
        let json = """
        {
          "product_id": "SKU-001",
          "title": "Trail Running Shoes",
          "brand": "Northstar",
          "category": "Shoes",
          "sub_category": "Running",
          "base_price": 129.99,
          "image_path": "images/sku-001.jpg",
          "reason": "Matches waterproof trail preference"
        }
        """

        let product = try JSONDecoder().decode(Product.self, from: Data(json.utf8))

        XCTAssertEqual(product.id, "SKU-001")
        XCTAssertEqual(product.title, "Trail Running Shoes")
        XCTAssertEqual(product.brand, "Northstar")
        XCTAssertEqual(product.category, "Shoes")
        XCTAssertEqual(product.subCategory, "Running")
        XCTAssertEqual(product.basePrice, Decimal(string: "129.99")!)
        XCTAssertEqual(product.imagePath, "images/sku-001.jpg")
        XCTAssertEqual(product.reason, "Matches waterproof trail preference")
    }

    func testDecodesDatasetStyleProductWithoutOptionalReason() throws {
        let json = """
        {
          "product_id": "SKU-002",
          "title": "Merino Socks",
          "brand": "LayerLab",
          "category": "Accessories",
          "sub_category": "Socks",
          "base_price": 18.50,
          "image_path": "images/sku-002.jpg"
        }
        """

        let product = try JSONDecoder().decode(Product.self, from: Data(json.utf8))

        XCTAssertEqual(product.id, "SKU-002")
        XCTAssertNil(product.reason)
        XCTAssertEqual(product.basePrice, Decimal(string: "18.50")!)
    }

    func testDecodesBackendProductCardShape() throws {
        let json = """
        {
          "product_id": "p_beauty_012",
          "title": "理肤泉特安舒缓修复霜",
          "brand": "理肤泉",
          "category": "美妆护肤",
          "sub_category": "面霜",
          "price": 260.0,
          "price_label": "260元（40ml 清爽型 / 40ml 滋润型）",
          "price_summary": "40ml 清爽型 260元；40ml 滋润型 260元",
          "image_path": "1_美妆护肤/images/p_beauty_012_live.jpg",
          "detail_path": "/api/products/p_beauty_012",
          "matched_reason": "符合面霜需求"
        }
        """

        let product = try JSONDecoder().decode(Product.self, from: Data(json.utf8))

        XCTAssertEqual(product.id, "p_beauty_012")
        XCTAssertEqual(product.basePrice, Decimal(string: "260.0")!)
        XCTAssertEqual(product.priceLabel, "260元（40ml 清爽型 / 40ml 滋润型）")
        XCTAssertEqual(product.priceSummary, "40ml 清爽型 260元；40ml 滋润型 260元")
        XCTAssertEqual(product.reason, "符合面霜需求")
    }

    @available(iOS 17.0, macOS 13.0, *)
    func testFormattedPriceUsesBackendPriceLabelForMultiSkuProducts() throws {
        let json = """
        {
          "product_id": "p_beauty_007",
          "title": "薇诺娜舒敏保湿特护霜敏感肌修护屏障舒缓干痒保湿面霜50g",
          "brand": "薇诺娜",
          "category": "美妆护肤",
          "sub_category": "面霜",
          "price": 89.0,
          "price_label": "89元起（15g 体验装）",
          "price_summary": "15g 体验装 89元；50g 标准装 268元",
          "image_path": "1_美妆护肤/images/p_beauty_007_live.jpg"
        }
        """

        let product = try JSONDecoder().decode(Product.self, from: Data(json.utf8))

        XCTAssertEqual(product.formattedPrice, "89元起（15g 体验装）")
        XCTAssertEqual(product.priceSummary, "15g 体验装 89元；50g 标准装 268元")
    }

    func testDecodesPriceFromStringForRuntimeCompatibility() throws {
        let json = """
        {
          "product_id": "SKU-003",
          "title": "Runtime Safe Price",
          "brand": "LayerLab",
          "category": "Accessories",
          "sub_category": "Socks",
          "price": "18.50",
          "image_path": "images/sku-003.jpg"
        }
        """

        let product = try JSONDecoder().decode(Product.self, from: Data(json.utf8))

        XCTAssertEqual(product.basePrice, Decimal(string: "18.50")!)
    }

    func testDecodesOptionalComparisonMetadata() throws {
        let json = """
        {
          "product_id": "SKU-004",
          "title": "Comparison Ready Product",
          "brand": "GuideLab",
          "category": "Beauty",
          "sub_category": "Sunscreen",
          "base_price": 158,
          "image_path": "images/sku-004.jpg",
          "spec": "SPF50+ PA+++ 60ml",
          "rating": 4.8,
          "sales": "10万+",
          "pros": ["无酒精无香精", "不闷痘"],
          "cons": ["轻微泛白"]
        }
        """

        let product = try JSONDecoder().decode(Product.self, from: Data(json.utf8))

        XCTAssertEqual(product.spec, "SPF50+ PA+++ 60ml")
        XCTAssertEqual(product.rating, 4.8)
        XCTAssertEqual(product.sales, "10万+")
        XCTAssertEqual(product.pros, ["无酒精无香精", "不闷痘"])
        XCTAssertEqual(product.cons, ["轻微泛白"])
    }
}
