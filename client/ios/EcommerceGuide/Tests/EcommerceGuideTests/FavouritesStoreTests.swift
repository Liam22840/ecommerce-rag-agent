import XCTest
@testable import EcommerceGuide

@MainActor
final class FavouritesStoreTests: XCTestCase {
    private static let suiteName = "FavouritesStoreTests"
    private var defaults: UserDefaults!

    override func setUp() {
        super.setUp()
        defaults = UserDefaults(suiteName: Self.suiteName)!
        defaults.removePersistentDomain(forName: Self.suiteName)
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: Self.suiteName)
        super.tearDown()
    }

    private func makeProduct(id: String = "p_test_001") -> Product {
        Product(
            id: id,
            title: "测试洗面奶",
            brand: "测试品牌",
            category: "美妆护肤",
            subCategory: "洁面",
            basePrice: 52,
            imagePath: "1_美妆护肤/images/p_test_001_live.jpg"
        )
    }

    func testToggleAddsThenRemoves() {
        let store = FavouritesStore(defaults: defaults)
        let product = makeProduct()

        store.toggle(product)
        XCTAssertEqual(store.items.map(\.id), ["p_test_001"])
        XCTAssertTrue(store.isFavourite(product))

        store.toggle(product)
        XCTAssertTrue(store.items.isEmpty)
        XCTAssertFalse(store.isFavourite(product))
    }

    func testRemoveByProductID() {
        let store = FavouritesStore(defaults: defaults)
        store.toggle(makeProduct(id: "a"))
        store.toggle(makeProduct(id: "b"))

        store.remove(productID: "a")

        XCTAssertEqual(store.items.map(\.id), ["b"])
    }

    func testPersistenceRoundTrip() {
        let store = FavouritesStore(defaults: defaults)
        store.toggle(makeProduct())

        let reloaded = FavouritesStore(defaults: defaults)

        XCTAssertEqual(reloaded.items.map(\.id), ["p_test_001"])
    }

    func testCorruptPayloadFallsBackToEmpty() {
        defaults.set(Data("not json".utf8), forKey: "EcommerceGuideFavourites")

        let store = FavouritesStore(defaults: defaults)

        XCTAssertTrue(store.items.isEmpty)
    }
}
