import Foundation

public enum FixtureLoader {
    public static func loadProducts(named name: String = "mock_products") throws -> [Product] {
        guard let url = Bundle.module.url(forResource: name, withExtension: "json") else {
            throw ChatServiceError.missingFixture("\(name).json")
        }

        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([Product].self, from: data)
    }
}
