import Foundation

public enum ChatServiceError: LocalizedError, Equatable {
    case invalidResponse
    case malformedEvent(String)
    case missingFixture(String)

    public var errorDescription: String? {
        switch self {
        case .invalidResponse:
            "服务器返回了无效响应。"
        case .malformedEvent(let event):
            "无法解析流式事件：\(event)"
        case .missingFixture(let name):
            "缺少测试数据：\(name)"
        }
    }
}
