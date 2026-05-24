# iOS Frontend UI Design

## Direction

The iOS client is a native SwiftUI shopping-assistant demo. The first screen is
the conversation, not a landing page: users can type a product need, watch the
assistant stream a response, inspect inline product cards, and simulate cart
actions without a live backend.

## Frontend Architecture

- `SwiftUI` for screens and components.
- `Observable` view models with Swift concurrency for streaming state.
- `ChatService` protocol to switch between local fixtures and the future backend.
- `MockChatService` as the default demo path while the backend API is still being
  built.
- `SSEChatService` as the integration shell for the eventual streaming endpoint.

The Simulator host app defaults to the mock service. To exercise a backend
without changing source, set the launch environment variable
`ECOMMERCE_GUIDE_SERVICE=sse` in the `EcommerceGuideApp` scheme, or set the
`EcommerceGuideService` user default to `sse`.

## Backend Contract Assumptions

The iOS app assumes the backend will eventually expose:

```http
POST /api/v1/chat/stream
Content-Type: application/json
Accept: text/event-stream
```

Request body:

```json
{
  "conversation_id": "uuid",
  "message": "推荐一款适合油皮的洗面奶",
  "attachments": [],
  "client_context": {
    "cart_items": []
  }
}
```

Streaming events:

```text
event: token
data: {"text":"推荐"}

event: products
data: {"items":[{"product_id":"p_beauty_011","title":"...","brand":"珊珂","category":"美妆护肤","sub_category":"洁面","base_price":52,"image_path":"1_美妆护肤/images/p_beauty_011_live.jpg","reason":"适合预算内温和清洁"}]}

event: cart
data: {"items":[{"product_id":"p_beauty_011","quantity":1}],"summary":"已加入购物车"}

event: done
data: {"message_id":"uuid"}
```

## Safety Rules

- API keys stay backend-only and must never ship in the iOS app.
- The client only displays product fields returned by the backend or bundled
  fixtures.
- Price, product title, and SKU-like facts should be treated as trusted backend
  data, not generated assistant prose.
- Cart events with only `product_id` are shown as status updates; to replace the
  native cart list, the backend should include full product objects in
  `cart_items`.
- If the backend is unavailable, the mock service keeps the app demoable.
