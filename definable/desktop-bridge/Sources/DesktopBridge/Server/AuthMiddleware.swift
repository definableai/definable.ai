import Vapor

struct BearerAuthMiddleware: AsyncMiddleware {
  let validToken: String

  func respond(to request: Request, chainingTo next: any AsyncResponder) async throws -> Response {
    guard let bearer = request.headers.bearerAuthorization else {
      throw Abort(.unauthorized, reason: "Missing Authorization header")
    }
    guard bearer.token == validToken else {
      throw Abort(.forbidden, reason: "Invalid token")
    }
    return try await next.respond(to: request)
  }
}
