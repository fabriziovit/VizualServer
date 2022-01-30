from werkzeug.wrappers import Request, Response, ResponseStream


class AuthorizationMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        request = Request(environ)
        print(request.headers)
        return self.app(environ, start_response)
        # res = Response(u'Authorization failed', mimetype='text/plain', status=401)
        # return res(environ, start_response)
