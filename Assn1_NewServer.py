import xmlrpc.server

def factorial(n):
    if n < 0:
        return "Invalid input"
    result = 1
    for i in range(2, n + 1):
        result *= i
    return str(result)  # ✅ return as string to avoid int overflow

server = xmlrpc.server.SimpleXMLRPCServer(('localhost', 8000))
server.register_function(factorial)
print("Server started on http://localhost:8000")
server.serve_forever()
