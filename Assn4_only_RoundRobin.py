# def create_load_balancer(server_names):
#     return {"servers": server_names, "current_index": 0}

# def get_next_server(load_balancer):
#     index = load_balancer["current_index"]
#     server = load_balancer["servers"][index]
#     load_balancer["current_index"] = (index + 1) % len(load_balancer["servers"])
#     return server

# if __name__ == "__main__":
#     num_servers = int(input("Enter the number of servers: "))
#     servers = [input(f"Enter the name of server {i+1}: ") for i in range(num_servers)]
    
#     lb = create_load_balancer(servers)
#     num_loads = int(input("Enter the number of loads: "))

#     print("\nLoad balancing result:")
#     for i in range(1, num_loads + 1):
#         server = get_next_server(lb)
#         print(f"Load {i} assigned to server: {server}")


def create_server_list():
    num_servers = int(input("Enter number of servers: "))
    servers = []
    for i in range(num_servers):
        name = input(f"Enter the name of server {i+1}: ")
        servers.append(name)  # simple addition, not repeated by weight
    return servers

def get_next_server(servers, current_index):
    server = servers[current_index]
    current_index = (current_index + 1) % len(servers)
    return server, current_index

if __name__ == "__main__":
    servers = create_server_list()
    num_loads = int(input("Enter number of loads to distribute: "))
    current_index = 0

    print("\nRound Robin Load Balancing:")
    for i in range(1, num_loads + 1):
        server, current_index = get_next_server(servers, current_index)
        print(f"Load {i} assigned to server: {server}")
