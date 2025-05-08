def create_server_list():
    num_servers = int(input("Enter number of servers: "))
    servers = []
    weights = []
    for i in range(num_servers):
        name = input(f"Enter the name of server {i+1}: ")
        weight = int(input(f"Enter weight for server {name}: "))
        servers.append({"name": name, "weight": weight, "current_weight": 0})
    return servers

def get_next_server(servers):
    total_weight = sum(server["weight"] for server in servers)

    for server in servers:
        server["current_weight"] += server["weight"]

    # Select server with max current_weight
    selected = max(servers, key=lambda s: s["current_weight"])
    selected["current_weight"] -= total_weight
    return selected["name"]

if __name__ == "__main__":
    servers = create_server_list()
    num_loads = int(input("Enter number of loads to distribute: "))

    print("\nWeighted Round Robin Load Balancing:")
    for i in range(1, num_loads + 1):
        server_name = get_next_server(servers)
        print(f"Load {i} assigned to server: {server_name}")
