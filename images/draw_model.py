import matplotlib.pyplot as plt
import networkx as nx

def plot_neural_network():
    G = nx.DiGraph()

    # Define the layers and their respective sizes
    layers = {
        "Input": 3,               # Input layer with 3 channels (RGB)
        "Conv1": 32,              # First convolution layer
        "BatchNorm1": 32,         # Batch normalization after Conv1
        "Conv2": 64,              # Second convolution layer
        "BatchNorm2": 64,         # Batch normalization after Conv2
        "Conv3": 128,             # Third convolution layer
        "BatchNorm3": 128,        # Batch normalization after Conv3
        "Conv4": 256,             # Fourth convolution layer (new)
        "BatchNorm4": 256,        # Batch normalization after Conv4
        "Conv5": 512,             # Fifth convolution layer (new)
        "BatchNorm5": 512,        # Batch normalization after Conv5
        "Flatten": 512 * 2 * 2,   # Flatten for fully connected layers
        "FC1": 256,               # Fully connected layer 1
        "Dropout": 256,           # Dropout applied (10%)
        "FC2": 15     # Final output layer
    }

    # Create nodes for each layer
    pos = {}
    layer_width = 10  # Space between layers
    y_offset = 20     # Height for nodes
    for i, (layer, size) in enumerate(layers.items()):
        for j in range(min(10, size if isinstance(size, int) else 1)):
            node_name = f"{layer}_{j}"
            G.add_node(node_name, layer=layer)
            pos[node_name] = (i * layer_width, j * y_offset)

    # Add edges between consecutive layers
    previous_layer = None
    for layer, size in layers.items():
        if previous_layer:
            for prev_node in [n for n in G.nodes if G.nodes[n]['layer'] == previous_layer]:
                for curr_node in [n for n in G.nodes if G.nodes[n]['layer'] == layer]:
                    G.add_edge(prev_node, curr_node)
        previous_layer = layer

    # Plot the graph
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_size=50, alpha=0.7, edge_color="blue")
    labels = {n: G.nodes[n]['layer'] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels={k: k.split("_")[0] for k in labels}, font_size=8)
    plt.title("Neural Network Architecture Visualization")
    plt.show()

# Execute the function to plot the neural network
plot_neural_network()