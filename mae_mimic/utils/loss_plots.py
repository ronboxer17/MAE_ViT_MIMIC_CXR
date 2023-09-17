import os
import json
import matplotlib.pyplot as plt


def calculate_mean_losses(file_path: str):
    with open(file_path, 'r') as file:
        data = json.load(file)

    transformer_name = data['metadata']['transformer']
    sample_size = data['metadata']['sample_size']
    results = data['results']

    epoch_numbers = []
    mean_losses = []

    max_epochs = 50 if sample_size == 1000 else 20

    for epoch_key in results.keys():
        if int(epoch_key) < max_epochs:
            epoch_data = results[epoch_key]
            losses = epoch_data.get('losses', [])
            if losses:
                mean_loss = sum(losses) / len(losses)
                epoch_numbers.append(int(epoch_key))
                mean_losses.append(mean_loss)

    return sample_size, transformer_name, epoch_numbers, mean_losses


def plot_losses(file_path: str, out_directory: str):
    data_by_sample_size = {}

    # Iterate over all JSON files in the directory
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)

                # Extract sample_size and categorize the file
                sample_size, transformer_name, epoch_numbers, mean_losses = calculate_mean_losses(file_path)

                if sample_size not in data_by_sample_size:
                    data_by_sample_size[sample_size] = []

                data_by_sample_size[sample_size].append((transformer_name, epoch_numbers, mean_losses))

    #  Create two separate plots for each sample_size
    for sample_size, data_list in data_by_sample_size.items():
        plt.figure()
        for data in data_list:
            transformer_name, epoch_numbers, mean_losses = data
            plt.plot(epoch_numbers, mean_losses, marker='o', label=transformer_name)

        plt.xlabel('Epoch')
        plt.ylabel('Mean Loss')
        plt.title(f'Mean Loss per Epoch (Sample Size: {sample_size})')
        plt.grid(True)
        plt.legend()

        if sample_size == 10000:
            plt.xlim(0, 20)  # Set x-axis limit to 0-20 for sample_size=10000

        # plt.show()
        plt.savefig(os.path.join(out_directory, f'mean_loss_{sample_size}.png'))
        plt.close()


input_directory = r"../assets/metrics"
output_directory = "../assets/imgs"

plot_losses(input_directory, output_directory)
