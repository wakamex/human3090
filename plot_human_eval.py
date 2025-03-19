#!/usr/bin/env python3
"""Generate a scatter plot of HumanEval accuracy vs. time taken from README.md data."""
import matplotlib.pyplot as plt
import numpy as np


def parse_readme_data(readme_path):
    """Parse the HumanEval data from the README.md file."""
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the table section
    table_start = content.find('## HumanEval on a local 3090 results')
    if table_start == -1 or (header_start := content.find('| Model', table_start)) == -1:
        return []

    # Extract table content
    next_section = content.find('##', header_start + 1)
    table_end = next_section if next_section != -1 else len(content)
    table_lines = content[header_start:table_end].split('\n')[2:]  # Skip header and separator

    data = []
    for line in table_lines:
        # Skip non-table lines
        if not line.strip() or not line.startswith('|'):
            continue

        # Extract cells
        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
        if len(cells) < 3:
            continue

        # Process accuracy
        accuracy_cell = cells[-2]
        if '%' not in accuracy_cell or accuracy_cell.lower() == 'failed':
            continue
            
        # Process time
        time_cell = cells[-1]
        if not time_cell:
            continue

        # Try to parse values
        try:
            accuracy = float(accuracy_cell.rstrip('%'))
            time_taken = float(time_cell.replace(',', '').rstrip('s'))
            if time_taken <= 0:
                continue
                
            # Add valid data point
            data.append({
                'model': cells[0],
                'config': cells[1] if len(cells) > 3 else "",
                'accuracy': accuracy,
                'time_taken': time_taken
            })
        except ValueError:
            continue

    return data


def find_efficient_frontier(times, accuracies):
    """Find non-dominated models (better accuracy, lower time)."""
    n = len(times)
    is_efficient = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j and (
                (accuracies[j] > accuracies[i] and times[j] <= times[i]) or
                (accuracies[j] >= accuracies[i] and times[j] < times[i])
            ):
                is_efficient[i] = False
                break

    return is_efficient


def create_scatter_plot(data):
    """Create a scatter plot of HumanEval accuracy vs. time taken."""
    # Extract data
    times = [item['time_taken'] for item in data]
    accuracies = [item['accuracy'] for item in data]
    models = [item['model'] for item in data]

    # Find efficient frontier
    efficient = find_efficient_frontier(times, accuracies)

    # Setup plot
    plt.figure(figsize=(12, 8))

    # Plot dominated and non-dominated points
    dominated_indices = [i for i, e in enumerate(efficient) if not e]
    frontier_indices = [i for i, e in enumerate(efficient) if e]

    plt.scatter(
        [times[i] for i in dominated_indices],
        [accuracies[i] for i in dominated_indices],
        alpha=0.7, s=100, color='blue', label='Dominated Models'
    )

    plt.scatter(
        [times[i] for i in frontier_indices],
        [accuracies[i] for i in frontier_indices],
        alpha=0.7, s=100, color='red', label='Efficient Frontier'
    )

    # Add labels and formatting
    plt.xlabel('Time Taken (seconds)', fontsize=12)
    plt.ylabel('HumanEval Accuracy (%)', fontsize=12)
    plt.title('HumanEval Accuracy vs. Time Taken by Model', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 100)

    # Add model name annotations
    for i, model in enumerate(models):
        plt.annotate(model, 
                    (times[i], accuracies[i]),
                    xytext=(10, 0),  # Increased horizontal offset
                    textcoords='offset points', 
                    fontsize=9,       # Slightly larger font
                    ha='left',        # Left-align text
                    va='center'      # Align to bottom of point
                    )

    # Use log scale if time range is large
    non_zero_times = [t for t in times if t > 0]
    if non_zero_times and max(non_zero_times) / min(non_zero_times) > 100:
        plt.xscale('log')
        plt.xlabel('Time Taken (seconds, log scale)', fontsize=12)

    # Draw efficient frontier line
    if len(frontier_indices) >= 2:
        sorted_indices = sorted(frontier_indices, key=lambda i: times[i])
        plt.plot(
            [times[i] for i in sorted_indices],
            [accuracies[i] for i in sorted_indices],
            'r-', alpha=0.5
        )

    # Save and show plot
    plt.tight_layout()
    plt.savefig('human_eval_scatter.png', dpi=300)
    print("Plot saved as 'human_eval_scatter.png'")
    plt.show()


def main():
    """Main function to run the script."""
    readme_path = '/code/human3090/README.md'
    data = parse_readme_data(readme_path)

    if not data:
        print("No data found in README.md")
        return

    print(f"Found {len(data)} data points")

    # Print data points sorted by time
    print("\nData points for inspection:")
    print(f"{"Model":<30} {"Accuracy (%)":<15} {"Time (s)":<15}")
    print("-" * 60)

    for item in sorted(data, key=lambda x: x['time_taken']):
        print(f"{item['model'][:30]:<30} {item['accuracy']:<15.1f} {item['time_taken']:<15.2f}")

    # Print efficient frontier models
    times = [item['time_taken'] for item in data]
    accuracies = [item['accuracy'] for item in data]
    models = [item['model'] for item in data]
    efficient = find_efficient_frontier(times, accuracies)

    print("\nModels on the Efficient Frontier:")
    print("-" * 60)

    efficient_indices = [i for i, e in enumerate(efficient) if e]
    for i in sorted(efficient_indices, key=lambda i: times[i]):
        print(f"{models[i][:30]:<30} {accuracies[i]:<15.1f} {times[i]:<15.2f}")

    create_scatter_plot(data)


if __name__ == "__main__":
    main()
