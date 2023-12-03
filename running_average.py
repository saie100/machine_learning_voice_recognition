def running_average(sequence):
    total = 0
    averages = []

    for i, number in enumerate(sequence, start=1):
        total += number
        averages.append(total / i)

    return averages


# Example usage
data = [10, 20, 30, 40, 50]
print(running_average(data))
