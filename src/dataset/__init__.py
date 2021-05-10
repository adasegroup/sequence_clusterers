from src.dataset.random_seq import RandomGeneratedSequences


if __name__ == '__main__':
    dataset_ex = RandomGeneratedSequences('data/sin_K2_C5', num_of_steps=1000, num_of_event_types=8)
    partitions = []
