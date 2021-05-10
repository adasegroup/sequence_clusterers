import torch


def get_partition(sample, num_of_steps, num_of_event_types, end_time=None):
    """
    Transforms a sample into partition
    inputs:
            sample -  pd.DataFrame; columns - time and type of event, sorted by time
            num_of_steps - int, number of steps in partition
            num_of_event_type - int, number of event types
            end_time - float, end time or None
    outputs:
            partition - torch.tensor, size = (num_of_steps, num_of_classes + 1)
    """
    if end_time is None:
        end_time = sample['time'][len(sample['time']) - 1]

    partition = torch.zeros(num_of_steps, num_of_event_types + 1)

    # finding time stamp
    dt = end_time / num_of_steps
    partition[:, 0] = end_time / num_of_steps

    # converting time to timestamps
    sample['time'] = (sample['time'] / dt).astype(int)
    mask = (sample['time'] == num_of_steps)
    sample.loc[mask, 'time'] -= 1

    # counting points
    sample = sample.reset_index()
    sample = sample.groupby(['time', 'event']).count()
    sample = sample.reset_index()
    sample.columns = ['time', 'event', 'num']
    try:
        sample['event'] = sample['event'].astype(int)
    except:
        global events
        global cur
        for i in range(len(sample['event'])):
            if sample['event'].iloc[i] not in events:
                events[sample['event'].iloc[i]] = cur
                cur += 1
            sample['event'].iloc[i] = events[sample['event'].iloc[i]]
        sample['event'] = sample['event'].astype(int)

    # computing partition
    temp = torch.from_numpy(sample.to_numpy())
    partition[temp[:, 0], temp[:, 1] + 1] = temp[:, 2].float()
    return partition