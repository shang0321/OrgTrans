from copy import deepcopy

class YoloStructure:
    def __init__(self,nas_channels,metric=0.,flops=0.,params=0.):
        self.nas_channels = nas_channels
        self.metric = metric
        self.flops = flops
        self.params = params
    def mutate(self,mutate_ratio):
        pass

def regularized_evolution(
    cycles,
    population_size,
    sample_size,
    time_budget,
    random_arch,
    mutate_arch,
    nas_bench,
    extra_info,
    dataname,
):
    population = collections.deque()
    history, total_time_cost = (
        [],
        0,
    )

    while len(population) < population_size:
        model = Model()
        model.arch = random_arch()
        model.accuracy, time_cost = train_and_eval(
            model.arch, nas_bench, extra_info, dataname
        )
        population.append(model)
        history.append(model)
        total_time_cost += time_cost

    while total_time_cost < time_budget:
        start_time, sample = time.time(), []
        while len(sample) < sample_size:
            candidate = random.choice(list(population))
            sample.append(candidate)

        parent = max(sample, key=lambda i: i.accuracy)

        child = Model()
        child.arch = mutate_arch(parent.arch)
        total_time_cost += time.time() - start_time
        child.accuracy, time_cost = train_and_eval(
            child.arch, nas_bench, extra_info, dataname
        )
        if total_time_cost + time_cost > time_budget:
            return history, total_time_cost
        else:
            total_time_cost += time_cost
        population.append(child)
        history.append(child)

        population.popleft()
    return history, total_time_cost