from pytorch_lightning.callbacks import Callback
import torch
import math


class GammaChanger(Callback):
    def __init__(self,
                 max_m_step_epoch: int = 50,
                 random_walking_max_epoch: int = 40,
                 true_clusters: int = 5,
                 upper_bound_clusters: int = 10,
                 verbose=True
                 ):
        self.max_m_step_epoch = max_m_step_epoch
        self.random_walking_max_epoch = random_walking_max_epoch
        self.true_clusters = true_clusters
        self.upper_bound_clusters = upper_bound_clusters
        self.verbose = verbose

    @staticmethod
    def __split(trainer, pl_module, cluster, batch):
        pl_module.model.split_cluster(cluster)
        pl_module.n_clusters += 1
        post_ll, _, _ = pl_module.self_gamma_step(batch)
        return post_ll

    @staticmethod
    def __merge(trainer, pl_module, cluster1, cluster2, batch):
        pl_module.model.merge_clusters(cluster1, cluster2)
        pl_module.n_clusters -= 1
        post_ll, _, _ = pl_module.self_gamma_step(batch)
        return post_ll

    @staticmethod
    def __delete(trainer, pl_module, cluster, batch):
        pl_module.model.delete_cluster(cluster)
        pl_module.n_clusters -= 1
        post_ll, _, _ = pl_module.self_gamma_step(batch)
        return post_ll

    @staticmethod
    def __batch_to_device(batch, device):
        x, y, g = batch
        x = x.to(device)
        y = y.to(device)
        g = g.to(device)
        return x, y, g

    def update_clusters_no_enforce(self, trainer, pl_module, operation):
        batch = self.__batch_to_device(trainer.datamodule.data_train[:], pl_module.device)
        pre_ll, _, _ = pl_module.self_gamma_step(batch)
        for cluster in range(pl_module.n_clusters):
            assert operation in ["split", "merge", "delete"]

            if operation == "split":
                post_ll = self.__split(trainer, pl_module, cluster, batch)
            if operation == "delete":
                post_ll = self.__delete(trainer, pl_module, cluster, batch)
            if operation == "merge":
                cluster2 = cluster
                while cluster2 == cluster:
                    cluster2 = int(torch.randint(pl_module.n_clusters, size=(1,))[0])
                post_ll = self.__merge(trainer, pl_module, cluster, cluster2, batch)

            remain_prob = min(
                1, math.exp(min(-post_ll + pre_ll, 1.0))  # TODO 1.0 is ok?
            )

            if (torch.rand(1) > remain_prob)[0]:
                pl_module.model.reverse_weighs()
                if operation == "split":
                    pl_module.n_clusters -= 1
                elif operation == "merge" or operation == "delete":
                    pl_module.n_clusters += 1
            else:
                if operation == "split":
                    trainer.datamodule.n_clusters += 1
                elif operation == "merge" or operation == "delete":
                    trainer.datamodule.n_clusters -= 1
                break

    def update_clusters_enforce(self, trainer, pl_module, operation):
        # TODO delete only enforcing and full_search merge enforcing
        batch = self.__batch_to_device(trainer.datamodule.data_train[:], pl_module.device)
        best_loss = 1e9
        cluster2 = None
        best_clusters = None
        for cluster in range(pl_module.n_clusters):
            assert operation in ["split", "merge", "delete"]

            if operation == "split":
                post_ll = self.__split(trainer, pl_module, cluster, batch)
            if operation == "delete":
                post_ll = self.__delete(trainer, pl_module, cluster, batch)
            if operation == "merge":
                cluster2 = cluster
                while cluster2 == cluster:
                    cluster2 = int(torch.randint(pl_module.n_clusters, size=(1,))[0])
                post_ll = self.__merge(trainer, pl_module, cluster, cluster2, batch)

            if post_ll < best_loss:
                best_loss = post_ll
                best_clusters = [cluster, cluster2]

            pl_module.model.reverse_weighs()
            if operation == "split":
                pl_module.n_clusters -= 1
            elif operation == "merge" or operation == "delete":
                pl_module.n_clusters += 1

        if operation == "split":
            pl_module.model.split_cluster(best_clusters[0])
            pl_module.n_clusters += 1
            trainer.datamodule.n_clusters += 1
        elif operation == "delete":
            pl_module.model.delete_cluster(best_clusters[0])
            pl_module.n_clusters -= 1
            trainer.datamodule.n_clusters -= 1
        elif operation == "merge":
            pl_module.model.merge_clusters(best_clusters[0], best_clusters[1])
            pl_module.n_clusters -= 1
            trainer.datamodule.n_clusters -= 1

    def update_clusters(self, trainer, pl_module):
        if pl_module.current_epoch >= self.random_walking_max_epoch:
            enforce = True
            if pl_module.n_clusters > self.true_clusters:
                # operation = "merge" if (torch.rand(1) > 0.5)[0] else "delete"
                operation = "delete"
            elif pl_module.n_clusters < self.true_clusters:
                operation = "split"
            else:
                operation = None
        else:
            enforce = False
            if pl_module.n_clusters == self.upper_bound_clusters:
                operation = "merge" if (torch.rand(1) > 0.5)[0] else "delete"
            elif pl_module.n_clusters == 1:
                operation = "split"
            else:
                operation = "split" if (torch.rand(1) > 0.5)[0] else "merge" if (torch.rand(1) > 0.5)[0] else "delete"
        
        # updating number of clusters
        if enforce:
            if operation is not None:
                self.update_clusters_enforce(trainer, pl_module, operation)
        else:
            self.update_clusters_no_enforce(trainer, pl_module, operation)

        trainer.datamodule.n_clusters = pl_module.n_clusters
        trainer.datamodule.dataset.reset_gamma(pl_module.n_clusters)

    def update_dataset(self, trainer, pl_module):
        trainer.datamodule.reset_datasets()
        # train
        _, gamma = pl_module(trainer.datamodule.data_train.data.to(pl_module.device))
        trainer.datamodule.data_train.gamma.gamma = gamma.detach().clone()
        # val
        _, gamma = pl_module(trainer.datamodule.data_val.data.to(pl_module.device))
        trainer.datamodule.data_val.gamma.gamma = gamma.detach().clone()
    
    def on_validation_epoch_end(self, trainer, pl_module):      
        pl_module.model.eval()
        with torch.no_grad():
            if (pl_module.current_epoch + 1) % self.max_m_step_epoch == 0:
                self.update_clusters(trainer, pl_module)
                self.update_dataset(trainer, pl_module)
                for opt in trainer.optimizers:
                    group = [param_group["lr"] for param_group in opt.param_groups]
                    lr = group[0]
                    new_opt = torch.optim.Adam(pl_module.model.parameters(), lr=lr, weight_decay = 1e-5)
                trainer.optimizers = [new_opt]
        pl_module.model.train()
