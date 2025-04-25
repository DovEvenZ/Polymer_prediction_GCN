import torch

class calc_error():
    def __init__(self, DataLoader, model, device, mean, std, transform = None) -> None:
        self.DataLoader = DataLoader
        self.model = model
        self.device = device
        self.mean = mean
        self.std = std
        self.transform = transform
        self.pred, self.target = self._get_pred()

    def _get_pred(self):
        self.model.eval()
        pred = []
        target = []
        for data in self.DataLoader:
            data = data.to(self.device)
            pred.append(self.model(data))
            target.append(data.y)
        sum_pred = (torch.cat(pred) * self.std + self.mean).detach()
        sum_target = (torch.cat(target) * self.std + self.mean).detach()
        del pred, target

        sum_pred = self.target_inverse_transform(sum_pred)
        sum_target = self.target_inverse_transform(sum_target)

        return sum_pred, sum_target

    def target_inverse_transform(self, data: torch.tensor):
        if self.transform == 'LN':
            return torch.e ** data
        elif self.transform == 'LG':
            return 10 ** data
        elif self.transform == 'E^-x':
            return -torch.log(data)
        elif not self.transform:
            return data

    def MAE(self):
        return (self.pred - self.target).abs().sum().item() / len(self.DataLoader.dataset)

    def MSE(self):
        return ((self.pred - self.target) ** 2).sum().item() / len(self.DataLoader.dataset)

    def S2(self):
        return ((self.target - self.mean) ** 2).sum().item() / len(self.DataLoader.dataset)

    def R2(self):
        return 1 - self.MSE() / self.S2()

    def ARD(self):
        return ((self.target - self.pred).abs() / self.target).sum().item() / len(self.DataLoader.dataset)

    def RD_each(self):
        return (self.pred - self.target) / self.target

    def MRD(self):
        if abs(max(self.RD_each())) > abs(min(self.RD_each())):
            return max(self.RD_each())
        else:
            return min(self.RD_each())
