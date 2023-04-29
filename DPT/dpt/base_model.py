import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        cuda = torch.cuda.is_available()
        if cuda:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            torch.cuda.empty_cache()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        parameters = torch.load(path, map_location=device)
        # print(parameters)
        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
