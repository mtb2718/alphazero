import torch

from alphazero.model_server import ModelServer

class SimpleLinearModel(torch.nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        # Very simple y = m * x + b
        self.fc = torch.nn.Linear(1, 1)
    
    @property
    def params(self):
        m = float(self.fc.weight.data)
        b = float(self.fc.bias.data)
        return m, b

    @params.setter
    def params(self, p):
        m, b = p
        self.fc.weight.data.fill_(m)
        self.fc.bias.data.fill_(b)

    def forward(self, x):
        return self.fc(x)


def test_model_server(tmp_path):

    model_server = ModelServer(tmp_path)
    model_server.reset()

    model = SimpleLinearModel()
    model.params = (1, 0)
    model_version = 0
    model_version = model_server.latest(model)
    assert model_version == 0
    assert (1, 0) == model.params

    retrieved_model = SimpleLinearModel()
    for i in range(1, 11):
        model.params = (0, i)
        model_server.update(model, i)
        model_version = model_server.latest(retrieved_model)
        assert model_version == i
        assert (0, i) == retrieved_model.params

# TODO: Test concurrency
