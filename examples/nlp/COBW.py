import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, softmax
from XuanJing.utils.data.nlp_data import corpus, process_w2v_data
from XuanJing.utils.torch_utils import tensorify
from XuanJing.utils.show.nlp_utils import show_w2v_word_embedding


class CBOW(nn.Module):
    def __init__(self, v_dim, emb_dim):
        super().__init__()
        self.v_dim = v_dim
        self.embeddings = nn.Embedding(v_dim, emb_dim)
        self.embeddings.weight.data.normal_(0, 0.1)

        # self.opt = torch.optim.Adam(0.01)
        self.hidden_out = nn.Linear(emb_dim, v_dim)
        self.opt = torch.optim.SGD(self.parameters(), momentum=0.9, lr=0.01)

    def forward(self, x, training=None, mask=None):
        # x.shape = [n,skip_window*2]
        o = self.embeddings(x)  # [n, skip_window*2, emb_dim]
        o = torch.mean(o, dim=1)  # [n, emb_dim]
        return o

    def loss(self, x, y, training=None):
        embedded = self(x, training)
        pred = self.hidden_out(embedded)
        return cross_entropy(pred, y)

    def step(self, x, y):
        self.opt.zero_grad()
        loss = self.loss(x, y, True)
        loss.backward()
        self.opt.step()
        return loss.detach().numpy()


def train(model, data):
    if torch.cuda.is_available():
        print("GPU train avaliable")
        device = torch.device("cuda")
        model = model.cuda()
    else:
        device = torch.device("cpu")
        model = model.cpu()
    for t in range(2500):
        bx, by = data.sample(16)
        bx, by = tensorify(bx), tensorify(by)
        loss = model.step(bx, by)
        if t % 200 == 0:
            print(f"step: {t}  |  loss: {loss}")


if __name__ == "__main__":
    d = process_w2v_data(corpus, skip_window=2, method="cbow")
    m = CBOW(d.num_word, 2)
    train(m, d)

    show_w2v_word_embedding(m, d, "./visual/results/cbow.png")