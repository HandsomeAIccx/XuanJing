import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, softmax
from XuanJing.utils.data.nlp_data import corpus, process_w2v_data
from XuanJing.utils.torch_utils import tensorify
from XuanJing.utils.show.nlp_utils import show_w2v_word_embedding


class SkipGram(nn.Module):
    def __init__(self, vocab_dim, embed_dim):
        super(SkipGram, self).__init__()
        self.vocab_dim = vocab_dim

        self.embeddings = nn.Embedding(vocab_dim, embed_dim)
        # self.embeddings.weight.data.normal_(0, 0.1)
        self.hidden_out = nn.Linear(embed_dim, vocab_dim)

        self.opt = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        return self.embeddings(x)

    def loss(self, x, y):
        embedded = self(x)
        pred = self.hidden_out(embedded)
        return cross_entropy(pred, y)

    def step(self, x, y):
        self.opt.zero_grad()
        loss = self.loss(x, y)
        loss.backward()
        self.opt.step()
        return loss.detach().numpy()


def train(model,data):
    if torch.cuda.is_available():
        print("GPU train avaliable")
        device =torch.device("cuda")
        model = model.cuda()
    else:
        device = torch.device("cpu")
        model = model.cpu()
    for t in range(2500):
        bx,by = data.sample(8)
        bx, by = tensorify(bx), tensorify(by)
        loss = model.step(bx,by)
        if t%200 == 0:
            print(f"step: {t}  |  loss: {loss}")


if __name__ == "__main__":
    d = process_w2v_data(corpus=corpus, skip_window=2, method="skip_gram")
    m = SkipGram(d.num_word, 2)
    train(m, d)
    show_w2v_word_embedding(m, d, "./visual/results/skipgram.png")