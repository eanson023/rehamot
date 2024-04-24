import torch
import torch.nn as nn
import numpy as np

class MPNetSimilarity:
    def __init__(self, sent_model:nn.Module):
        self.model = sent_model

    def _similarity(self, emb1, emb2):
        emb1 = emb1.unsqueeze(0)
        emb2 = emb2.unsqueeze(-1)
        sim = torch.mm(emb1, emb2)
        return sim.cpu().detach().numpy()

    def compute_score(self, dset, query):
        """
        dset: List[List[str]]
        query: str
        """
        _, dset_embds = dset
        scores = np.zeros(len(dset_embds))
        query_embs = self.model([query])
        for i, sent_embs in enumerate(dset_embds):
            sims = torch.mm(query_embs, sent_embs.T).clamp(min=0).detach().cpu().numpy()
            # scores[i] = max([np.clip(self._similarity(query_emb, t_emb), a_min=0, a_max=1.0) for t_emb in sent_embs])
            scores[i] = sims.max()

        return scores

if __name__ == "__main__":
    a = torch.randn((1, 1024))
    b = torch.randn((1, 1024))
    c = torch.mm(a,b.T)
    print(c)
    print(a@b.T)
