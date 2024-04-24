import matplotlib
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn.utils.clip_grad import clip_grad_norm_

matplotlib.use('Agg')  # NOQA
from matplotlib import pyplot as plt

from model.losses import cosine_sim, cross_perceptual_salience_mapping


class Rehamot(object):
    """
    Rehamot: coss-modal retrieval human motion and text
    """

    def __init__(self,
                 textencoder: DictConfig,
                 motionencoder: DictConfig,
                 loss: DictConfig,
                 sim: str,
                 nfeats: int,
                 learning_rate: float,
                 grad_clip: float,
                 device: str,
                 finetune: bool,
                 enable_momentum: bool,
                 **kwargs):
        self.grad_clip = grad_clip
        self.device = device
        self.enable_momentum = enable_momentum
        if sim == "cpsmapping":
            self.sim = cross_perceptual_salience_mapping
            only_return_cls_token = False
        elif sim == "cosine":
            self.sim = cosine_sim
            only_return_cls_token = True
        else:
            raise NotImplementedError()
        # Build Models
        self.motionencoder = instantiate(
            motionencoder, nfeats=nfeats, only_return_cls_token=only_return_cls_token).to(device)
        self.textencoder = instantiate(textencoder, only_return_cls_token=only_return_cls_token).to(device)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
        num_params = sum(p.numel() for p in self.motionencoder.parameters() if p.requires_grad)
        print(f"Number of parameters in Rehamot's motionencoder: {num_params}")
        num_params = sum(p.numel() for p in self.textencoder.parameters() if p.requires_grad)
        print(f"Number of parameters in Rehamot's textencoder: {num_params}")

        # Loss and Optimizer
        self.criterion = instantiate(loss, sim=self.sim).to(device)

        params = []
        # Fine-tuning with different learning rates for parts of the neural network
        if finetune:
            lr_multiplier = 10
            for prefix, module in [('motion', self.motionencoder), ('text', self.textencoder)]:
                for name, param in module.named_parameters():
                    lr = learning_rate * lr_multiplier if any(name.startswith(
                        s) for s in module.learning_rates_x) else learning_rate
                    params.append({'name': name, 'params': param, 'lr': lr})
        else:
            params += [{'name': name, 'params': param, 'lr': learning_rate}
                       for name, param in self.motionencoder.named_parameters()]
            params += [{'name': name, 'params': param, 'lr': learning_rate}
                       for name, param in self.textencoder.named_parameters()]

        self.params = [param['params'] for param in params]

        self.optimizer = torch.optim.Adam(params)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.motionencoder.state_dict(),
                      self.textencoder.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.motionencoder.load_state_dict(state_dict[0])
        self.textencoder.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.motionencoder.train()
        self.textencoder.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.motionencoder.eval()
        self.textencoder.eval()

    def hard_negative_mining(self, flag=True):
        self.criterion.max_violation = flag

    def forward_emb(self, motion, text, length, **kwargs):
        """Compute the motion and text embeddings
        """
        motion = motion.to(self.device)
        motion_emb, motion_mask = self.motionencoder(motion, length)
        text_emb, text_mask = self.textencoder(text)
        return motion_emb, text_emb, motion_mask, text_mask

    def forward_loss(self, motion_emb, text_emb, motion_mask=None, text_mask=None, idx=None, is_train=True):
        """Compute the loss given pairs of motion and text embeddings
        """
        n = motion_emb[0].size(0)
        loss, drop_num = self.criterion(motion_emb, text_emb, motion_mask, text_mask)
        self.logger.update('Le', loss.item(), n)
        self.logger.update('Drop_num', drop_num, n)
        return loss

    def train_emb(self, motion, text, length, index, **kwargs):
        """One training step given motions and texts.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        motion_emb, text_emb, motion_mask, text_mask = self.forward_emb(motion, text, length)

        # measure similarity in a mini-batch
        if self.Eiters % kwargs['val_step'] == 0 or kwargs['init']:
            self.log_similarity(motion_emb, text_emb, motion_mask, text_mask)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(motion_emb, text_emb, motion_mask, text_mask, index)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
    
    def get_similarity(self, motion_emb, text_emb, motion_mask=None, text_mask=None):
        return self.sim(motion_emb, text_emb, motion_mask, text_mask)

    def log_similarity(self, motion_emb, text_emb, motion_mask, text_mask):
        """Measure similarity in a mini-batch.
        """
        # compute similarity matrix
        # the key is connect with LogCollector
        similarity_matrices = {
            'sim_matrix_inter': self.sim(motion_emb, text_emb, motion_mask, text_mask).detach().cpu().numpy(),
            'sim_matrix_m': self.sim(motion_emb, motion_emb, motion_mask, motion_mask).detach().cpu().numpy(),
            'sim_matrix_t': self.sim(text_emb, text_emb, text_mask, text_mask).detach().cpu().numpy()
        }
        # add similarity matrix and mean similarity to tensorboard
        for name, similarity_matrix in similarity_matrices.items():
            fig = plot_similarity_matrix(similarity_matrix, name)
            self.logger.tb_figure(name, fig, self.Eiters)


def plot_similarity_matrix(similarity_matrix, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(similarity_matrix, cmap='coolwarm', vmin=0, vmax=1)
    plt.colorbar(im)
    plt.tight_layout()
    ax.set_title(title)
    ax.set_xlabel('Index')
    ax.set_ylabel('Index')
    return fig
