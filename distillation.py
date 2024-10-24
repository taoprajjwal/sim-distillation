from typing import List, Optional, Tuple
from torch import nn, Tensor
from torch.nn import functional as F
import torch
from similarity_measures import LinearMeasure, CKA, MSE_w_padding
from transformers import Trainer


class DistilModel(nn.Module):
    _keys_to_ignore_on_save = set()

    def __init__(self, student_model: nn.Module, teacher_model: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        DistilModel._keys_to_ignore_on_save = set(['teacher_model.'+k for k in self.teacher_model.state_dict()])

    def forward(self,*args, **kwargs):
        #if 'return_loss' in x:
          #  del x['return_loss']
        student_output = self.student_model(*args, output_hidden_states=True, **kwargs)
        # absolutely unnecessary, but gives me peace of mind
        with torch.no_grad():
            teacher_output = self.teacher_model(*args, output_hidden_states=True, **kwargs)

        return student_output, teacher_output 

    def train(self, mode: bool = True):
        self.student_model.train(mode)
        return self

    def parameters(self, recurse: bool = True):
        return self.student_model.parameters(recurse=recurse)

    def named_parameters(
            self,
            prefix: str = '',
            recurse: bool = True,
            remove_duplicate: bool = True
    ):
        return self.student_model.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)


class DistillationLoss(nn.Module):
    def __init__(
            self,
            gamma=0.6,
            temperature=2.,
            similarity_measure=None,
            full_similarity=False,
            align_match= None,
            **similarity_measure_kwargs,
    ):
        super().__init__()
        self.register_buffer('gamma', torch.tensor(gamma))
        self.register_buffer('temperature', torch.tensor(temperature))

        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.similarity_measure = similarity_measure
        self.full_similarity = full_similarity
        self.align_match=align_match
        if similarity_measure == 'cosine':
            self.similarity_loss = nn.CosineEmbeddingLoss()
        elif similarity_measure == 'linear':
            self.similarity_loss = LinearMeasure(**similarity_measure_kwargs)
        elif similarity_measure =="cka":
            self.similarity_loss=CKA(**similarity_measure_kwargs)
        elif similarity_measure == "euclidean":
            self.similarity_loss = MSE_w_padding()
        elif similarity_measure is None or similarity_measure == 'none':
            self.similarity_loss = None
        else:
            raise ValueError(f'Unrecognized similarity measure {similarity_measure}')


    def forward(self,
                student_logits,
                teacher_logits,
                student_hidden : Tuple[Tensor, ...]=None,
                teacher_hidden : Tuple[Tensor, ...] =None,
                return_parts=False):
        if (1-self.gamma)>1e-8:
            soft_log_student = F.log_softmax(student_logits / self.temperature, dim=-1)
            soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
            kl_loss = self.kl_div(soft_log_student, soft_teacher)
        else:
            kl_loss=torch.tensor(0, device=student_logits.device)

        output = (1-self.gamma) * self.temperature ** 2 * kl_loss

        sim_loss = None
        if not (self.gamma < 1e-8 or self.similarity_loss is None):

            if self.full_similarity:
                assert (len(teacher_hidden) -1)  % len(student_hidden)-1  == 0
                subsampling_ratio = (len(teacher_hidden) -1)   // (len(student_hidden)-1)
                #student_hidden = student_hidden.view(-1, student_hidden.shape[-1])
                #teacher_hidden = teacher_hidden[:, ::subsampling_ratio].view(-1, teacher_hidden.shape[-1])
                #sim_loss = self.similarity_loss(student_hidden.unsqueeze(1),teacher_hidden.unsqueeze(1))
                teacher_extracted= teacher_hidden[::subsampling_ratio]
                student_extracted= student_hidden
            elif self.align_match is not None:
                if len(self.align_match[0]) != len(self.align_match[1]):
                    raise ValueError("Need to have the same number of layers for align matches")

                student_extracted= torch.stack(student_hidden)[self.align_match[0]]
                teacher_extracted= torch.stack(teacher_hidden)[self.align_match[1]]

            to_align_student = student_extracted.flatten(end_dim=1)
            to_align_teacher = teacher_extracted.flatten(end_dim=1)
            
            if self.similarity_measure == 'cosine':
                sim_loss = self.similarity_loss(to_align_student.flatten(end_dim=1),
                                                to_align_teacher.flatten(end_dim=1),
                                                torch.ones(to_align_student.shape[0] * to_align_student.shape[1],
                                                           device=to_align_student.device).long())
            elif self.similarity_measure in["linear", "cka", "euclidean"]:
                sim_loss= self.similarity_loss(to_align_student, to_align_teacher)
            else:
                raise NotImplementedError

        if sim_loss is not None:
            output += self.gamma * sim_loss
        if return_parts:
            output = (output, kl_loss) + ((sim_loss,) if sim_loss else ())
        return output



class DistilTrainer(Trainer):
    def __init__(self, student_model=None, teacher_model=None, loss_fn=None, temperature=None, include_targets=False, *args, **kwargs):
        model = DistilModel(student_model, teacher_model)
        super().__init__(model=model, *args, **kwargs)
        self.temperature = temperature if temperature else 1.
        self.loss_fn = loss_fn
        self.include_targets=include_targets

    def compute_loss(self, model, inputs, return_outputs=False):
        student_output, teacher_output = model(**inputs)
        loss=self.loss_fn(student_output.logits, teacher_output.logits, student_output.hidden_states, teacher_output.hidden_states)
        if self.include_targets:
            loss+=F.cross_entropy(student_output.logits, inputs["labels"])
        return (loss, student_output) if return_outputs else loss