# BalancedMetaSoftmax - Instance Segmentation

Code for the paper "Balanced Meta-Softmax for Long-Tailed Visual Recognition" on LVIS-0.5 dataset. The repository is developed based on Detectron2.

**[Balanced Meta-Softmax for Long-Tailed Visual Recognition](https://papers.nips.cc/paper/2020/file/2ba61cc3a8f44143e1f2f13b2b729ab3-Paper.pdf)**  
Jiawei Ren, Cunjun Yu, Shunan Sheng, Xiao Ma, Haiyu Zhao, Shuai Yi, Hongsheng Li  
NeurIPS 2020

## Snapshot
```python

def balanced_softmax_loss(self):
    """
    Sigmoid variant of Balanced Softmax
    """
    self.n_i, self.n_c = self.pred_class_logits.size()
    self.target = self.get_expanded_label()

    njIn = self.freq_info.type_as(self.pred_class_logits)

    weight = (1. - njIn) / njIn     # Discard the constant 1/(k-1) to keep log(weight) mostly positive
    weight = weight.unsqueeze(0).expand(self.n_i, -1)

    fg_ind = self.gt_classes != self.n_c
    self.pred_class_logits[fg_ind] = (self.pred_class_logits - weight.log())[fg_ind]    # Only apply to  FG samples

    cls_loss = F.binary_cross_entropy_with_logits(self.pred_class_logits, self.target,
                                                  reduction='none')

    return torch.sum(cls_loss) / self.n_i

```

## Installation 
Clone this repo by `git clone https://github.com/Majiker/BalancedMetaSoftmax-InstanceSeg.git`

Install detectron2 by `python -m pip install -e BalancedMetaSoftmax-InstanceSeg`

To set up LVIS-0.5 dataset, please follow the procedures described [here](datasets/README.md).

Please install `higher` in order to run BALMS:
```bash
pip install higher
```

## Training

You may want to download a pretrained model [here](https://drive.google.com/file/d/1OlGyvDwBwSaU3ohiK8Z8yH3mLqOW5aJz/view?usp=sharing) and put it in `pretrains` folder. 
Otherwise you may train the base model by yourself using the following command with 8 GPUs:
```bash
python ./projects/BALMS/train_net.py  --config-file ./projects/BALMS/configs/feature/sigmoid_resampling_mask_rcnn_R_50_FPN_1x.yaml --num-gpus 8
```

After obtaining the base model and putting it in `pretrains`, train the model with the following command:
```bash
python ./projects/BALMS/train_net.py  --config-file ./projects/BALMS/configs/classifier/balms_decouple_resampling_mask_rcnn_R_50_FPN_1x.yaml --num-gpus 8
```

## Evaluation

Model evaluation can be done using the following command:
```bash
python ./projects/BALMS/train_net.py --config-file ./projects/BALMS/configs/classifier/balms_decouple_resampling_mask_rcnn_R_50_FPN_1x.yaml--eval-only MODEL.WEIGHTS /path/to/model_checkpoint
```

## Experiment Results
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom", align="left">Backbone</th>
<th valign="bottom", align="left">Method</th>
<th valign="bottom">AP</th>
<th valign="bottom">AP.r</th>
<th valign="bottom">AP.c</th>
<th valign="bottom">AP.f</th>
<th valign="bottom">AP.bbox</th>
<th valign="bottom">download</th>

<!-- TABLE BODY -->
<tr>
<td align="left">MaskRCNN-R50-FPN</td>
<td align="left">Baseline</td>
<td align="center">24.1</td>
<td align="center">13.4</td>
<td align="center">24.3</td>
<td align="center">28.1</td>
<td align="center">23.4</td>
<td align="center"><a href="https://drive.google.com/file/d/1OlGyvDwBwSaU3ohiK8Z8yH3mLqOW5aJz/view?usp=sharing">model</a>&nbsp;
    <a href="https://drive.google.com/file/d/1Ibe6XwVqg_ICr5X_Z4PgVKWjRMtnNPVF/view?usp=sharing">metrics</a>
</tr>
<tr>
<td align="left">MaskRCNN-R50-FPN</td>
<td align="left">BalancedSoftmax</td>
<td align="center">26.4</td>
<td align="center">15.9</td>
<td align="center">27.3</td>
<td align="center">29.6</td>
<td align="center">25.9</td>
<td align="center"><a href="https://drive.google.com/file/d/12FaO94sQhfM2MWEhWkep0-2PxlWtHBTm/view?usp=sharing">model</a>&nbsp;
    <a href="https://drive.google.com/file/d/1lFC3Rzxba06zJGIif3f19BtA-a4oOR-H/view?usp=sharing">metrics</a>
</td>
</tr>
<tr>
<td align="left">MaskRCNN-R50-FPN</td>
<td align="left">BALMS</td>
<td align="center">27.0</td>
<td align="center">17.3</td>
<td align="center">28.1</td>
<td align="center">29.5</td>
<td align="center">26.4</td>
<td align="center"><a href="https://drive.google.com/file/d/15HJiq_ZvqGLnW_g4q9FdLSllWwL-ig-P/view?usp=sharing">model</a>&nbsp;
    <a href="https://drive.google.com/file/d/1OtZCKP5hZNBUv2XMSVqbbMBrybzHmio_/view?usp=sharing">metrics</a>
</td>
</tr>
</tbody></table>


## Cite BALMS
```bibtex
@inproceedings{
    Ren2020balms,
    title={Balanced Meta-Softmax for Long-Tailed Visual Recognition},
    author={Jiawei Ren and Cunjun Yu and Shunan Sheng and Xiao Ma and Haiyu Zhao and Shuai Yi and Hongsheng Li},
    booktitle={Proceedings of Neural Information Processing Systems(NeurIPS)},
    month = {Dec},
    year={2020}
}
```

## Visual Recognition

For BALMS on visual recognition, please try out this [**repo**](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification).


## Reference 
- Based on [Detectron2](https://github.com/facebookresearch/detectron2)
- LVIS-v0.5 class frequency is from [Equalization Loss](https://github.com/tztztztztz/eql.detectron2)

