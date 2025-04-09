# ENHANCED MULTIMODAL EMOTION RECOGNITION IN CONVERSATIONS VIA CONTEXTUAL FILTERING AND MULTI-FREQUENCY GRAPH PROPAGATION
## Accepted by ICASSP 2025! 

### Dataset

The raw data can be found at [IEMOCAP](https://sail.usc.edu/iemocap/ "IEMOCAP") and [MELD](https://github.com/SenticNet/MELD "MELD").

In our paper, we use pre-extracted features. The multimodal features (including RoBERTa-based and GloVe-based textual features) are available at [here](https://www.dropbox.com/sh/4b21lympehwdg4l/AADXMURD5uCECN_pvvJpCAy9a?dl=0 "here").

We also provide some pre-trained checkpoints on RoBERTa-based IEMOCAP at [here](https://www.dropbox.com/sh/gd32s36v7l3c3u9/AACOipUURd7gEbEcdYSrmP-0a?dl=0 "here").


Abstract:

Multimodal Emotion Recognition in Conversations (ERC) plays a crucial role in understanding human language and behavior in real-world scenarios. However, existing research tends to simply concatenate multimodal representations, failing to capture the complex relationships between modalities. Recent advances have shown that Graph Neural Networks (GNNs) are effective in capturing complex data relationships, offering a promising solution for multimodal ERC. Despite this, current GNN-based methods still face challenges, including weak interactions between modalities, neglecting the information entropy of utterances, and erasure of high-frequency signals that capture key variations and discrepancies between closely related nodes. To address these limitations, we propose a GNNs-based multi-frequency propagation method enhanced by contextual filtering for multimodal ERC. Our approach introduces a context filtering module that combines a similarity matrix and an information entropy matrix, enabling GNNs to effectively capture the inherent relationships among utterances and provide sufficient multimodal and contextual modeling. Additionally, our method explores multivariate relationships by recognizing the varying importance of emotional discrepancies and commonalities through multi-frequency signals. Experimental results on two benchmark datasets, IEMOCAP and MELD, demonstrate that our method outperforms the latest (non-)graph-based works.

![image](https://github.com/user-attachments/assets/480b2dbb-0b69-459f-abf9-57f597be905b)

Run on IEMOCAP dataset:

nohup python -u train_our.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size 16 --graph_type='hyper' --epochs=80 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='IEMOCAP' --norm BN  --num_L=5 --num_K=4 > train_our.log 2>&1 &

Run on MELD dataset:

nohup python -u train_one.py --base-model 'GRU' --dropout 0.4 --lr 0.0001 --batch-size 16 --graph_type='hyper' --epochs=15 --graph_construct='direct' --multi_modal --use_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='MELD' --norm BN --num_L=3 --num_K=3 > train_one.log 2>&1 &

## Citation

If you find the repository useful, please cite the following paper:

```bibtex
@inproceedings{zhao2025enhanced,
  title={Enhanced Multimodal Emotion Recognition in Conversations via Contextual Filtering and Multi-Frequency Graph Propagation},
  author={Zhao, Huan and Gao, Yingxue and Chen, Haijiao and Li, Bo and Ye, Guanghui and Zhang, Zixing},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```
