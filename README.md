# ConFilMER
Accepted to ICASSP 2025! 

Abstract:

Multimodal Emotion Recognition in Conversations (ERC) plays a crucial role in understanding human language and behavior in real-world scenarios. However, existing research tends to simply concatenate multimodal representations, failing to capture the complex relationships between modalities. Recent advances have shown that Graph Neural Networks (GNNs) are effective in capturing complex data relationships, offering a promising solution for multimodal ERC. Despite this, current GNN-based methods still face challenges, including weak interactions between modalities, neglecting the information entropy of utterances, and erasure of high-frequency signals that capture key variations and discrepancies between closely related nodes. To address these limitations, we propose a GNNs-based multi-frequency propagation method enhanced by contextual filtering for multimodal ERC. Our approach introduces a context filtering module that combines a similarity matrix and an information entropy matrix, enabling GNNs to effectively capture the inherent relationships among utterances and provide sufficient multimodal and contextual modeling. Additionally, our method explores multivariate relationships by recognizing the varying importance of emotional discrepancies and commonalities through multi-frequency signals. Experimental results on two benchmark datasets, IEMOCAP and MELD, demonstrate that our method outperforms the latest (non-)graph-based works.

![image](https://github.com/user-attachments/assets/480b2dbb-0b69-459f-abf9-57f597be905b)

Run on IEMOCAP dataset:

nohup python -u train_our.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size 16 --graph_type='hyper' --epochs=80 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='IEMOCAP' --norm BN  --num_L=5 --num_K=4 > train_our.log 2>&1 &

Run on MELD dataset:

nohup python -u train_one.py --base-model 'GRU' --dropout 0.4 --lr 0.0001 --batch-size 16 --graph_type='hyper' --epochs=15 --graph_construct='direct' --multi_modal --use_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='MELD' --norm BN --num_L=3 --num_K=3 > train_one.log 2>&1 &

