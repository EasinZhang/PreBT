# Pre-trained Battery Transformer (PreBT)

This code is associated with the paper: ["A Multi-task General Model for Lithium-ion Battery Health Management."](https://github.com/EasinZhang/PreBT) If you find this code useful, please cite it. DOI:


> :relaxed: **This paper is currently under review**.


## Let's Start
Install the corresponding dependency packages. We have provided a ```requirements.txt``` file:
```
pip install -r requirements.txt
```

## Pre-training
### Reconstruct

```commandline
python src/main.py --comment "pretraining" --name PreBT --records_file records.xls  --val_ratio 0.01 --epochs 100 --lr 0.00005 --optimizer RAdam --batch_size 64 --model BBTE --pos_encoding learnable --d_model 256 --dim_feedforward 512 --num_layers 8 --normalization_layer LayerNorm
```
### SOC/SOE/SOH
:star: Change the corresponding command according to your task.
```commandline
python src/main.py --comment "finetune for SOC" --name PreBT_SOC_finetuned --records_file SOC_records.xls --data_dir Yourpath\train --epochs 5 --lr 0.001 --d_model 256 --dim_feedforward 512 --num_layers 8 --load_model YourPath\PreBT\checkpoints\model_last.pth --task regression --change_output --batch_size 64 --val_ratio 0.05 --data_class socdataset --norm_from YourPath\PreBT\normalization.pickle --freeze --data_window_len 20
```

## Test
:star: Change the corresponding command according to your task.
```commandline
python src/main.py --comment "test" --name PreBT_test --data_dir Yourpath\data\SOH  --val_ratio 0 --batch_size 64 --model BBTE --pos_encoding learnable --d_model 256 --dim_feedforward 512 --num_layers 8 --normalization_layer LayerNorm --load_model YourPath\PreBT\checkpoints\model_last.pth --task regression --data_class sohdataset --norm_from YourPath\PreBT\normalization.pickle --test_only testset
```

## Data template
The data should be saved as a CSV file.
>[!Note]  'Capacity' means charge throughput. If there are discontinuous time periods within the same CSV file, add a row with the [10000] delimiter between these time periods to separate the discontinuous data



| Voltage |  Capacity   | Temperature | Current |
|:-------:|:-----------:|:-----------:|:-------:|
| 2.7139  |      0      |     30      | 5.4985  |
| 3.0998  |   0.0076    |     30      |   5.4985   |
|   ...   |     ...     |     ...     |   ...   |
|   3.4433   |     0.0076     |     30      |   5.4988   |

## Acknowledgement

We thank the following Github projects and related personnel for their efforts:
* https://github.com/pytorch/examples/blob/main/word_language_model/model.py
* https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_self_supervised/patchtst_pretrain.py
* https://github.com/gzerveas/mvts_transformer?tab=readme-ov-file
* https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TST.py
* https://github.com/hujie-frank/SENet


## Contributors

If you have any questions or concerns, please feel free to contact:
* Yixing Zhang (YixingZhang@cqu.edu.cn)
* Fei Feng (feifeng@cqu.edu.cn)
