# A Bottom-Up Clustering Approach to Unsupervised Person Re-identification

Pytorch implementation for our paper [[Link]](https://vana77.github.io/vana77.github.io/images/AAAI19.pdf).
This code is based on the [Open-ReID](https://github.com/Cysu/open-reid) library.

## Updates
We found a bug and fixed it by deleting line 269~270 in `reid/bottom_up.py`. These two lines are used to initialize the memory vector by the averaged feature of each new cluster. However, we made a mistake and used the feature from the previous iteration to initialize the memory. This would produce redundant classes and mismatch the memory vector and the cluster label. For example, the 89th cluster might contain a memory vector from the 90th image by mistake. Coincidentally, in loading datasets, images of the same identity are adjacent in indexes. Therefore, the mistake will teach the model to pull the features of the adjacent two clusters close to each other, which utilizes some supervised information involuntarily. We corrected this mistake by initializing the memory vector by zeros at each iteration. We are sorry for the mistake we made and for any inconvenience caused. We have updated our paper [[Link]](https://vana77.github.io/vana77.github.io/images/AAAI19.pdf) and code. 

The updated performances on the four re-ID datasets are listed below:

|       | rank-1     | rank-5     | rank-10     | mAP     |
| ---------- | :-----------:  | :-----------: |:-----------:  | :-----------: |
| Market-1501     | 61.9     | 73.5     |78.2     | 29.6     |
| DukeMTMC-reID     | 40.4     | 52.5     |58.2     | 22.1     |
| MARS     | 55.1     | 68.3     |72.8     | 29.4     |
| DukeMTMC-VideoReID     | 74.8     | 86.8     |89.7     | 66.7     |


## Preparation
### Dependencies
- Python 3.6
- PyTorch (version >= 0.4.1)
- h5py, scikit-learn, metric-learn, tqdm

### Download datasets 
- DukeMTMC-VideoReID: This [page](https://github.com/Yu-Wu/DukeMTMC-VideoReID) contains more details and baseline code.
- MARS: [[Google Drive]](https://drive.google.com/open?id=1m6yLgtQdhb6pLCcb6_m7sj0LLBRvkDW0)   [[BaiduYun]](https://pan.baidu.com/s/1mByTdvXFsmobXOXBEkIWFw).
- Market-1501: [[Direct Link]](http://108.61.70.170/share/market1501.tar) [[Google Drive]](https://drive.google.com/file/d/1kbDAPetylhb350LX3EINoEtFsXeXB0uW/view?usp=sharing)
- DukeMTMC-reID: [[Direct Link]](http://108.61.70.170/share/duke.tar)[[Google Drive]](https://drive.google.com/file/d/17mHIip2x5DXWqDUT97aiqKsrTQvSI830/view?usp=sharing)
- Move the downloaded zip files to `./data/` and unzip here.

## Usage

```shell
sh ./run.sh
```
`--size_penalty` parameter lambda to balance the diversity regularization term.

`--merge_percent` percent of data to merge at each iteration.
We utilize 1 GTX-1080TI GPU for training on image-based datasets and 2 GTX-1080TI GPUs for training on video-based datasets.

## Citation

Please cite the following paper in your publications if it helps your research:

    @inproceedings{lin2019bottom,
        title={A bottom-up clustering approach to unsupervised person re-identification},
        author={Lin, Yutian and Dong, Xuanyi and Zheng, Liang and Yan, Yan and Yang, Yi},
        booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
        volume={2},
        pages={1--8},
        year={2019}
        }





