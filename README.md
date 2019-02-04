# A Bottom-Up Clustering Approach to Unsupervised Person Re-identification

Pytorch implementation for our paper [[Link]](https://vana77.github.io/vana77.github.io/images/AAAI19.pdf).
This code is based on the [Open-ReID](https://github.com/Cysu/open-reid) library.

## Preparation
### Dependencies
- Python 3.6
- PyTorch (version >= 0.4.1)
- h5py, scikit-learn, metric-learn, tqdm

### Download datasets 
- DukeMTMC-VideoReID: [[Direct Link]](http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-VideoReID.zip)  [[Google Drive]](https://drive.google.com/open?id=1Fdu5GK-C7P8M9QiLbiQNyT_RUFt8oFco)  [[BaiduYun]](https://pan.baidu.com/s/1qL39rnjTjyzjqaD-Wuv8KQ). This [page](https://github.com/Yu-Wu/DukeMTMC-VideoReID) contains more details and baseline code.
- MARS: [[Google Drive]](https://drive.google.com/open?id=1m6yLgtQdhb6pLCcb6_m7sj0LLBRvkDW0)   [[BaiduYun]](https://pan.baidu.com/s/1mByTdvXFsmobXOXBEkIWFw).
- Market-1501: [[Direct Link]](http://45.32.69.75/share/market1501.tar)
- DukeMTMC-reID: [[Direct Link]](http://45.32.69.75/share/duke.tar)
- Move the downloaded zip files to `./data/` and unzip here.

## Usage

```shell
sh ./run.sh
```
`--init_epochs` the number of init epochs.

`--later_epochs` the number of init epochs (set to 2 for image-based datasets, 5 for video-based datasets).

`--size_penalty` parameter lambda to balance the diversity regularization term.

`--merge_percent` percent of data to merge at each iteration.

## Citation

Please cite the following paper in your publications if it helps your research:
    
    @inproceedings{lin2019aBottom,
        title     = {A Bottom-Up Clustering Approach to Unsupervised Person Re-identification},
        author    = {Lin, Yutian and Dong, Xuanyi and Zheng, Liang and Yan, Yan and Yang, Yi},
        booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},
        year      = {2019}
    }





