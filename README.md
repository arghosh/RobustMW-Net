# RobustMW-Net
WACV'21: Do We Really Need Gold Samples for Sample Weighting Under Label Noise?

This is the official code for the paper:
[Do We Really Need Gold Samples for Sample Weighting Under Label Noise?](https://openaccess.thecvf.com/content/WACV2021/papers/Ghosh_Do_We_Really_Need_Gold_Samples_for_Sample_Weighting_Under_WACV_2021_paper.pdf)  
Aritra Ghosh, and Andrew Lan
Presented at [WACV 2021](http://wacv2021.thecvf.com/home).  

If you find this code useful in your research then please cite  
```bash
@InProceedings{Ghosh_2021_WACV,
    author    = {Ghosh, Aritra and Lan, Andrew},
    title     = {Do We Really Need Gold Samples for Sample Weighting Under Label Noise?},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {3922-3931}
}
``` 


## Running RobustMW-Net on benchmark datasets.
To run on cifar10 (cifar100) dataset with uniform (flip2/flip) noise with noise rate 0.4 and noisy (clean) meta dataset with robust loss (or CE loss like MWnet), run
```bash
python trainer.py --dataset cifar10 (cifar100) --corruption_type unif (flip2/flip) --corruption_prob 0.4 --noisy 1 (0) --meta_loss mae (cross)
```


## Acknowledgements
We thank the Pytorch implementation on MWNet(https://github.com/xjtushujun/meta-weight-net).


Contact: Aritra Ghosh (aritraghosh.iem@gmail.com).




