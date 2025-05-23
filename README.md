# wsisegqc

[Download models](https://drive.google.com/drive/folders/1P3E9kZDM7A7cM06RR47kywvQCL3X0HJz?usp=sharing))

Download datasets:
[HistoROI preds](https://drive.google.com/file/d/1CDFpoJmKoN34CIxsr88fGWI-XckynF6G/view?usp=drive_link) --
[TCGA-Foreground](https://github.com/abhijeetptl5/historoi/tree/main/tcga_tissue) -- 
[Pen Marker Segmentation](https://drive.google.com/file/d/18QRodOY-D-hdIeplT4D4W22AtbxvqXKT/view?usp=sharing) -- 
[Tissue Folds Segmentation](https://drive.google.com/file/d/16yw0j3C9raapZWfMwXjaYkHffyl4rl31/view?usp=sharing) -- 
[Predictions on TCGA](https://drive.google.com/file/d/1a5DxtOs7nEtmWdjZ74Go3ZLln3ZjgLFO/view?usp=sharing) -- 
[Patch Compare Sets](https://drive.google.com/file/d/1Tis2C4UJqmdXExEQvz0-paOCQeDXYi5k/view?usp=sharing)

![Model predictions](https://github.com/abhijeetptl5/wsisegqc/blob/main/preds_all.png)

All models inference: `python inference.py /path/to/wsis cuda:id`

This saves model predictions in npz format and creates visualization.

Sample visualization:
![Sample Visualization](https://github.com/abhijeetptl5/wsisegqc/blob/main/viz_sample.png)
