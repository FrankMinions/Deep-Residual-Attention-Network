# Deep-Residual-Attention-Network
We proposed a novel deep residual attention network (DRAN), an end-to-end trainable network which is used to achieve the face mosaic removal. We selected a large-scale CelebFaces attributes (CelebA) dataset, and the first 18000 images are selected as the training set, while the following 100 images are selected for evaluation which is similar to the previous work. The CelebA dataset can be available at [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Requirements
* PyTorch
* Torchvision
* OpenCV

## Training
First, you should run `dran_train.py` to obtain the optimal models, they are saved in the `saved_models` folder. The pre-trained model which the mean PSNR and the mean SSIM are 20.67dB and 0.8509 (with mosaic level 5) respectively was saved in the `saved_models` folder, you can use the test set to test directly.

## Testing
You can just run `demo.py` to obtain the demosaicing images, and they are saved in the `test_results` folder.

## Saved Model
We uploaded the pre-trained model (with mosaic level 5) to Baidu cloud (the model which the mean PSNR and the mean SSIM of test set are 20.67dB and 0.8509, repectively), and you can get it through the following URL and code: URL: [https://pan.baidu.com/s/1LV8O4AZBxDSTRrjBt2A1vw](https://pan.baidu.com/s/1LV8O4AZBxDSTRrjBt2A1vw) Code: `kriz`

## Some Visual Results of Different Demosaicing Methods
![Image text](https://raw.githubusercontent.com/FrankMinions/Deep-Residual-Attention-Network/main/visual_results.png)

The visual results of different face mosaic removal methods are shown above. The first row is the case of mosaic level 5, the second row is 10, and the last row is 15. In the experiment, we compared the performance of DRAN with different mosaic levels, and the extensive experiments show that DRAN can achieve the face mosaic removal with different mosaic levels, and it outperforms the state-of-the-art methods.

## A Drawback of PULSE Concluded in Our Paper
![Image text](https://raw.githubusercontent.com/FrankMinions/Deep-Residual-Attention-Network/main/align_PULSE.png)

In our 100 test face images, when the mosaic level is 5, there are 24 face images are not detected by the predictor for any facial key points. Since the face images need to be aligned in advance, PULSE can only restore the remaining 76 LR face images to corresponding HR face images. 
