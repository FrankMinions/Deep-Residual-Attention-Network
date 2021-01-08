# Deep-Residual-Attention-Autoencoder
We proposed a deep residual attention autoencoder (DRAA), an end-to-end trainable network which is used to achieve the face mosaic removal. We selected a large-scale CelebFaces attributes (CelebA) dataset, and the first 18000 images are selected as the training set, while the following 100 images are selected for evaluation which is similar to the previous work. The CelebA dataset can be available at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.
## Requirements
* PyTorch
* torchvision
## Training
''' First, you should run draa_train.py to obtain the optimal models, they are saved in the "saved_models" folder. The pre-trained model which the mean PSNR and the mean SSIM are 20.60dB and 0.8485 respectively was saved in the "saved_models" folder, you can use the test set to test directly. '''
## Testing
''' You can just run demo.py to obtain the demosaicing images, and they are saved in the "test_results" folder. '''
## Saved model
We uploaded the pre-trained model to Baidu cloud (the model which the mean PSNR and the mean SSIM of test set are 20.60dB and 0.8485, repectively), and you can get it through the following URL and code: URL: https://pan.baidu.com/s/1LV8O4AZBxDSTRrjBt2A1vw Code: kriz 
## Some visual results of different demosaicing methods
![Image text](https://raw.githubusercontent.com/FrankMinions/Deep-Residual-Attention-Autoencoder/main/visual_results.png)
## A drawback of PULSE concluded in our paper
![Image text](https://raw.githubusercontent.com/FrankMinions/Deep-Residual-Attention-Autoencoder/main/align_PULSE.png)
In our 100 test face images, there are 24 face images are not detected by the predictor for any facial key points. Since the face images need to be aligned in advance, PULSE can only restore the remaining 76 LR face images to corresponding HR face images.
