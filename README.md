# Recurrent Feature Reasoning for Image Inpainting
Accepted in [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Recurrent_Feature_Reasoning_for_Image_Inpainting_CVPR_2020_paper)
## Requirements

Python >= 3.5

PyTorch >= 1.0.0

Opencv2 ==3.4.1

Scipy == 1.1.0

Numpy == 1.14.3

Scikit-image (skimage) == 0.13.1

This is the environment for our experiments. Later versions of these packages might need a few modifications of the code.

Although our method is not limited to any specific CUDA and cudnn version, it's strongly encouraged that you use the latest version of these toolkits. It seems that the RFR-Net could run slowly in older CUDA version due to its recurrent design.

## Pretrained Models

The link to the pretrained model. (Currently, Paris StreetView, CelebA datasets). We are expecting to release the Places2 weights before the end of January, we are sorry for the delay caused by the failure in our storage system.

https://drive.google.com/drive/folders/1EbRSL6SlJqeMliT9qU8V5g0idJqvirZr?usp=sharing

We strongly encourage the users to retrain the models if they are used for academic purpose, to ensure fair comparisons (which has been always desired). Achieving a good performance using the current version of code should not be difficult.

## Results (From Pretrained models)

| ![avatar](https://github.com/jingyuanli001/RFR-Inpainting/blob/master/results/masked_img_321.png) | ![avatar](https://github.com/jingyuanli001/RFR-Inpainting/blob/master/results/img_321.png) | ![avatar](https://github.com/jingyuanli001/RFR-Inpainting/blob/master/results/masked_img_326.png) | ![avatar](https://github.com/jingyuanli001/RFR-Inpainting/blob/master/results/img_326.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![avatar](https://github.com/jingyuanli001/RFR-Inpainting/blob/master/results/masked_img_106.png) | ![avatar](https://github.com/jingyuanli001/RFR-Inpainting/blob/master/results/img_106.png) | ![avatar](https://github.com/jingyuanli001/RFR-Inpainting/blob/master/results/masked_img_586.png) | ![avatar](https://github.com/jingyuanli001/RFR-Inpainting/blob/master/results/img_586.png) |



## Reproducibility

We've checked the reproducibilities of the results in the paper. 
| |Reproducible|
|:----:|:----:|
|Paris StreetView|True|
|CelebA|True|

## Running the program

To perform training or testing, use 
```
python run.py
```
There are several arguments that can be used, which are
```
--data_root +str #where to get the images for training/testing
--mask_root +str #where to get the masks for training/testing
--model_save_path +str #where to save the model during training
--result_save_path +str #where to save the inpainting results during testing
--model_path +str #the pretrained generator to use during training/testing
--target_size +int #the size of images and masks
--mask_mode +int #which kind of mask to be used, 0 for external masks with random order, 1 for randomly generated masks, 2 for external masks with fixed order
--batch_size +int #the size of mini-batch for training
--n_threads +int
--gpu_id +int #which gpu to use
--finetune #to finetune the model during training
--test #test the model
```
For example, to train the network using gpu 1, with pretrained models
```
python run.py --data_root data --mask_root mask --model_path checkpoints/g_10000.pth --batch_size 6 --gpu 1
```
to test the network
```
python run.py --data_root data/images --mask_root data/masks --model_path checkpoints/g_10000.pth --test --mask_mode 2
```
The RFR-Net for filling smaller holes is added. The only difference is the smaller number of pixels fixed in each iteration.  If you are fixing small holes, you can use that version of code, to gain some speep-up.
## Training procedure
To fully exploit the performance of the network, we suggest to use the following training procedure, in specific

1. Train the network, i.e. use the command
```
python run.py
```

2. Finetune the network, i.e. use the command
```
python run.py --finetune --model_path path-to-trained-generator
```

3. Test the model
```
python run.py --test
```
## How long to train the model for

All the descriptions below are under the assumption that the size of mini-batch is 6

For Paris Street View Dataset, train the model for 400,000 iterations and finetune for 200,000 iterations. (600,000 in total)

For CelebA Dataset, train the model for 350,000 iterations and finetune for 150,000 iterations. (500,000 in total)

For Places2 Challenge Dataset, train the model for 2,000,000 iterations and finetune for 1,000,000 iterations. (3,000,000 in total)

## The organization of this code

This part is for people who want to build their own methods based on this code.

The core of this code is the `model.py` file. In specific, it defines the organization of the model, training procedures, loss functions and the parameter updating procedure.

Before we start training/testing, the model and its components are initialized by `initialize_model(self, path=None, train=True)` method which builds a randomly initialized model and tries to load the pretrained parameters. The pipeline of the initialized model is provided in `modules`(The RFR-Net in our case).

After the model is initialized, the method `cuda(self, path=None, train=True)` is called, which moves the model to the gpu given there exists avaliable cuda devices.

When training the network, `train(self, train_loader, save_path, finetune = False)`  is called. This method requires an external dataloader that provides images and masks and the path to save the model. Given the dataloader correctly produces training data, the forward and backward propagation procedures are alternatively performed by calling `forward(self, masked_image, mask, gt_image)` and `update_parameters(self)`. The `forward` method simply feeds the data to the generator network and saves the output results. The `update_parameters(self)` updates the generator and discriminator separately (in our case, the discriminator doesn't exist). When updating the generator and discriminator, we calculate the loss functions and update the parameters.

After training, we can test the data. At this time, a dataloader that provides the test data are required and the path where you want to save the generated results should also be given.
## Building your own method
To modify the method or build your own method based on this code, you can do this by changing the `RFRNet.py` and `model.py` files.
Some examples are given below:

To change the training targets for generator, you can modify the `get_g_loss` method in model.py.

To change the architecture of the model, you might want to modify the `RFRNet.py` file.

To add a discriminator for the RFR-Net, you need to 1.define the discirminator and its optimizer in `initialize_model` and `cuda` methods and 2.define the new loss functions for the discriminator and generator and 3. define parameter updating procedure in `update_D` method.
## Improving the code
This code will be improved constantly. More functions for visualization are still to be developed.
## Citation
If you find the article or code useful for your project, please refer to
```
@InProceedings{Li_2020_CVPR,
	author = {Li, Jingyuan and Wang, Ning and Zhang, Lefei and Du, Bo and Tao, Dacheng},
	title = {Recurrent Feature Reasoning for Image Inpainting},
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June},
	year = {2020}
}
```
## Paper
See the Paper folder



# Penalaran Fitur Berulang untuk Gambar Inpainting
Diterima di [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Recurrent_Feature_Reasoning_for_Image_Inpainting_CVPR_2020_paper)
## Persyaratan

Python >= 3.5

PyTorch >= 1.0.0

Opencv2 = 3.4.1

Scipy == 1.1.0

Numpy == 1.14.3

Scikit-gambar (skimage) == 0.13.1

Ini adalah lingkungan untuk eksperimen kami. Versi selanjutnya dari paket ini mungkin memerlukan beberapa modifikasi kode.

Meskipun metode kami tidak terbatas pada versi CUDA dan cudnn tertentu, sangat disarankan agar Anda menggunakan versi terbaru dari toolkit ini. Tampaknya RFR-Net dapat berjalan lambat di versi CUDA yang lebih lama karena desainnya yang berulang.

## Model Terlatih

Tautan ke model yang telah dilatih sebelumnya. (Saat ini, kumpulan data Paris StreetView, CelebA). Kami mengharapkan untuk merilis bobot Places2 sebelum akhir Januari, kami mohon maaf atas keterlambatan yang disebabkan oleh kegagalan dalam sistem penyimpanan kami.

https://drive.google.com/drive/folders/1EbRSL6SlJqeMliT9qU8V5g0idJqvirZr?usp=sharing

Kami sangat mendorong pengguna untuk melatih kembali model jika digunakan untuk tujuan akademis, untuk memastikan perbandingan yang adil (yang selalu diinginkan). Mencapai kinerja yang baik menggunakan versi kode saat ini seharusnya tidak sulit.

## Hasil (Dari model yang telah dilatih sebelumnya)

| ![avatar](https://github.com/jingyuanli001/RFR-Inpainting/blob/master/results/masked_img_321.png) | ![avatar](https://github.com/jingyuanli001/RFR-Inpainting/blob/master/results/img_321.png) | ![avatar](https://github.com/jingyuanli001/RFR-Inpainting/blob/master/results/masked_img_326.png) | ![avatar](https://github.com/jingyuanli001/RFR-Inpainting/blob/master/results/img_326.png) |
| -------------------------------------------------- ---------- | -------------------------------------------------- ---------- | -------------------------------------------------- ---------- | -------------------------------------------------- ---------- |
| ![avatar](https://github.com/jingyuanli001/RFR-Inpainting/blob/master/results/masked_img_106.png) | ![avatar](https://github.com/jingyuanli001/RFR-Inpainting/blob/master/results/img_106.png) | ![avatar](https://github.com/jingyuanli001/RFR-Inpainting/blob/master/results/masked_img_586.png) | ![avatar](https://github.com/jingyuanli001/RFR-Inpainting/blob/master/results/img_586.png) |

-----------------------------------
TRANSLATED INDONESIAN

## Reproduksibilitas

Kami telah memeriksa reproduktifitas hasil di koran.
| |Direproduksi|
|:----:|:----:|
|Paris StreetView|Benar|
|CelebA|Benar|

## Menjalankan program

Untuk melakukan pelatihan atau pengujian, gunakan
```
python run.py
```
Ada beberapa argumen yang dapat digunakan, yaitu
```
--data_root +str #di mana mendapatkan gambar untuk pelatihan/pengujian
--mask_root +str #di mana mendapatkan masker untuk pelatihan/pengujian
--model_save_path +str #tempat menyimpan model selama pelatihan
--result_save_path +str #tempat menyimpan hasil inpainting selama pengujian
--model_path +str #generator terlatih untuk digunakan selama pelatihan/pengujian
--target_size +int #ukuran gambar dan topeng
--mask_mode +int #masker jenis apa yang akan digunakan, 0 untuk masker eksternal dengan urutan acak, 1 untuk masker yang dibuat secara acak, 2 untuk masker eksternal dengan urutan tetap
--batch_size +int #ukuran mini-batch untuk pelatihan
--n_threads +int
--gpu_id +int #gpu mana yang akan digunakan
--finetune #untuk menyempurnakan model selama pelatihan
--test #test modelnya
```
Misalnya, untuk melatih jaringan menggunakan GPU 1, dengan model yang telah dilatih sebelumnya
```
python run.py --data_root data --mask_root mask --model_path checkpoints/g_10000.pth --batch_size 6 --gpu 1
```
untuk menguji jaringan
```
python run.py --data_root data/images --mask_root data/masks --model_path checkpoints/g_10000.pth --test --mask_mode 2
```
RFR-Net untuk mengisi lubang yang lebih kecil ditambahkan. Satu-satunya perbedaan adalah jumlah piksel yang lebih kecil yang diperbaiki di setiap iterasi. Jika Anda memperbaiki lubang kecil, Anda dapat menggunakan versi kode tersebut, untuk mendapatkan beberapa peningkatan.
## Prosedur pelatihan
Untuk sepenuhnya memanfaatkan kinerja jaringan, kami menyarankan untuk menggunakan prosedur pelatihan berikut, khususnya:

1. Latih jaringan, yaitu gunakan perintah
```
python run.py
```

2. Sempurnakan jaringan, mis. gunakan perintah
```
python run.py --finetune --model_path path-to-trained-generator
```

3. Uji modelnya
```
python run.py --test
```
## Berapa lama melatih model

Semua deskripsi di bawah ini dengan asumsi ukuran mini-batch adalah 6

Untuk Kumpulan Data Paris Street View, latih model untuk 400.000 iterasi dan sempurnakan untuk 200.000 iterasi. (Total 600.000)

Untuk CelebA Dataset, latih model untuk 350.000 iterasi dan sempurnakan untuk 150.000 iterasi. (totalnya 500.000)

Untuk Kumpulan Data Tantangan Places2, latih model untuk 2.000.000 iterasi dan sempurnakan untuk 1.000.000 iterasi. (total 3.000.000)

## Organisasi kode ini

Bagian ini untuk orang yang ingin membangun metode mereka sendiri berdasarkan kode ini.

Inti dari kode ini adalah file `model.py`. Secara khusus, ini mendefinisikan organisasi t
