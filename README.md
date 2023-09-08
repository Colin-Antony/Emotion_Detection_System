# Emotion Detection System
DL model that detects what emotion one is displaying.

This model was trained on the FER 2013 dataset. It is a dataset that comprises of 35887 48×48 gray-scale images images.  
![FER2013-training-data-distribution](https://github.com/Colin-Antony/Emotion_Detection_System/assets/123204978/dc9fc5d5-6f67-4e8c-9d49-f6024cbde92d)  

As one can see it is an unbalanced dataset.  
Run the image_loader.py file to test the model. Training dataset will be available online.  
Model accuracy is low, around 65 percent. Lot of room to improve.  
Notes to improve:  
1) Use stratifying while splitting the dataset. Chances of skewing are high due to the number of images in a class being scarce
2) Try a different model after doing the above step
3) Remove the disgust class fully and try.
4) Goal. Aim for 75 percent

