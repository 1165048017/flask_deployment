# An implement of flask to deploy the deep learning models

## keras model

the image classification model is trained in keras and saved by tf.saved_model.builder.SavedModelBuilder
which contains a variable folder and a pb model file.

## server.py

this python file will be runned in server.

## client.py

this python file will be runed in client, it will upload a image to server and get a 
predict result. using these code to test success or not

```bash
python client.py  --file=filename

```
More detail information please go to https://zhuanlan.zhihu.com/p/83894642
