# DeepNets implemented using with Keras framework

Please install the dependencies inside the 'requirements.txt' file to run any model.

### Two-Stream network 

[Python 2.7, Tensorflow 1.8, Keras 2.0]

Two-Stream Network (2SCNN) from

```
Simonyan, K., & Zisserman, A. (2014).
Two-stream convolutional networks for action recognition in videos.
In Advances in neural information processing systems (pp. 568-576).
```

with joint training as in

```
Ma, M., Fan, H., & Kitani, K. M. (2016).
Going deeper into first-person activity recognition.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 1894-1903).
```

Two CNN-M-2048 networks from

```
Chatfield, K., Simonyan, K., Vedaldi, A., & Zisserman, A. (2014).
Return of the devil in the details: Delving deep into convolutional nets.
arXiv preprint arXiv:1405.3531.
```

are used as backbone networks for each stream. The parameters of the network are included in the file 'parameters.json'.
 
| Evaluation method  | Test accuracy | Test Macro-F1  |
| ------------------ |:-------------:| --------------:|
| Method 1   	     | xx.xx% 	     | xx.xx 	      |
| Method 2   	     | 31.30%        | 3.69           |

Both methods do the evaluation over videos.

* Method 1: As proposed in the paper, 25 equidistant frames are sampled from a video. 5 crops (4 corners and center) are obtained from each frame and then mirroring is applied, obtaining 10 images per each of the original image. The optical flow stack is created by taking the previous 5 frames and the next 4 frames. The result for the video is given by majority voting between the 25*10=250 inputs.

* Method 2: Non-overlapping inputs are obtained from a video (stack of optical flow and image, the middle one in the set of L frames), except for the last input: if there are some spare frames (less than L, necessary to create a stack of optical flow) the last 10 frames of the video are sampled to create another input.
