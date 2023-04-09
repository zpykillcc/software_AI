import torch
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import os
from os.path import join as pjoin
import sys
import copy
import recgnize.detect_face as detect_face
#import detect_face
import random
import recgnize.facenet as facenet
from scipy import misc
import imageio
#import pyttsx3
from PIL import Image, ImageDraw, ImageFont
from numba import cuda

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# face detection parameters
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
dist = []
name_tmp = []
Emb_data = []
image_tmp = []

pic_store = 'source/faceImg'  # "Points to a module containing the definition of the inference graph.")
image_size = 160  # "Image size (height, width) in pixels."
pool_type = 'MAX'  # "The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn = False  # "Enables Local Response Normalization after the first layers of the inception network."
seed = 42,  # "Random seed."
batch_size = None  # "Number of images to process in a batch."
sess = None
#engine = pyttsx3.init()

def scipy_misc_imresize(arr, size, interp='bilinear', mode=None):
	im = Image.fromarray(arr, mode=mode)
	ts = type(size)
	if np.issubdtype(ts, np.signedinteger):
		percent = size / 100.0
		size = tuple((np.array(im.size)*percent).astype(int))
	elif np.issubdtype(type(size), np.floating):
		size = tuple((np.array(im.size)*size).astype(int))
	else:
		size = (size[1], size[0])
	func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
	imnew = im.resize(size, resample=func[interp]) # 调用PIL库中的resize函数
	return np.array(imnew)


class FaceNet(object):
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # face detection parameters
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        dist = []
        name_tmp = []
        Emb_data = []
        image_tmp = []

        pic_store = 'source/faceImg'  # "Points to a module containing the definition of the inference graph.")
        image_size = 160  # "Image size (height, width) in pixels."
        pool_type = 'MAX'  # "The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
        use_lrn = False  # "Enables Local Response Normalization after the first layers of the inception network."
        seed = 42,  # "Random seed."
        batch_size = None  # "Number of images to process in a batch."
        sess = None
        self.inputs = None
        self.pnet = None
        self.rnet = None
        self.onet = None
        self.embeddings = None
        self.G = None
        self.sess = None
        self.images_tmp=None
        self.images_placeholder = None
        self.phase_train_placeholder =None
        # face detection parameters

    def __del__(self):
        print('delete Model')
        self.sess.close()
        device = cuda.get_current_device()
        device.reset()
        cuda.close()

    def to_rgb(self, img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret


    def load_and_align_data(self, image_paths, image_size, margin, gpu_memory_fraction):
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        tmp_image_paths = []
        img_list = []
        path = pjoin(pic_store, image_paths)

        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction, allow_growth = True)
            sess1 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess1.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess1, 'recgnize/model_check_point/')

        if (os.path.isdir(path)):
            for item in os.listdir(path):
                print(item)
                tmp_image_paths.insert(0, pjoin(path, item))
        else:
            tmp_image_paths.append(copy.copy(path))


        print(tmp_image_paths)
        for image in tmp_image_paths:
            img = imageio.imread(os.path.expanduser(image), pilmode='RGB')
            img_size = np.asarray(img.shape)[0:2]
            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            if len(bounding_boxes) < 1:
                tmp_image_paths.remove(image)
                os.remove(image)
                print("can't detect face, remove ", image)
                continue
            det = np.squeeze(bounding_boxes[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = scipy_misc_imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
            image_tmp.append(prewhitened)
        images = np.stack(img_list)
        return images, len(tmp_image_paths), pnet, rnet, onet
    
    def buildModel(self):
        global dist
        global name_tmp
        global Emb_data
        global image_tmp
        dist = []
        name_tmp = []
        Emb_data = []
        image_tmp = []
        # restore facenet model
        print('建立Real-time face recognize模型')
        # Get input and output tensors、
        self.G = tf.Graph()
        with (self.G).as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth = True)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            self.sess = sess
            with sess.as_default():
                print(sess)
                print('开始加载模型')
                # Load the model
                model_checkpoint_path = 'recgnize/model_check_point/20221230-175523'
                facenet.load_model(model_checkpoint_path)
                print(tf.get_default_graph())
                print('建立facenet embedding模型')
                # Get input and output tensors
                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                print('模型建立完毕！')

                print('载入人脸库>>>>>>>>')
                for items in os.listdir(pic_store):
                    emb_data1 = []
                    name_tmp.append(items)
                    #print(items)
                    self.images_tmp, count, self.pnet, self.rnet, self.onet = self.load_and_align_data(items, 160, 44, 0.3)
                    for i in range(9):
                        emb_data = sess.run(self.embeddings,
                                            feed_dict={self.images_placeholder: self.images_tmp, self.phase_train_placeholder: False})
                        emb_data = emb_data.sum(axis=0)
                        emb_data1.append(emb_data)
                    emb_data1 = np.array(emb_data1)
                    emb_data = emb_data1.sum(axis=0)
                    Emb_data.append(np.true_divide(emb_data, 9 * count))

                nrof_images = len(name_tmp)
                print('Images:')
                for i in range(nrof_images):
                    print('%1d: %s' % (i, name_tmp[i]))
                print('')
        return "success"
    
    
    def compare(self ,Faceimage):
        global dist
        global name_tmp
        global Emb_data
        global image_tmp
        faceimg = cv2.imdecode(np.frombuffer(Faceimage, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(faceimg, cv2.COLOR_BGR2GRAY)
        if gray.ndim == 2:
            img = self.to_rgb(gray)
        bounding_boxes, _ = detect_face.detect_face(img, minsize, self.pnet, self.rnet, self.onet, threshold, factor)

        if len(bounding_boxes) < 1:
            print('未检测到人脸！')
            return "None"
        else:
            img_size = np.asarray(faceimg.shape)[0:2]

            nrof_faces = bounding_boxes.shape[0]  # number of faces
            print('找到人脸数目为：{}'.format(nrof_faces))

            for item, face_position in enumerate(bounding_boxes):
                face_position = face_position.astype(int)
                print((int(face_position[0]), int(face_position[1])))
                det = np.squeeze(bounding_boxes[item, 0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - 44 / 2, 0)
                bb[1] = np.maximum(det[1] - 44 / 2, 0)
                bb[2] = np.minimum(det[2] + 44 / 2, img_size[1])
                bb[3] = np.minimum(det[3] + 44 / 2, img_size[0])
                cropped = faceimg[bb[1]:bb[3], bb[0]:bb[2], :]

                cv2.rectangle(faceimg, (face_position[0],
                                    face_position[1]),
                            (face_position[2], face_position[3]),
                            (0, 255, 0), 2)

                aligned = scipy_misc_imresize(cropped, (image_size, image_size), interp='bilinear')
                prewhitened = facenet.prewhiten(aligned)
                image_tmp.append(prewhitened)
                print(len(self.images_tmp))
                image = np.stack(image_tmp)

                
                with (self.G).as_default():
                    sess = self.sess
                    with sess.as_default():
                        #model_checkpoint_path = 'recgnize/model_check_point/20221230-175523'
                        #facenet.load_model(model_checkpoint_path)
                        # for i in range(3):
                        emb_data = sess.run(self.embeddings,
                                    feed_dict={self.images_placeholder: image, self.phase_train_placeholder: False})
                        image_tmp.pop()
                        # print('整体比对结果：',emb_data)
                        # a = images_tmp.pop()
                        # print('单个比对人脸：',emb_data[len(emb_data)-1,:])

                        for i in range(len(Emb_data)):
                            dist.append(
                                np.sqrt(np.sum(np.square(np.subtract(emb_data[len(emb_data) - 1, :], Emb_data[i])))))


                        if min(dist) > 1.03:
                            print(min(dist))
                            print("未收录入人脸识别库")
                            dist = []
                            return "None"
                        else:
                            print(name_tmp)
                            print(min(dist))
                            a = dist.index(min(dist))
                            print(a)
                            name = os.path.splitext(os.path.basename(name_tmp[a]))[0]
                            print(name)
                            dist = []
                            return name
                            # engine.say('你好'+name)
                            # engine.runAndWait()
                            
                
                """
                sess = Sess
                print(sess)
                emb_data = sess.run(self.embeddings,
                                feed_dict={self.images_placeholder: image, self.phase_train_placeholder: False})
                image_tmp.pop()
                # print('整体比对结果：',emb_data)
                # a = images_tmp.pop()
                # print('单个比对人脸：',emb_data[len(emb_data)-1,:])

                for i in range(len(Emb_data)):
                    dist.append(
                        np.sqrt(np.sum(np.square(np.subtract(emb_data[len(emb_data) - 1, :], Emb_data[i])))))

                if min(dist) > 1.03:
                    print(min(dist))
                    print("未收录入人脸识别库")
                    dist = []
                else:
                    print(min(dist))
                    a = dist.index(min(dist))
                    name = os.path.splitext(os.path.basename(name_tmp[a]))[0]
                    print(name)
                    # engine.say('你好'+name)
                    # engine.runAndWait()
                    dist = []
                """



