from flask import Flask, request, jsonify, render_template

import tarfile
from six.moves import urllib

import cv2
from demo import main
import tensorflow as tf

import os
from network import UNet as UNet_H

import torch
from torchvision import transforms
from PIL import Image, ImageOps

import numpy as np

app = Flask(__name__)

# Height
model_h = UNet_H(128)
pretrained_model = torch.load('models/model_ep_48.pth.tar', map_location=torch.device('cpu'))
model_h.load_state_dict(pretrained_model["state_dict"])


class ResizeWithPadding(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        old_size = img.size
        ratio = max(self.size[0] / old_size[0], self.size[1] / old_size[1])
        new_size = tuple([int(x * ratio) for x in old_size])

        img = img.resize(new_size, Image.ANTIALIAS)

        delta_w = self.size[0] - new_size[0]
        delta_h = self.size[1] - new_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        img = ImageOps.expand(img, padding)

        return img


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file is allowed
    if file and allowed_file(file.filename):
        # Read the image from the file
        img = Image.open(file.stream)

        # Get the input height from the request
        region = (request.form['region'])
        trans_h = transforms.Compose([
            ResizeWithPadding((128, 128)),  # Resize the image with padding
            transforms.ToTensor(),  # Convert the image to tensor
        ])
        img_h = trans_h(img)

        # Add a batch dimension to the input tensor
        img_h = img_h.unsqueeze(0)

        pred = []

        model_h.eval()

        with torch.no_grad():
            m_p, j_p, h_p = model_h(img_h)

            pred.append(h_p.item())

        pred = (np.array(pred))
        height = int(np.mean(abs(100 * (pred))))

        # Get the selected gender from the request
        gender = request.form['gender']

        # Process the image
        processed_result = process_image(img, height, gender, region)

        # Return the processed image
        return jsonify(processed_result)
    else:
        return jsonify({'error': 'File type not allowed'})


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


def process_image(image, height, gender, region):
    # Rest of your image processing code here...

    class DeepLabModel(object):
        """Class to load deeplab model and run inference."""

        INPUT_TENSOR_NAME = 'ImageTensor:0'
        OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
        INPUT_SIZE = 513
        FROZEN_GRAPH_NAME = 'frozen_inference_graph'

        def __init__(self, tarball_path):
            # """Creates and loads pretrained deeplab model."""
            self.graph = tf.Graph()
            graph_def = None
            # Extract frozen graph from tar archive.
            tar_file = tarfile.open(tarball_path)
            for tar_info in tar_file.getmembers():
                if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                    file_handle = tar_file.extractfile(tar_info)
                    graph_def = tf.GraphDef.FromString(file_handle.read())
                    break

            tar_file.close()

            if graph_def is None:
                raise RuntimeError('Cannot find inference graph in tar archive.')

            with self.graph.as_default():
                tf.import_graph_def(graph_def, name='')

            self.sess = tf.Session(graph=self.graph)

        def run(self, image):
            """Runs inference on a single image.

            Args:
            image: A PIL.Image object, raw input image.

            Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
            """
            width, height = image.size
            resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
            target_size = (int(resize_ratio * width), int(resize_ratio * height))
            resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
            batch_seg_map = self.sess.run(
                self.OUTPUT_TENSOR_NAME,
                feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
            seg_map = batch_seg_map[0]
            return resized_image, seg_map

    def create_pascal_label_colormap():
        """Creates a label colormap used in PASCAL VOC segmentation benchmark.

        Returns:
        A Colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 3), dtype=int)
        ind = np.arange(256, dtype=int)

        for shift in reversed(range(8)):
            for channel in range(3):
                colormap[:, channel] |= ((ind >> channel) & 1) << shift
            ind >>= 3

        return colormap

    def label_to_color_image(label):
        """Adds color defined by the dataset colormap to the label.

        Args:
        label: A 2D array with integer type, storing the segmentation label.

        Returns:
        result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

        Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
        """
        if label.ndim != 2:
            raise ValueError('Expect 2-D input label')

        colormap = create_pascal_label_colormap()

        if np.max(label) >= len(colormap):
            raise ValueError('label value too large.')

        return colormap[label]

    ## setup ####################

    LABEL_NAMES = np.asarray([
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ])

    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

    MODEL_NAME = 'xception_coco_voctrainval'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

    _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    _MODEL_URLS = {
        'mobilenetv2_coco_voctrainaug':
            'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
        'mobilenetv2_coco_voctrainval':
            'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
        'xception_coco_voctrainaug':
            'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
        'xception_coco_voctrainval':
            'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    }
    _TARBALL_NAME = _MODEL_URLS[MODEL_NAME]

    model_dir = 'deeplab_model'
    if not os.path.exists(model_dir):
        tf.gfile.MakeDirs(model_dir)

    download_path = os.path.join(model_dir, _TARBALL_NAME)
    if not os.path.exists(download_path):
        print('downloading model to %s, this might take a while...' % download_path)
        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                                   download_path)
        print('download completed! loading DeepLab model...')

    MODEL = DeepLabModel(download_path)
    print('model loaded successfully!')

    #######################################################################################

    back = cv2.imread('sample_data/input/background.jpeg', cv2.IMREAD_COLOR)

    res_im, seg = MODEL.run(image)

    seg = cv2.resize(seg.astype(np.uint8), image.size)
    mask_sel = (seg == 15).astype(np.float32)
    mask = 255 * mask_sel.astype(np.uint8)

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    res = cv2.bitwise_and(img, img, mask=mask)
    bg_removed = res + (255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    processed_result = main(bg_removed, height, gender, region, None)

    return processed_result


if __name__ == '__main__':
    app.run(debug=True)
