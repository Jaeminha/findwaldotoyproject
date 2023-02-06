import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import keras.layers as layers
import keras.optimizers as optimizers
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau
import seaborn as sns
from PIL import Image
from skimage.transform import resize

import threading, random, os

##gpus = tf.config.experimental.list_logical_devices('GPU')

##print('>>> Tensorflow Version: {}'.format(tf.__version__))
##print('>>> Load GPUS: {}'.format(gpus))

imgs = np.load('dataset/imgs_uint8.npy').astype(np.float32) / 255.
labels = np.load('dataset/labels_uint8.npy').astype(np.float32) / 255.
waldo_sub_imgs = np.load('dataset/waldo_sub_imgs_uint8.npy') / 255.
waldo_sub_labels = np.load('dataset/waldo_sub_labels_uint8.npy') / 255.

print(imgs.shape, labels.shape)
print(waldo_sub_imgs.shape, waldo_sub_labels.shape)

BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')

imgs = np.load(os.path.join(DATASET_DIR, 'imgs_uint8.npy'), allow_pickle=True).astype(np.float32) / 255.
labels = np.load(os.path.join(DATASET_DIR, 'labels_uint8.npy'), allow_pickle=True).astype(np.float32) / 255.
waldo_sub_imgs = np.load(os.path.join(DATASET_DIR, 'waldo_sub_imgs_uint8.npy'), allow_pickle=True) / 255.
waldo_sub_labels = np.load(os.path.join(DATASET_DIR, 'waldo_sub_labels_uint8.npy'), allow_pickle=True) / 255.


# image visualize
# ##imgs[0].shape, waldo_sub_imgs[0].shape
#
# print('>>> Image Visualization')
#
# plt.figure(figsize=(10, 10))
# plt.title("Whole Shape: {}".format(imgs[0].shape))
# plt.imshow(imgs[0])
#
# plt.figure(figsize=(2, 2))
# plt.title("Waldo Shape: {}".format(waldo_sub_imgs[0].shape))
# plt.imshow(waldo_sub_imgs[0])
# plt.show()


PANNEL_SIZE = 224


class BatchIndices(object):
    """
    Generates batches of shuffled indices.
    # Arguments
        n: number of indices
        bs: batch size
        shuffle: whether to shuffle indices, default False

    """

    def __init__(self, n, bs, shuffle=False):
        self.n, self.bs, self.shuffle = n, bs, shuffle
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.idxs = (np.random.permutation(self.n)
                     if self.shuffle else np.arange(0, self.n))
        self.curr = 0

    def __next__(self):
        with self.lock:
            if self.curr >= self.n: self.reset()
            ni = min(self.bs, self.n - self.curr)
            res = self.idxs[self.curr:self.curr + ni]
            self.curr += ni
            return res

# iterator check
# sample_train = imgs[:100]
# total_count = sample_train.shape[0]
# batch_size = 10
#
# print('>>> No Shuffle')
# idx_gen = BatchIndices(total_count, batch_size, False)
# print(idx_gen.__next__())
# print(idx_gen.__next__())
# print(idx_gen.__next__())
#
# print(' ')
# print('>>> Shuffle')
# idx_gen = BatchIndices(total_count, batch_size, True)
# print(idx_gen.__next__())
# print(idx_gen.__next__())
# print(idx_gen.__next__())

class segm_generator(object):
    """
    Generates batches of sub-images.
    # Arguments
        x: array of inputs
        y: array of targets
        bs: batch size
        out_sz: dimension of sub-image
        train: If true, will shuffle/randomize sub-images
        waldo: If true, allow sub-images to contain targets.
    """
    def __init__(self, x, y, bs=64, out_sz=(224,224), train=True, waldo=True):
        self.x, self.y, self.bs, self.train = x,y,bs,train
        self.waldo = waldo
        self.n = x.shape[0]
        self.ri, self.ci = [], []
        for i in range(self.n):
            ri, ci, _ = x[i].shape
            self.ri.append(ri), self.ci.append(ci)
        self.idx_gen = BatchIndices(self.n, bs, train)
        self.ro, self.co = out_sz
        self.ych = self.y.shape[-1] if len(y.shape)==4 else 1

    def get_slice(self, i,o):
        start = random.randint(0, i-o) if self.train else (i-o)
        return slice(start, start+o)

    def get_item(self, idx):
        slice_r = self.get_slice(self.ri[idx], self.ro)
        slice_c = self.get_slice(self.ci[idx], self.co)
        x = self.x[idx][slice_r, slice_c]
        y = self.y[idx][slice_r, slice_c]
        if self.train and (random.random()>0.5):
            y = y[:,::-1]
            x = x[:,::-1]
        if not self.waldo and np.sum(y)!=0:
            return None
        return x, to_categorical(y, num_classes=2).reshape((y.shape[0] * y.shape[1], 2))

    def __next__(self):
        idxs = self.idx_gen.__next__()
        items = []
        for idx in idxs:
            item = self.get_item(idx)
            if item is not None:
                items.append(item)
        if not items:
            return None
        xs,ys = zip(*tuple(items))
        return np.stack(xs), np.stack(ys)

freq0 = np.sum(labels==0)
freq1 = np.sum(labels==1)

print(freq0, freq1)

sns.distplot(labels.flatten(), kde=False, hist_kws={'log':True})

sample_weights = np.zeros((6, PANNEL_SIZE * PANNEL_SIZE, 2))

sample_weights[:,:,0] = 1. / freq0
sample_weights[:,:,1] = 1.

plt.subplot(1,2,1)
plt.imshow(sample_weights[0,:,0].reshape((224, 224)))
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(sample_weights[0,:,1].reshape((224, 224)))
plt.colorbar()


def build_model():
    inputs = layers.Input(shape=(PANNEL_SIZE, PANNEL_SIZE, 3))

    net = layers.Conv2D(64, kernel_size=3, padding='same')(inputs)
    net = layers.LeakyReLU()(net)
    net = layers.MaxPool2D(pool_size=2)(net)

    shortcut_1 = net

    net = layers.Conv2D(128, kernel_size=3, padding='same')(net)
    net = layers.LeakyReLU()(net)
    net = layers.MaxPool2D(pool_size=2)(net)

    shortcut_2 = net

    net = layers.Conv2D(256, kernel_size=3, padding='same')(net)
    net = layers.LeakyReLU()(net)
    net = layers.MaxPool2D(pool_size=2)(net)

    shortcut_3 = net

    net = layers.Conv2D(256, kernel_size=1, padding='same')(net)
    net = layers.LeakyReLU()(net)
    net = layers.MaxPool2D(pool_size=2)(net)

    net = layers.UpSampling2D(size=2)(net)
    net = layers.Conv2D(256, kernel_size=3, padding='same')(net)
    net = layers.Activation('relu')(net)

    net = layers.Add()([net, shortcut_3])

    net = layers.UpSampling2D(size=2)(net)
    net = layers.Conv2D(128, kernel_size=3, padding='same')(net)
    net = layers.Activation('relu')(net)

    net = layers.Add()([net, shortcut_2])

    net = layers.UpSampling2D(size=2)(net)
    net = layers.Conv2D(64, kernel_size=3, padding='same')(net)
    net = layers.Activation('relu')(net)

    net = layers.Add()([net, shortcut_1])

    net = layers.UpSampling2D(size=2)(net)
    net = layers.Conv2D(2, kernel_size=1, padding='same')(net)

    net = layers.Reshape((-1, 2))(net)
    net = layers.Activation('softmax')(net)

    model = Model(inputs=inputs, outputs=net)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(),
        metrics=['acc'],
        sample_weight_mode='temporal'
    )
    return model

model = build_model()
# model.summary()

gen_mix = seg_gen_mix(waldo_sub_imgs, waldo_sub_labels, imgs, labels, tot_bs=6, prop=0.34, out_sz=(PANNEL_SIZE, PANNEL_SIZE))

def on_epoch_end(epoch, logs):
    print('\r', 'Epoch:%5d - loss: %.4f - acc: %.4f' % (epoch, logs['loss'], logs['acc']), end='')

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

history = model.fit_generator(
    gen_mix, steps_per_epoch=6, epochs=500,
    class_weight=sample_weights,
    verbose=0,
    callbacks=[
        print_callback,
        ReduceLROnPlateau(monitor='loss', factor=0.2, patience=100, verbose=1, mode='auto', min_lr=1e-05)
    ]
)

model.save('model.h5')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title('loss')
plt.plot(history.history['loss'])
plt.subplot(1, 2, 2)
plt.title('accuracy')
plt.plot(history.history['acc'])



def bbox_from_mask(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return x1, y1, x2, y2

x1, y1, x2, y2 = bbox_from_mask((pred_out > 0.8).astype(np.uint8))
print(x1, y1, x2, y2)

# make overlay
overlay = np.repeat(np.expand_dims(np.zeros_like(pred_out, dtype=np.uint8), axis=-1), 3, axis=-1)
alpha = np.expand_dims(np.full_like(pred_out, 255, dtype=np.uint8), axis=-1)

overlay = np.concatenate([overlay, alpha], axis=-1)

overlay[y1:y2, x1:x2, 3] = 0

plt.figure(figsize=(20, 10))
plt.imshow(overlay)

fig, ax = plt.subplots(figsize=(20, 10))

ax.imshow(test_img)
ax.imshow(overlay, alpha=0.5)

rect = patches.Rectangle((x1, y1), width=x2-x1, height=y2-y1, linewidth=1.5, edgecolor='r', facecolor='none')
ax.add_patch(rect)

ax.set_axis_off()

fig.savefig(os.path.join('test_result', img_filename), bbox_inches='tight')