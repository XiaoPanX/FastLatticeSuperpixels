from scipy.ndimage import imread
from skimage.color import rgb2lab
from skimage.util import img_as_float
from skimage import io
from modelsA.Spixel_trans_feature import SpixelTransNet
from pixelfeaturecuda.pixelfeature import PixelFeatureFunction
from superpixelcolorcuda.superpixelcolor import SuperpixelColorFunction


input_transform1 = transforms.Compose([
    flow_transforms.ArrayToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
    transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
])
p_scale = 0.4
color_scale = 0.26
s_h=2
s_w=2
s_l=3

 modelA = SpixelTransNet(batchNorm=True)
  modelA.load_weights_from_pkl('BSDweight.pkl')
  modelA = modelA.cuda()
  modelA.eval()
image = img_as_float(io.imread(imgname))
img = rgb2lab(image)
img = input_transform1(img )
img_width = img.shape[3]
img_height = img.shape[2]
num_spixels_w = int(np.floor(img_width /(s_w*2^(s_l-1))))
num_spixels_h = int(np.floor(img_height / (s_h*2^(s_l-1))))
pos_scale_w = (1.0 * num_spixels_w) / (float(p_scale) * img_width)
pos_scale_h = (1.0 * num_spixels_h) / (float(p_scale) * img_height)
pos_scale = np.max([pos_scale_h, pos_scale_w])
img_trans = PixelFeatureFunction.apply(img, pos_scale, color_scale) 
outputL = modelA(img_trans, s_h,s_w,s_l) #get superpixel label
img_supcolor = SuperpixelColorFunction.apply(imgL, outputL, s_h,s_w,s_l)# the color of superpixels which is got by averaging the color of pixels in each superpxiel.
