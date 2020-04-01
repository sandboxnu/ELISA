from __future__ import print_function
import binascii
import struct
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import imageio


NUM_CLUSTERS = 5

print('reading image')
im = Image.open('../images/plate_black_1.jpeg')
im = im.resize((150, 150))      # optional, to reduce time
ar = np.asarray(im)
shape = ar.shape
ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

# print('finding clusters')
codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
# print('cluster centres:\n', codes)

vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
counts, bins = np.histogram(vecs, len(codes))    # count occurrences

index_max = np.argmax(counts)                    # find most frequent
peak = codes[index_max]
peak = tuple([int(round(num)) for num in peak][::-1])
print (peak)
# colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
# print('most frequent is %s (#%s)' % (peak, colour))

'''
# save the reduced-size image with only the N most-frequent colours
c = ar.copy()
for i, code in enumerate(codes):
    c[scipy.r_[np.where(vecs==i)],:] = code
imageio.imwrite('clusters.png', c.reshape(*shape).astype(np.uint8))
print('saved clustered image')


from colorthief import ColorThief
color_thief = ColorThief("../images/plate_black_1.jpeg")
# get the dominant color
dominant_color = color_thief.get_color(quality=1)
print("Color thief")
print(dominant_color)
'''
