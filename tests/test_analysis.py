import unittest
import numpy as np
import torch
from analysis import features,TinyCNN,Patches
class ImageChecks(unittest.TestCase):
 def test_feature_dimensions_and_finiteness(self):
  images=np.zeros((2,64,64,3),dtype=np.uint8);images[1]=255
  f=features(images);self.assertEqual(f.shape,(2,18));self.assertTrue(np.isfinite(f).all())
  np.testing.assert_allclose(f[:,:3],[[0,0,0],[1,1,1]])
  np.testing.assert_allclose(f[:,6:16].sum(axis=1),1)
 def test_image_scaling_and_logits(self):
  images=np.full((2,64,64,3),255,dtype=np.uint8)
  x,y=Patches(images,np.array([0,8]))[1];self.assertEqual(y,8);self.assertEqual(x.shape,(3,64,64));self.assertEqual(float(x.max()),1.)
  output=TinyCNN()(torch.stack([x,x]));self.assertEqual(output.shape,(2,9));self.assertTrue(torch.isfinite(output).all())
if __name__=='__main__':unittest.main()
