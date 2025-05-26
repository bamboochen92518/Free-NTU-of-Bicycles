
#
# Edited by: Jingwei Xu, ShanghaiTech University
# Based on the code from: https://github.com/graphdeco-inria/gaussian-splatting
#

from scene.dataset_readers.pandaset import readPandasetInfo


sceneLoadTypeCallbacks = {
    "Pandaset": readPandasetInfo,
}
