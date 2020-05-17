import numpy as np
from nnmnkwii.io import hts
from os.path import join
from glob import glob
from tqdm import tqdm


paths = sorted(glob(join('../data/basic5000', "label_phone_align", "*.lab")))

for i, filepath in tqdm(enumerate(paths)):
    label = hts.load(filepath)


    end_times = label.end_times
    end_index = np.array(end_times) / 50000


    mora_index = np.array([0]*int(end_index[-1]))

    mora_index = [1 if i in list(end_index.astype(int)) else 0 for i in range(end_index[-1].astype(int))]
    indices = label.silence_frame_indices().astype(int)
    mora_index = np.delete(mora_index, indices, axis=0)


    np.savetxt('../data/basic5000/mora_index/mora_index_'+ '0'*(3-len(str(i+1))) + str(i+1) + '.csv', mora_index)
