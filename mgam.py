import os
import torch
import pandas as pd

from .dataset_modules.dataset_generic import Generic_MIL_Dataset


class mgam(Generic_MIL_Dataset):
    def getlabel(self, ids):
        self.slide_data: pd.DataFrame
        non_label = self.slide_data.iloc[ids].drop(columns=['slide_id'])
        label = non_label.loc[:, ~non_label.columns.str.contains('标签')]
        return label
        
    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.getlabel(idx)
        if type(self.data_dir) == dict:
            source = self.slide_data['source'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if not self.use_h5:
            if self.data_dir:
                full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))
                features = torch.load(full_path)
                return features, label
            
            else:
                return slide_id, label

        else:
            full_path = os.path.join(data_dir,'h5_files','{}.h5'.format(slide_id))
            with h5py.File(full_path,'r') as hdf5_file:
                features = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:]

            features = torch.from_numpy(features)
            return features, label, coords
