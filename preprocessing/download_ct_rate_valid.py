import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download


if __name__ == "__main__":

    split = 'valid_fixed'
    batch_size = 100
    start_at = 0

    repo_id = 'ibrahimhamamci/CT-RATE'
    directory_name = f'dataset/{split}/'
    hf_token = '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'

    data = pd.read_csv(rf"C:\Users\20173869\Downloads\{split}_predicted_labels.csv")

    for i in tqdm(range(start_at, len(data), batch_size)):

        data_batched = data[i:i+batch_size]

        for name in data_batched['VolumeName']:
            folder1 = name.split('_')[0]
            folder2 = name.split('_')[1]
            folder = folder1 + '_' + folder2
            folder3 = name.split('_')[2]
            subfolder = folder + '_' + folder3
            subfolder = directory_name + folder + '/' + subfolder

            hf_hub_download(repo_id=repo_id,
                repo_type='dataset',
                token=hf_token,
                subfolder=subfolder,
                filename=name,
                cache_dir=r'E:\Datasets\CT-RATE\.cache',
                local_dir=r'E:\Datasets\CT-RATE',
                local_dir_use_symlinks=False,
                resume_download=True,
                )
