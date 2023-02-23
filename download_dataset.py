import os
import requests
from tqdm import tqdm
from absl import flags, app


FLAGS = flags.FLAGS

flags.DEFINE_string('data_path', 'gpt2_text_document', 'data path')
flags.DEFINE_string('datasets', None, 'datasets')
flags.DEFINE_enum('dtype', 'base', ['base', 's', 'm', 'l', 'xl', 'a'], 'datasets')

def main(_):
  if not os.path.exists(FLAGS.data_path):
    os.makedirs(FLAGS.data_path)

  if FLAGS.datasets:
    datasets = FLAGS.datasets.split(',')
  else:
    if FLAGS.dtype == 'a':
      datasets = [
        'webtext', 
        'small-117M',  'small-117M-k40',
        'medium-345M', 'medium-345M-k40',
        'large-762M',  'large-762M-k40',
        'xl-1542M',    'xl-1542M-k40',
      ]
    elif FLAGS.dtype == 's':
      datasets = [
        'small-117M',  'small-117M-k40',
      ]
    elif FLAGS.dtype == 'm':
      datasets = [
        'medium-345M', 'medium-345M-k40',
      ] 
    elif FLAGS.dtype == 'l':
      datasets = [
        'large-762M',  'large-762M-k40',
      ]
    elif FLAGS.dtype == 'xl':
      datasets = [
        'xl-1542M',    'xl-1542M-k40',
      ]
    else:
      datasets = ['webtext']
  
  for ds in datasets:
    for split in ['train', 'valid', 'test']:
      filename = ds + "." + split + '.jsonl'
      r = requests.get("https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/" + filename, stream=True)

      with open(os.path.join(FLAGS.data_path, filename), 'wb') as f:
        file_size = int(r.headers["content-length"])
        chunk_size = 1000
        with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
          # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
          for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(chunk_size)


if __name__ == '__main__':
  app.run(main)
