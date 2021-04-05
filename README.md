# Traditional Chinese Lyrics Generation by GPT-2

The repostiroy enables the generation of Traditional Chinese lyrics. The model behind the repostitory is powered by GPT-2. Given the user provided input pre-text, the model will generate a sheet of lyric.

## Samples

Input:

```
在這個美麗的時刻
```

Output:

```
在這個美麗的時刻
是否還留在我心田
每當你齣現
有些事你還不明瞭
我真的愛你你不明瞭
是誰說你不願意
每當我齣現
有些事你還不明瞭
我真的愛你你不明瞭
```

## Prerequisites

Please install the required Python packages via the following command:

```bash
pip3 install -r requirements.txt
```

## Use the Pre-Trainined Models

* Run the following command to invoke the inference script. The script runs a infinite loop to ask you to provide the pre-text, and the model will output its generated lyric.

```bash
python3 inference.py /PATH/TO/MODEL bert-base-chinese
```

## Steps to Run the Training Process

* Create a folder for the lyrics dataset.

```bash
mkdir -p lyrcis_dataset
```
* Download the zip file of lyrics raw data from [Chinese-Lyric-Corpus](https://github.com/gaussic/Chinese-Lyric-Corpus).

```bash
wget https://github.com/gaussic/Chinese-Lyric-Corpus/raw/master/Chinese_Lyrics.zip -O lyrcis_dataset/Chinese_Lyrics.zip
```

* Unzip the zip file.

```bash
unzip _lyrcis_dataset/Chinese_Lyrics.zip -d _lyrcis_dataset
```

* Run the dataset preparation script.

```bash
python3 prepare_dataset.py lyrcis_dataset/Chinese_Lyrics
```

* Run the training script. The script may take hours for processing. (If OOM is occured, please modify the values of `per_device_train_batch_size` and `per_device_eval_batch_size` to be less than 4)

```bash
python3 train.py \
  --model_name_or_path ckiplab/gpt2-base-chinese \
  --tokenizer_name bert-base-chinese \
  --train_file ./lyrcis_dataset/train.txt \
  --validation_file ./lyrcis_dataset/val.txt \
  --per_device_train_batch_size=4 \
  --per_device_eval_batch_size=4 \
  --do_train \
  --do_eval \
  --output_dir test-clm
```

* After the training is finished, the trained model will be saved in `test-clm`. You can use the model as described in [Use the Pre-Trainined Models](#use-the-pre-trainined-models)
