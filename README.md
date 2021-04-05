# Traditional Chinese Lyrics Generation by GPT-2

The repostiroy enables the generation of Traditional Chinese lyrics. The model behind the repostitory is powered by GPT-2. Given the user provided input pre-text, the model will generate a sheet of lyric.

## Samples

Input 1:

```
在這個美麗的時刻
```

Output 2:

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

Input 2:
```
親愛
```

Output 2:
```
親愛的方嚮
如果再不擁有你
無意義地一個人去漂流
就一路走下去看過去
就這樣我們曾經擁有
從此你就讓我看見一個迴憶
曾經你給我的愛留下的痕跡

讓我愛你

就從今天起  我就是今天
..

我們曾經擁有  你曾經愛過我
也曾有一段沒有離開的日子
但如今你卻不在我身邊
我一定會找迴今天的每一份心情


曾經你給我的愛留下的痕跡
給我的愛留下的痕跡

讓我愛你


曾經你給我的愛留下的痕跡
給我的愛留下的痕跡
隻是我承受瞭傷害

讓我愛你

讓我愛你

把我們的心念  我一輩子保存
就在這裏  從今以後

..
讓我愛你

曾經你給我的愛留下的痕跡

給我的愛讓我 一輩子保存
就在這裏  從今以後
讓我愛你<>
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
