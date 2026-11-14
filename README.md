# MolGPT
In this work, we train small custom GPT on Moses and Guacamol dataset with next token prediction task. The model is then used for unconditional and conditional molecular generation. We compare our model with previous approaches on the Moses and Guacamol datasets. Saliency maps are obtained for interpretability using Ecco library.

- The processed Guacamol and MOSES datasets in csv format can be downloaded from this link:

https://drive.google.com/drive/folders/1LrtGru7Srj_62WMR4Zcfs7xJ3GZr9N4E?usp=sharing

已经不可用，需要下载原始数据，对数据做简单预处理（基于rdkit计算分子属性和骨架）

- Original Guacamol dataset can be found here:

https://github.com/BenevolentAI/guacamol

- Original Moses dataset can be found here:

https://github.com/molecularsets/moses

- All trained weights can be found here:

https://www.kaggle.com/virajbagal/ligflow-final-weights


To train the model, make sure you have the datasets' csv file in the same directory as the code files.

# Training

```
./train_moses.sh
```

```
./train_guacamol.sh
```

# Generation

```
./generate_guacamol_prop.sh
```

```
./generate_moses_prop_scaf.sh
```

If you find this work useful, please cite:

Bagal, Viraj; Aggarwal, Rishal; Vinod, P. K.; Priyakumar, U. Deva (2021): MolGPT: Molecular Generation using a Transformer-Decoder Model. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.14561901.v1 



## Experiment logging (SwanLab)

Install the logger into the active environment and log in for cloud tracking:

```bash
uv pip install swanlab
swanlab login
bash train_moses.sh
```

Training records hyperparameters, batch loss, learning rate, and epoch training/validation
losses under the `lig_gpt` project. For offline training without login:

```bash
python train/train.py --run_name unconditional_moses --data_name moses2 --num_props 0 --swanlab_mode offline
```

Use `--swanlab_mode disabled` to disable logging.
