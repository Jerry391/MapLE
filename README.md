# MapLE: Matching Molecular Analogues Promptly with Low Computational Resources by Multi-Metrics Evaluation
This general strategy aims to promptly match analogous molecules with low computational resources by multi-metrics evaluation.

Our work is published in AAAI-2024 conference. If you want to know the main ideas and results of the work, please read this student abstract published on AAAI-2023, which is only two pages.

## Usage
### Getting Raw Data
Please go to the official website of PDBbind dataset to download raw data([Welcome to PDBbind-CN database](http://pdbbind.org.cn/casf.php)), the CASF-2016 version is used in this work. Then place the downloaded data in the project's"mol_data/pre_process" directory. 

Due to the large size of the original data set, this project provides some samples as a demo in the project's "mol_data/sample" directory.

### Preprocess
Use the pre_process.py file to process the raw data.

After processing, we can get multi-metrics inverted lists which are used in the next step.

### Progressive prompt evaluation
After setting the dataset directory, please directly use the match.py file to match analogous molecules.


## Reference:
If you find our work useful, please consider citing:

``` 
@Article{Chen2024,
  author  = {Chen, Xiaojian and Liao, Chuyue and Gu, Yanhui and Li, Yafie and Wang, Jinlan and Chen, Yi and Kitsuregawa Masaru},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  title   = {{MapLE}: Matching Molecular Analogues Promptly with Low Computational Resources by Multi-Metrics Evaluation},
  year    = {2024},
}
```
