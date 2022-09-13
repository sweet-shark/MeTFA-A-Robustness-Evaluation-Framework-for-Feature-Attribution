# MeTFA: A Robustness Evaluation Framework for Feature Attribution
## Description
This is the implementation the paper: “Is your explanation stable?”: A Robustness Evaluation Framework for Feature Attribution
https://arxiv.org/abs/2209.01782

MeTFA is a technique to decrease and quantify the randomness in the feature attribution algorithms.
In this repository, we show how to use the codes to generate the MeTFA-significant map, the upper bound map, the lower bound map and the MeTFA-smoothed map. Then, we show how to evaluate the stability of the explanation with $std$ and the faithfulness with (robust) insertion, (robust) deletion and (robust) overall. We show how to use the codes with 3 examples as follows.

To visualize MeTFA for LIME, you can run
```
python test_exp.py --base_explanation LIME --img_num 0001
```

To show the stability of MeTFA and SmoothGrad for Gradient, you can run
```
python test_stability.py --base_explanation Grad --outer_noise Uniform
```

To show the faithfulness of MeTFA for RISE with vanilla metrics and robust metrics, you can run the following codes, respectively.
```
python test_faithfulness.py --base_explanation RISE --metric vanilla
```
```
python test_faithfulness.py --base_explanation RISE --metric robust --outer_noise Uniform
```


## Requirements:
Pytorch,

lime: ```pip install lime``` 

jenkspy: ```pip install jenkspy```

scipy: ```pip install scipy```

## Contact:
If you have any questions, feel free to open an issue or directly contact me via: ganyuyou@zju.edu.cn
