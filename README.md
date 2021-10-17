# vision_transformers

This is my personnal repo to implement new transofrmers based and other computer vision DL models

I am currenlty working without a lot of GPU ressources therefore I mainly trained models on CIFAR 10. But my implementation are build to be fast and effective at scale.

Current paper implemented:

* An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, from [Dosovitskiy et al](https://arxiv.org/abs/2010.11929) (2020)
* Patch Are All You Need ? [anonymous](https://openreview.net/forum?id=TVHS5Y4dNvM)

Baseline:

* Deep Residual Learning for Image Recognition, from [He et al](https://arxiv.org/abs/1512.03385) (2015)


Models are implemented in pure pytorch and trained via pytorchlightning. Dependencies are managed by poetry. It is included an Dockerfile to create a cuda ready container with jupyter lab inside.
On the development part, I use jupytext in order to avoid commit every metadata change on the notebook. Fully tested with pytest and formatted with black and isort.

If you want to create a project with similar config, just use my [boilerplat](https://github.com/samsja/pytorch-boilerplate). 

## How to use it ?

first install the dependecies:

```shell
poetry install
```

Then, only for development: 

add the precommit hook

```
poetry run pre-commit install
```

sync the notebook (only once)

```
poetry shell
make notebook-sync
```

## launch a jupyter lab session

```shell
poetry run jupyter lab
```

## Use tensorboard

```shell

poetry shell
make tensorboard
```

## Format the code without the precommit hook

```shell
poetry shell
make formatting
```

## Tests:

to run the tests:

```shell
poetry shell
make tests
```



