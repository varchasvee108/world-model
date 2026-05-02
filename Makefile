# Makefile for World Model project

.PHONY: train infer

train:
	python -m scripts.train

infer:
	python -m scripts.infer
