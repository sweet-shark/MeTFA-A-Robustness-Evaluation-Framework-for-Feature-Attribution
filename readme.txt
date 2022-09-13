The vanilla explanation algorithms of IGOS, RISE, Gradient and ScoreCAM are shown in IGOS_generate_video.py, RISE.py, Gradient.py and cam.py, respectively.
For LIME, you can run
	pip install lime.
to install it.

The core parts of MeTFA are shown in ST_wo_thre.py

To reproduce the experiment, please run 
	python test_Grid.py --model {resnet50, vgg16, densenet169} --outer_noise {Uniform, Normal, Darken} --inner_noise {Uniform, Normal, Darken} --base_explanation {ScoreCAM, IGOS, RISE, Gradient, LIME}

For example, python test_Grid.py --model resnet50 --outer_noise Uniform --inner_noise Uniform --base_explanation IGOS

Then run 
	python test_stability.py
to show the results of stability and run
	python test_faithfulness.py
to show the results of faithfunlness.
