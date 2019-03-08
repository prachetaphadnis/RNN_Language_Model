# RNN_Language_Model
   This code implements a simple Recurrent Neural Network based language model based on [this paper](https://www.isca-speech.org/archive/interspeech_2010/i10_1045.html)
   by T. Mikolov et. al. and [this paper](https://arxiv.org/abs/1611.01368) by Linzen et.al.
   The model uses stochastic gradient descent and backpropagation through time.
   - The number of epochs can be set to the desired number (default = 10). The model runs until the number of maximum epochs or until convergence.
   - The train-lm mode is used for training the language model and test-lm is used for making predictions on the test set.
   - The model also implements a supervised verb number prediction model which can be trained using train-np and predictions can be made using np-test mode.
   - Unsupervised version of the verb number prediction model can be accessed using predict-lm (uses train-lm for training)
   - To study the effects of syntactic structure and to anaylse the extent to which a simple RNN can learn long distance syntax dependencies, the code also implements a noun only model, where only nouns in the sentence re passed as input and ver number predictions are made based on this lerning. This is loosely based on [noun baseline](https://arxiv.org/abs/1611.01368) by Linzen et. al.
     This can be accessed using train-np-noun and test-np-noun modes respectively.
   
# References
   - Tomas Mikolov, Martin Karafiat, Lukas Burget, Jan Cernock ´ y, and Sanjeev Khudanpur. Recurrent neural network based language model. In INTERSPEECH, volume 2, page 3, 2010.
   - Tal Linzen, Emmanuel Dupoux, and Yoav Goldberg. Assessing the ability of LSTMs to learn syntax-sensitive dependencies. Transactions of the Association for Computational Linguistics, 4:521–535, 2016.
   - Jiang Guo. Backpropagation Through Time. Unpubl. ms., Harbin Institute of Technology, 2013.
   - Kristina Gulordava, Piotr Bojanowski, Edouard Grave, Tal Linzen, and Marco Baroni. Colorless green recurrent networks dream hierarchically. In Proceedings of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2018.
