# DeepComment
___DeepComment___ is a method to recommend developers regarding appropriate commenting locations in the source code. Commenting is closely related to the code syntax and semantics, hence we adopt neural language model (word embeddings) to capture the code semantic information, and analyze the abstract syntax trees to capture code syntactic information.
## Requirements
  >__Tensorflow__ : 1.10.0  
>__Numpy__ : 1.15.2  
>__Pandas__ : 0.23.4  
>__sklearn__ : 0.20.0  
>__gensim__ : 3.4.0  
## Usage
  The `deepComment/deepit_varlength.py` file is used to train the prediction model. The dataset is in the fold `data/`. And the trained model is in the
  folder `model/`. the trainning results on valid&test datasets are in the folder `testresult/`.   
  
The test_code_len.py file is used to analyze code lines effect on the performance of DeepComment , the test_comment_len.py file is used to analyze comment amount effect on the performance of DeepComment. Their results are in the folder testresult/.

  
  The files in `originData/` are the origin data of the experiment. The  `code.txt` is all of the code lines with line numbers, 
  the  `token.txt` is tokens extracted by JDT, the `startline.txt` and  `endline.txt` are mean the code snippets' start lines and end lines.
